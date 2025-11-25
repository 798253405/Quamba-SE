"""
Scale Enhancement (SE) versions of selective scan for research.
These are pure PyTorch implementations based on Mamba's selective_scan_ref.
"""
import torch
import torch.nn.functional as F
from einops import rearrange


def selective_scan_SE_float(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                             return_last_state=False):
    """
    SE Mode 1: FP32 SSM Input (Upper Bound)
    Keep everything in FP32 for theoretical upper bound performance.

    Args:
        u: r(B D L)  - SSM input (from conv1d)
        delta: r(B D L)
        A: c(D N) or r(D N)
        B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        D: r(D)
        z: r(B D L)
        delta_bias: r(D), fp32

    Returns:
        out: r(B D L)
        last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            from einops import repeat
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        from einops import repeat
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)

    # CUDA kernel returns (B, L, D), so transpose to match
    out = out.transpose(1, 2)  # (batch, dim, L) -> (batch, L, dim)

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_SE_int8Torch(u_int8, u_scale, delta_int8, delta_scale, A, A_scale, B_int8, B_scale, C_int8, C_scale,
                                 ssm_state_scale=None,
                                 D_int8=None, D_scale=None, z_int8=None, z_scale=None,
                                 delta_bias_int8=None, delta_bias_scale=None,
                                 delta_softplus=False, return_last_state=False):
    """
    SE Mode 2-1/2-3: PyTorch INT8 SSM - CUDA-Order Implementation
    Pure PyTorch implementation that exactly matches CUDA kernel computation order.

    **KEY CHANGES from previous version:**
    - Removed ALL einsum operations (einsum has undefined accumulation order)
    - Uses explicit loops and broadcasting to match CUDA kernel's sequential computation
    - Computes deltaA and deltaB_u per-timestep (not pre-computed for all timesteps)
    - Uses .sum() with explicit dim for controlled reduction order

    Mode 2-1: ssm_state_scale=None (no state quantization)
    Mode 2-3: ssm_state_scale=value (with state quantization, should match CUDA)

    All inputs are INT8, computation happens in FP32 after dequantization.

    Args:
        u_int8: INT8 tensor (B, D, L)
        u_scale: scale for u
        delta_int8: INT8 tensor (B, D, L)
        delta_scale: scale for delta
        A: FP32 tensor (D, N) - already A = -exp(A_log * A_scale)
        A_scale: scale for A (not used, A is already FP32)
        B_int8, C_int8: INT8 tensors
        B_scale, C_scale: scales
        ssm_state_scale: scale for SSM state quantization (Mode 2-3 only)
        D_int8, z_int8, delta_bias_int8: optional INT8 tensors

    Returns:
        FP32 output (B, L, D) matching CUDA kernel output format
    """
    # ========================================================================
    # 1. Dequantize inputs to FP32
    # ========================================================================
    u = u_int8.float() * u_scale              # (B, D, L)
    delta = delta_int8.float() * delta_scale  # (B, D, L)

    if delta_bias_int8 is not None:
        delta_bias = delta_bias_int8.float() * delta_bias_scale  # (D,)
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(2)     # (B, D, L)

    if delta_softplus:
        # Match CUDA implementation: delta <= 20 ? log1p(exp(delta)) : delta
        # This prevents overflow and matches CUDA's numerical behavior exactly
        delta = torch.where(delta <= 20.0, torch.log1p(torch.exp(delta)), delta)

    B = B_int8.float() * B_scale  # (D, N) or (B, N, L) or ...
    C = C_int8.float() * C_scale  # (D, N) or (B, N, L) or ...

    if D_int8 is not None:
        D = D_int8.float() * D_scale  # (D,)
    else:
        D = None

    if z_int8 is not None:
        z = z_int8.float() * z_scale  # (B, D, L)
    else:
        z = None

    # ========================================================================
    # 2. Setup dimensions and handle complex numbers
    # ========================================================================
    batch, dim, seqlen = u.shape[0], u.shape[1], u.shape[2]
    dstate = A.shape[1]

    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))

    # Handle variable B/C with groups
    if is_variable_B and B.dim() == 4:
        # B: (B, G, N, L) → (B, D, N, L)
        from einops import repeat
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])

    if is_variable_C and C.dim() == 4:
        # C: (B, G, N, L) → (B, D, N, L)
        from einops import repeat
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    # ========================================================================
    # 3. Main SSM Loop - Time Outer Loop (Fast Vectorized Version)
    # ========================================================================
    # Initialize state: x[b, d, n] for each batch, dim, dstate
    x = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)
    ys = []
    last_state = None

    # Process each timestep sequentially
    for i in range(seqlen):
        # ====================================================================
        # Step 3.1: Extract timestep i data
        # ====================================================================
        delta_i = delta[:, :, i]  # (B, D)
        u_i = u[:, :, i]          # (B, D)

        # ====================================================================
        # Step 3.2: Compute deltaA[b, d, n] = exp(delta[b, d, i] * A[d, n])
        # ====================================================================
        # delta_i: (B, D) → (B, D, 1)
        # A: (D, N) → (1, D, N)
        # Result: (B, D, N)
        deltaA_i = torch.exp(delta_i.unsqueeze(2) * A.unsqueeze(0))  # (B, D, N)

        # ====================================================================
        # Step 3.3: Compute deltaB_u[b, d, n] = delta[b,d,i] * B[d,n] * u[b,d,i]
        # ====================================================================
        if not is_variable_B:
            # B: (D, N) - constant across sequence
            # delta_i: (B, D) → (B, D, 1)
            # u_i: (B, D) → (B, D, 1)
            # B: (D, N) → (1, D, N)
            # Result: (B, D, N)
            deltaB_u_i = delta_i.unsqueeze(2) * B.unsqueeze(0) * u_i.unsqueeze(2)
        else:
            # B: (B, N, L) or (B, D, N, L)
            if B.dim() == 3:
                # B: (B, N, L)
                B_i = B[:, :, i]  # (B, N)
                # delta_i: (B, D, 1, 1), B_i: (B, 1, N), u_i: (B, D, 1)
                deltaB_u_i = delta_i.unsqueeze(2) * B_i.unsqueeze(1) * u_i.unsqueeze(2)
            else:
                # B: (B, D, N, L)
                B_i = B[:, :, :, i]  # (B, D, N)
                deltaB_u_i = delta_i.unsqueeze(2) * B_i * u_i.unsqueeze(2)

        # ====================================================================
        # Step 3.4: State update (SSM recurrence)
        # ====================================================================
        x = deltaA_i * x + deltaB_u_i  # (B, D, N)

        # ====================================================================
        # Step 3.5: Compute output y[b, d] = sum_n(x[b, d, n] * C[d, n])
        # ====================================================================
        if not is_variable_C:
            # C: (D, N) - constant across sequence
            # x: (B, D, N), C: (1, D, N)
            # Multiply element-wise, then sum over N dimension
            y_i = (x * C.unsqueeze(0)).sum(dim=2)  # (B, D)
        else:
            # C: (B, N, L) or (B, D, N, L)
            if C.dim() == 3:
                # C: (B, N, L)
                C_i = C[:, :, i]  # (B, N)
                # x: (B, D, N), C_i: (B, 1, N)
                y_i = (x * C_i.unsqueeze(1)).sum(dim=2)  # (B, D)
            else:
                # C: (B, D, N, L)
                C_i = C[:, :, :, i]  # (B, D, N)
                y_i = (x * C_i).sum(dim=2)  # (B, D)

        # Handle complex numbers (rare case)
        if y_i.is_complex():
            y_i = y_i.real * 2

        # Store output for this timestep
        ys.append(y_i)

        # Save last state if this is the final timestep
        if i == seqlen - 1:
            last_state = x

    # ========================================================================
    # 4. Stack outputs and apply D (skip connection) and z (gating)
    # ========================================================================
    y = torch.stack(ys, dim=2)  # (B, D, L)

    # Add skip connection (D)
    if D is not None:
        # D: (D,) → (1, D, 1)
        y = y + u * D.unsqueeze(0).unsqueeze(2)

    # Apply SiLU gating (z)
    if z is not None:
        y = y * F.silu(z)

    # ========================================================================
    # 5. Transpose to match CUDA output format: (B, D, L) → (B, L, D)
    # ========================================================================
    out = y.transpose(1, 2).contiguous()  # (B, L, D)

    # ========================================================================
    # 6. Return output (FP32, matching CUDA kernel)
    # ========================================================================
    return out if not return_last_state else (out, last_state)


def selective_scan_SE_mode22_fp32_replicates_mode21(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                                                      delta_softplus=False, return_last_state=False):
    """
    Mode 2-2: FP32 computation that exactly replicates Mode 2-1 logic

    This function uses IDENTICAL logic to selective_scan_SE_int8Torch (Mode 2-1),
    but with FP32 inputs instead of INT8 inputs.

    Purpose: Isolate quantization error by comparing:
    - Mode 2-1: INT8 data with INT8Torch logic
    - Mode 2-2: FP32 data with INT8Torch logic (this function)

    The ONLY difference from Mode 2-1 is precision (FP32 vs INT8).
    All computation logic is IDENTICAL.

    Args:
        u: FP32 tensor (B, D, L) - SSM input
        delta: FP32 tensor (B, D, L) - time step
        A: FP32 tensor (D, N) - already computed as -exp(A_log * A_scale)
        B: FP32 tensor (D, N) or (B, N, L) or (B, G, N, L)
        C: FP32 tensor (D, N) or (B, N, L) or (B, G, N, L)
        D: FP32 tensor (D,) - skip connection (optional)
        z: FP32 tensor (B, D, L) - gating (optional)
        delta_bias: FP32 tensor (D,) - bias for delta (optional)
        delta_softplus: bool - apply softplus to delta
        return_last_state: bool - return final state

    Returns:
        FP32 output (B, L, D) matching Mode 2-1 output format
    """
    # ========================================================================
    # 1. Process inputs (all already FP32, no dequantization needed)
    # ========================================================================
    # Unlike Mode 2-1, inputs are already FP32, so we skip the INT8→FP32 conversion

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(2)  # (B, D, L)

    if delta_softplus:
        # Match CUDA implementation: delta <= 20 ? log1p(exp(delta)) : delta
        delta = torch.where(delta <= 20.0, torch.log1p(torch.exp(delta)), delta)

    # ========================================================================
    # 2. Setup dimensions and handle complex numbers
    # ========================================================================
    batch, dim, seqlen = u.shape[0], u.shape[1], u.shape[2]
    dstate = A.shape[1]

    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))

    # Handle variable B/C with groups
    if is_variable_B and B.dim() == 4:
        # B: (B, G, N, L) → (B, D, N, L)
        from einops import repeat
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])

    if is_variable_C and C.dim() == 4:
        # C: (B, G, N, L) → (B, D, N, L)
        from einops import repeat
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    # ========================================================================
    # 3. Main SSM Loop - Time Outer Loop (Matches Mode 2-1 exactly)
    # ========================================================================
    x = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)
    ys = []
    last_state = None

    # Process each timestep sequentially
    for i in range(seqlen):
        # Extract timestep i data
        delta_i = delta[:, :, i]  # (B, D)
        u_i = u[:, :, i]          # (B, D)

        # Compute deltaA[b, d, n] = exp(delta[b, d, i] * A[d, n])
        deltaA_i = torch.exp(delta_i.unsqueeze(2) * A.unsqueeze(0))  # (B, D, N)

        # Compute deltaB_u[b, d, n] = delta[b,d,i] * B[d,n] * u[b,d,i]
        if not is_variable_B:
            deltaB_u_i = delta_i.unsqueeze(2) * B.unsqueeze(0) * u_i.unsqueeze(2)
        else:
            if B.dim() == 3:
                B_i = B[:, :, i]
                deltaB_u_i = delta_i.unsqueeze(2) * B_i.unsqueeze(1) * u_i.unsqueeze(2)
            else:
                B_i = B[:, :, :, i]
                deltaB_u_i = delta_i.unsqueeze(2) * B_i * u_i.unsqueeze(2)

        # State update (SSM recurrence)
        x = deltaA_i * x + deltaB_u_i  # (B, D, N)

        # Compute output y[b, d] = sum_n(x[b, d, n] * C[d, n])
        if not is_variable_C:
            y_i = (x * C.unsqueeze(0)).sum(dim=2)  # (B, D)
        else:
            if C.dim() == 3:
                C_i = C[:, :, i]
                y_i = (x * C_i.unsqueeze(1)).sum(dim=2)
            else:
                C_i = C[:, :, :, i]
                y_i = (x * C_i).sum(dim=2)

        # Handle complex numbers
        if y_i.is_complex():
            y_i = y_i.real * 2

        ys.append(y_i)

        if i == seqlen - 1:
            last_state = x

    # ========================================================================
    # 4. Stack outputs and apply D (skip connection) and z (gating)
    # ========================================================================
    y = torch.stack(ys, dim=2)  # (B, D, L)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(2)

    if z is not None:
        y = y * F.silu(z)

    # ========================================================================
    # 5. Transpose to match CUDA output format: (B, D, L) → (B, L, D)
    # ========================================================================
    out = y.transpose(1, 2).contiguous()  # (B, L, D)

    # ========================================================================
    # 6. Return output (FP32, matching Mode 2-1 format)
    # ========================================================================
    return out if not return_last_state else (out, last_state)


def selective_scan_SE_floatSimInt8(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                                    return_last_state=False, u_scale=1.0):
    """
    SE Mode 2-2: Float Simulation of INT8 (Verification)
    Simulate INT8 quantization behavior using FP32 arithmetic.
    Should produce identical results to INT8 baseline.

    Args:
        u_scale: Scale factor for simulating INT8 quantization of u
        Other args: same as selective_scan_SE_float

    Returns:
        Same as selective_scan_SE_float
    """
    dtype_in = u.dtype

    # Simulate INT8 quantization on u
    # u has already been quantized by conv1d: round(x/scale)*scale
    # Here we just use it as-is
    u = u.float()

    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            from einops import repeat
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        from einops import repeat
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)

    # CUDA kernel returns (B, L, D), so transpose to match
    out = out.transpose(1, 2)  # (batch, dim, L) -> (batch, L, dim)

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_SE_floatSimASIC_SoftEdge(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                                             return_last_state=False, scale_factor=2025.0, u_scale=1.0):
    """
    SE Mode 3: Float Simulation ASIC with Scale Enhancement (Research)
    Simulate INT8 quantization with dual-scale for outliers.

    Args:
        scale_factor: Enhancement factor for scale (default: 2025)
        u_scale: Original scale factor for u
        Other args: same as selective_scan_SE_float

    Returns:
        Same as selective_scan_SE_float
    """
    dtype_in = u.dtype

    # Check if u has dual-scale metadata attached by Conv1D
    if hasattr(u, '_dual_scale_overflow_mask'):
        # Dequantize with dual-scale
        overflow_mask = u._dual_scale_overflow_mask
        scale1 = u._dual_scale_scale1
        scale2 = u._dual_scale_scale2

        # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        num_outliers = overflow_mask.sum().item()
        print(f"\n[SSM DEBUG] Dual-scale dequantization:")
        print(f"  Outliers: {num_outliers} / {overflow_mask.numel()}")
        print(f"  Scale1: {scale1}, Scale2: {scale2}")
        print(f"  u (INT8) first 20 values: {u.flatten()[:20].tolist()}")
        # ===== END TEMPORARY DEBUG CODE =====

        # u contains INT8 values (stored as FP32)
        # Dequantize based on mask
        u_dequant = torch.zeros_like(u)
        u_dequant[~overflow_mask] = u[~overflow_mask] * scale1  # Inliers
        u_dequant[overflow_mask] = u[overflow_mask] * scale2    # Outliers

        # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        print(f"  u (dequantized) first 20 values: {u_dequant.flatten()[:20].tolist()}")
        print(f"  u (dequantized) outlier values (first 10): {u_dequant.flatten()[overflow_mask.flatten()][:10].tolist()}\n")
        # ===== END TEMPORARY DEBUG CODE =====

        u = u_dequant.float()
    else:
        # No dual-scale metadata, use as-is
        u = u.float()

    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            from einops import repeat
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        from einops import repeat
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)

    # CUDA kernel returns (B, L, D), so transpose to match
    out = out.transpose(1, 2)  # (batch, dim, L) -> (batch, L, dim)

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
