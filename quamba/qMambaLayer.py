import math
import copy
from functools import partial
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.modules.mamba_simple import Mamba

from .qActLayer import QAct, ActIdentity
from .qLinearLayer import W4A16B16O16Linear
from .qLinearLayer import W4A8B8O8Linear, W4A8B16O16Linear
from .qLinearLayer import W8A8B8O8Linear, W8A8B16O16Linear
from .qLinearLayer import HadLinear
from .qConvLayer import QCausalConv1D
from .qSelectiveScan import QSScan
from .qHadamard import Hadamard, QHadamard


class MambaSimple(nn.Module):
    def __init__(
        self,
        originalLayer: Mamba,
        use_had_transform: bool = True
    ):
        super().__init__()
        self.d_model = originalLayer.d_model
        self.d_state = originalLayer.d_state
        self.d_conv = originalLayer.d_conv
        self.expand = originalLayer.expand
        self.d_inner = originalLayer.d_inner
        self.dt_rank = originalLayer.dt_rank
        # self.use_fast_path = originalLayer.use_fast_path
        self.use_fast_path = False # DO NOT USE FAST PATH for quantization experiments
        self.layer_idx = originalLayer.layer_idx
        self.use_had_transform = use_had_transform
        
        # input proj
        if use_had_transform:
            self.in_proj = HadLinear(originalLayer.in_proj, input_transform=True, output_transform=False)
        else:
            self.in_proj = copy.deepcopy(originalLayer.in_proj)
        # causal conv
        self.conv1d = copy.deepcopy(originalLayer.conv1d)
        self.activation = "silu"
        self.act = nn.SiLU()
        # B, C, dt
        self.x_proj = copy.deepcopy(originalLayer.x_proj)
        self.dt_proj = copy.deepcopy(originalLayer.dt_proj)
        self.dt_proj.bias = None
        self.dt_proj_bias = originalLayer.dt_proj.bias.clone().float()
        # ascan
        self.A_log = copy.deepcopy(originalLayer.A_log)
        self.D = copy.deepcopy(originalLayer.D)
        self.ssm_state_act = ActIdentity(tensor_name="ssm_state_act")
        # output proj
        if use_had_transform:
            self.had = Hadamard(originalLayer.out_proj.in_features)
            self.out_proj = HadLinear(originalLayer.out_proj, input_transform=True, output_transform=True)
        else:
            self.had = nn.Identity()
            self.out_proj = copy.deepcopy(originalLayer.out_proj)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        
        #assert hidden_states.shape[0] == 1, "Current only support bsz=1"
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
            
        #NOTE(brian1009): Simplified of original implementation 
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L134
        xz = self.in_proj(hidden_states) #(B, L, 2*D)
        xz = rearrange(xz, "b l d -> b d l") 
        x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)
        
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

        x = self.conv1d(x)
        x = self.act(x[...,:seqlen])
        #NOTE(brian1009): 2024/03/27 Do not squeeze the dimension of (b l), to make sure that QAct will always take the input dimension of (b, l, d)
        x_reshape = rearrange(x, "b d l -> b l d")
        x_dbl = self.x_proj(x_reshape)  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        #NOTE(brian1009): Comment this line and do the inference directly with the forward in the module
        dt = self.dt_proj(dt)

        #NOTE(brian1009): 2024/03/27 Do not squeeze the dimension of (b l), to make sure that QAct will always take the input dimension of (b, l, d)
        dt = rearrange(dt, "b l d -> b d l", l=seqlen)
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        assert self.activation in ["silu", "swish"]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj_bias,
                # delta_bias=None,  # delta_bias has been added in dt_proj
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
            ssm_state = self.ssm_state_act(ssm_state)
        y = rearrange(y, "b d l -> b l d") 
        y = self.had(y)
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            w_quant, w_scales = self.conv1d.quant_weight
            x = torch.sum(conv_state * rearrange(w_quant, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        # Discretize A and B
        dt = F.softplus(dt+self.dt_proj_bias)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)  # (B D)

        y = self.had(y)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class W4A16QMamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_had_transform=True,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"

        self.in_proj = W4A16B16O16Linear(self.d_model, self.d_inner * 2, group_size=128)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = W4A16B16O16Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, group_size=128
        )
        # we seperate the bias, so we set bias=False here
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=False, **factory_kwargs)
        self.register_buffer("dt_proj_bias", torch.empty(
            self.d_inner, device=factory_kwargs["device"], dtype=torch.float32))

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj_bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        # output proj
        if use_had_transform:
            self.had = Hadamard(self.d_inner)
        else:
            self.had = nn.Identity()
        self.out_proj = W4A16B16O16Linear(self.d_inner, self.d_model, group_size=128)

    @classmethod
    def from_fp16(cls, originalLayer: Mamba, use_had_transform: bool = True):
        
        qmixer = cls(
            d_model=originalLayer.d_model,
            d_state=originalLayer.d_state,
            d_conv=originalLayer.d_conv,
            expand=originalLayer.expand,
            dt_rank=originalLayer.dt_rank,
            use_had_transform = use_had_transform,
            use_fast_path=False,  # Fused kernel options
            layer_idx=originalLayer.layer_idx,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
        
        # input proj, weight group_size=128
        qmixer.in_proj = W4A16B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
        )
        # causal conv
        qmixer.conv1d = copy.deepcopy(originalLayer.conv1d)
        qmixer.activation = "silu"
        qmixer.act = nn.SiLU()
        # B, C, dt
        qmixer.x_proj = W4A16B16O16Linear.from_fp16(copy.deepcopy(originalLayer.x_proj))
        # We use FP16 dt_proj, becuase w4a16o16 does not support M=bsize, K=48, N=1536
        qmixer.dt_proj = copy.deepcopy(originalLayer.dt_proj)
        qmixer.dt_proj_bias = originalLayer.dt_proj_bias.clone() # MambaSimple has separated bias 
        # ascan
        qmixer.A_log = copy.deepcopy(originalLayer.A_log)
        qmixer.D = copy.deepcopy(originalLayer.D)
        # output proj
        if use_had_transform:
            qmixer.had = Hadamard(originalLayer.out_proj.in_features)
        else:
            qmixer.had = nn.Identity()
        qmixer.out_proj = W4A16B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.out_proj),
        )
        return qmixer

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        
        #assert hidden_states.shape[0] == 1, "Current only support bsz=1"
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
            
        # xz = self.in_proj(hidden_states) #(B, 2*D, L)
        # xz = rearrange(xz, "b l d -> b d l")
        xz = self.in_proj.to_seqlen_last(hidden_states) #(B, 2*D, L)
        x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)
        
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

        assert self.activation in ["silu", "swish"]
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        )

        # we need a contiguous here for W4A16B16O16
        x_reshape = rearrange(x, "b d l -> (b l) d").contiguous()
        x_dbl = self.x_proj(x_reshape)  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        assert self.activation in ["silu", "swish"]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj_bias,
                # delta_bias=None,  # delta_bias has been added in dt_proj
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d") 
        y = self.had(y)
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        x = causal_conv1d_update(
            x,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt) # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        y = selective_state_update(
            ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj_bias, dt_softplus=True
        )
        y = self.had(y)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = torch.float16
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = torch.float16
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        # conv_dtype is torch.float16
        conv_dtype = torch.float16
        # ssm_dtype is torch.float16
        ssm_dtype = torch.float16
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=conv_dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=ssm_dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class W4A8QMamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_had_transform=True,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"

        self.in_proj = W4A8B8O8Linear(self.d_model, self.d_inner * 2, group_size=128)

        self.conv1d = QCausalConv1D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = W4A8B8O8Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, group_size=128
        )
        # we seperate the bias and put the bias in the QSScan
        self.dt_proj = W8A8B8O8Linear(self.dt_rank, self.d_inner)

        # Quantized selective scan
        self.selective_scan = QSScan(d_state=self.d_state, d_inner=self.d_inner, delta_softplus=True)

        # output proj
        if use_had_transform:
            self.had = QHadamard(self.d_inner, x_H_scale=1.0)
        else:
            self.had = QAct(scale=1.0)
        self.out_proj = W4A8B16O16Linear(self.d_inner, self.d_model, group_size=128)

    @classmethod
    def from_fp16(cls, originalLayer: MambaSimple, act_scales: Dict, use_had_transform: bool = True):

        qmixer = cls(
            d_model=originalLayer.d_model,
            d_state=originalLayer.d_state,
            d_conv=originalLayer.d_conv,
            expand=originalLayer.expand,
            dt_rank=originalLayer.dt_rank,
            use_had_transform = use_had_transform,
            use_fast_path=False,  # Fused kernel options
            layer_idx=originalLayer.layer_idx,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

        # input proj, weight group_size=128
        qmixer.in_proj = W4A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"],
            output_scale=act_scales["in_proj:output"],
        )

        # causal conv
        # no used, silu is fused in causal_conv1d
        qmixer.activation = "silu"
        assert qmixer.activation == "silu"
        qmixer.conv1d = QCausalConv1D.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.conv1d),
            input_scale=act_scales["in_proj:output"].item(),
            output_scale=act_scales["x_proj:input"].item(),            
        )

        # x_proj
        qmixer.x_proj = W4A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.x_proj),
            input_scale=act_scales["x_proj:input"],
            output_scale=act_scales["x_proj:output"],
        )

        # We use W8A8B8O8 dt_proj, becuase W4A8B8O8 does not support M=bsize, K=48, N=1536
        qmixer.dt_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.dt_proj),
            input_scale=act_scales["x_proj:output"].item(), # use x_proj_scale to avoid additional quantization operations
            output_scale=act_scales["dt_proj:output"].item(),
        )

        # ascan
        qmixer.selective_scan = QSScan.from_fp16(
            originalLayer.d_state, originalLayer.d_inner,
            originalLayer.A_log.clone(), D=originalLayer.D.clone(),
            dt_bias=originalLayer.dt_proj_bias.clone(), delta_softplus=True,
            ssm_state_scale=act_scales["ssm_state_act:input"],
            u_scale=act_scales["x_proj:input"],
            dt_scale=act_scales["dt_proj:output"],
            B_scale=act_scales["x_proj:output"],
            C_scale=act_scales["x_proj:output"],
            z_scale=act_scales["in_proj:output"],
        )

        # output proj
        if use_had_transform:
            qmixer.had.x_H_scale = act_scales["out_proj:input"].item()
        else:
            qmixer.had.scale = act_scales["out_proj:input"].item()
        qmixer.out_proj = W4A8B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.out_proj),
            input_scale=act_scales["out_proj:input"],
        )
        return qmixer

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        #assert hidden_states.shape[0] == 1, "Current only support bsz=1"
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = self.in_proj.to_seqlen_last(hidden_states) # (B, D, L) 
        x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)
        
        # Perform causal conv1d and return conv_state
        if conv_state is not None:
            # store quantized x into conv_state
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        x = self.conv1d.forward(x)

        # Check if FP32 SSM mode is enabled (requires requantization for x_proj)
        import os
        fp32_mode_enabled = (
            os.environ.get('FLOAT_SIM_ASIC_INT8', 'false').lower() == 'true' or
            os.environ.get('CONV1D_MODE24_FP32', 'false').lower() == 'true' or
            os.environ.get('CONV1D_MODE3_FP32', 'false').lower() == 'true'
        )

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # # Only debug last layer (layer_idx == 23 for 24-layer model)
        # DEBUG_ENABLED = self.layer_idx is not None and self.layer_idx == 23
        # if DEBUG_ENABLED:
        #     if not hasattr(self, '_debug_step_count'):
        #         self._debug_step_count = 0
        #     self._debug_step_count += 1
        #     # Only print first 3 calls for this layer
        #     if self._debug_step_count <= 3:
        #         print(f"\n{'='*80}")
        #         print(f"[Layer {self.layer_idx} Call #{self._debug_step_count}] After Conv1D")
        #         print(f"  x dtype: {x.dtype}, shape: {x.shape}")
        #         x_vals = x.flatten()[:10]
        #         print(f"  First 10 values (high precision):")
        #         for i in range(min(10, x_vals.numel())):
        #             print(f"    x[{i}] = {x_vals[i]:.12f}" if x.dtype == torch.float32 else f"    x[{i}] = {x_vals[i]}")
        #         if hasattr(x, '_dual_scale_overflow_mask'):
        #             print(f"  x has dual-scale metadata (outliers: {x._dual_scale_overflow_mask.sum().item()})")

        #         # Print all scales for this layer (first call only)
        #         if self._debug_step_count == 1:
        #             print(f"\n  === Layer {self.layer_idx} All Scales ===")
        #             print(f"  in_proj scales:")
        #             if hasattr(self.in_proj, 'input_scale'):
        #                 print(f"    input_scale: {self.in_proj.input_scale:.10f}")
        #             if hasattr(self.in_proj, 'weight_scale'):
        #                 print(f"    weight_scale: {self.in_proj.weight_scale:.10f}")
        #             if hasattr(self.in_proj, 'output_scale'):
        #                 print(f"    output_scale: {self.in_proj.output_scale:.10f}")

        #             print(f"  conv1d scales:")
        #             print(f"    input_scale: {self.conv1d.input_scale:.10f}")
        #             print(f"    weight_scale: {self.conv1d.weight_scale:.10f}")
        #             if hasattr(self.conv1d, 'bias_scale') and self.conv1d.bias_scale is not None:
        #                 print(f"    bias_scale: {self.conv1d.bias_scale:.10f}")
        #             print(f"    output_scale: {self.conv1d.output_scale:.10f}")

        #             print(f"  x_proj scales:")
        #             if hasattr(self.x_proj, 'input_scale'):
        #                 print(f"    input_scale: {self.x_proj.input_scale:.10f}")
        #             if hasattr(self.x_proj, 'weight_scale'):
        #                 print(f"    weight_scale: {self.x_proj.weight_scale:.10f}")
        #             if hasattr(self.x_proj, 'output_scale'):
        #                 print(f"    output_scale: {self.x_proj.output_scale:.10f}")

        #             print(f"  selective_scan scales:")
        #             if hasattr(self.selective_scan, 'u_scale'):
        #                 print(f"    u_scale: {self.selective_scan.u_scale:.10f}")
        #             if hasattr(self.selective_scan, 'dt_scale'):
        #                 print(f"    dt_scale: {self.selective_scan.dt_scale:.10f}")

        #             print(f"  out_proj scales:")
        #             if hasattr(self.out_proj, 'input_scale'):
        #                 print(f"    input_scale: {self.out_proj.input_scale:.10f}")
        #             if hasattr(self.out_proj, 'weight_scale'):
        #                 print(f"    weight_scale: {self.out_proj.weight_scale:.10f}")
        #             if hasattr(self.out_proj, 'output_scale'):
        #                 print(f"    output_scale: {self.out_proj.output_scale:.10f}")
        # # ===== END TEMPORARY DEBUG CODE =====

        # Convert Conv1D output for use in SSM
        # Different modes have different requirements for SSM input dtype
        if fp32_mode_enabled:
            # Check mode-specific environment variables
            ssm_use_pytorch_int8 = os.environ.get('SSM_USE_PYTORCH_INT8', 'false').lower() == 'true'
            conv1d_mode23_fp32 = os.environ.get('CONV1D_MODE23_FP32', 'false').lower() == 'true'

            if x.dtype == torch.int8:
                # x is INT8 from Conv1D (Mode 2-0, 2-1, 2-2, 2-3 when FLOAT_SIM_ASIC_INT8=true)
                x_for_xproj = x  # INT8 for x_proj

                # Mode 2-1: Keep INT8 for PyTorch INT8 SSM (direct pass, no dequant)
                # Mode 2-0, 2-2: Dequantize to FP32 for SSM
                if ssm_use_pytorch_int8 and not conv1d_mode23_fp32:
                    # Mode 2-1: PyTorch INT8 SSM expects INT8 input directly
                    x_for_ssm = x  # Keep INT8
                else:
                    # Mode 2-0 (CUDA INT8 SSM with requant), Mode 2-2 (FP32 SSM)
                    x_for_ssm = x.float() * self.conv1d.output_scale  # Dequantize to FP32
            elif x.dtype == torch.float32:
                # x is FP32 from Conv1D (Mode 2-3, 2-4)
                # Requantize to INT8 for x_proj, keep FP32 for SSM
                x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
                x_for_ssm = x  # Keep FP32 for SSM
            else:
                raise ValueError(f"Unexpected dtype in fp32_mode_enabled: {x.dtype}")

            # Compute dt, B, C using INT8
            x_reshape = rearrange(x_for_xproj, "b d l -> b l d").contiguous()
            x_dbl = self.x_proj(x_reshape)  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            # Compute dt proj with x_proj_scale
            dt = self.dt_proj.to_seqlen_last(dt.contiguous())
            B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()

            # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
            # if DEBUG_ENABLED and self._debug_step_count <= 3:
            #     print(f"  After dt_proj: dt dtype: {dt.dtype}, first 3: {dt.flatten()[:3].tolist()}")
            #     print(f"  B dtype: {B.dtype}, first 3: {B.flatten()[:3].tolist()}")
            #     print(f"  C dtype: {C.dtype}, first 3: {C.flatten()[:3].tolist()}")
            #     print(f"  z dtype: {z.dtype if z is not None else 'None'}, first 3: {z.flatten()[:3].tolist() if z is not None else 'N/A'}")
            # # ===== END TEMPORARY DEBUG CODE =====

            # Print Layer 24 SSM scales (before SSM forward) - only once
            if self.layer_idx == 23:
                if not hasattr(self, '_ssm_scales_printed'):
                    self._ssm_scales_printed = False

                if not self._ssm_scales_printed:
                    print(f"\n{'='*80}")
                    print(f"[Layer 24 / layer_idx {self.layer_idx}] SSM Scales")
                    print(f"{'='*80}")
                    print(f"  Location: qMambaLayer.py forward() - fp32_mode_enabled branch")
                    print(f"  ")
                    print(f"  SSM Input Data (before SSM.forward):")
                    print(f"    u dtype: {x_for_ssm.dtype}")
                    print(f"    u first 5 values [0,0,:5]: {x_for_ssm[0, 0, :5].tolist()}")
                    print(f"    dt first 5 values [0,0,:5]: {dt[0, 0, :5].tolist()}")
                    print(f"    B first 5 values [0,0,:5]: {B[0, 0, :5].tolist()}")
                    print(f"    C first 5 values [0,0,:5]: {C[0, 0, :5].tolist()}")
                    print(f"    dt/B/C dtype: {dt.dtype}")
                    print(f"  ")
                    print(f"  SSM Scales (from self.selective_scan / QSScan):")
                    print(f"    u_scale          = {self.selective_scan.u_scale.item():.10f}  (for SSM input u)")
                    print(f"    dt_scale         = {self.selective_scan.dt_scale.item():.10f}  (for dt)")
                    print(f"    B_scale          = {self.selective_scan.B_scale.item():.10f}  (for B)")
                    print(f"    C_scale          = {self.selective_scan.C_scale.item():.10f}  (for C)")
                    print(f"    A_scale          = {self.selective_scan.A_scale.item():.10f}  (for A)")
                    print(f"    D_scale          = {self.selective_scan.D_scale.item():.10f}  (for D)")
                    print(f"    z_scale          = {self.selective_scan.z_scale.item():.10f}  (for z)")
                    print(f"    ssm_state_scale  = {self.selective_scan.ssm_state_scale.item():.10f}  (for state)")
                    print(f"    dt_bias_scale    = {self.selective_scan.dt_bias_scale.item():.10f}  (for dt_bias)")
                    print(f"  ")
                    print(f"  ⚠️  Important: Conv1D output_scale should match SSM u_scale")
                    print(f"    (Conv1D output_scale printed above)")
                    print(f"    SSM u_scale = {self.selective_scan.u_scale.item():.10f}")
                    print(f"{'='*80}\n")
                    self._ssm_scales_printed = True

                    # Quick verification mode: exit after printing Layer 24 SSM scales
                    if os.environ.get('QUICK_VERIFY', 'false').lower() == 'true':
                        print("🔍 QUICK_VERIFY mode: Exiting after Layer 24 SSM input data print")
                        import sys
                        sys.exit(0)

            # ===== DUAL MODE DEBUG: Compare Mode 2-0 vs 2-1 =====
            # Run for ALL samples (remove the hasattr check)
            if self.layer_idx == 23 and os.environ.get('DUAL_MODE_DEBUG', 'false').lower() == 'true':
                dual_mode_compare_ssm(self, x, dt, B, C, z, self.conv1d.output_scale)

            # SSM with FP32 input (ONLY u is FP32, dt/B/C are INT8)
            y = self.selective_scan.forward(x_for_ssm, dt, B, C, z=z, return_last_state=ssm_state is not None)
        else:
            # Original INT8 path (completely unchanged)
            x_reshape = rearrange(x, "b d l -> b l d").contiguous()
            x_dbl = self.x_proj(x_reshape)  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
            # if DEBUG_ENABLED and self._debug_step_count <= 3:
            #     print(f"  After x_proj: x_dbl dtype: {x_dbl.dtype}, first 3: {x_dbl.flatten()[:3].tolist()}")
            #     print(f"  dt (before dt_proj) dtype: {dt.dtype}, first 3: {dt.flatten()[:3].tolist()}")
            # # ===== END TEMPORARY DEBUG CODE =====

            # Compute dt proj with x_proj_scale
            dt = self.dt_proj.to_seqlen_last(dt.contiguous())
            B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()

            # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
            # if DEBUG_ENABLED and self._debug_step_count <= 3:
            #     print(f"  After dt_proj: dt dtype: {dt.dtype}, first 3: {dt.flatten()[:3].tolist()}")
            #     print(f"  B dtype: {B.dtype}, first 3: {B.flatten()[:3].tolist()}")
            #     print(f"  C dtype: {C.dtype}, first 3: {C.flatten()[:3].tolist()}")
            #     print(f"  z dtype: {z.dtype if z is not None else 'None'}, first 3: {z.flatten()[:3].tolist() if z is not None else 'N/A'}")
            # # ===== END TEMPORARY DEBUG CODE =====

            # SSM step and return ssm_state
            y = self.selective_scan.forward(x, dt, B, C, z=z, return_last_state=ssm_state is not None)
        if ssm_state is not None:
            y, last_state = y # y: fp16, last_state: fp32
            ssm_state.copy_(last_state) # last_state: fp32 copy to ssm_state: fp16

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # if DEBUG_ENABLED and self._debug_step_count <= 3:
        #     print(f"  After SSM: y dtype: {y.dtype}, first 3: {y.flatten()[:3].tolist()}")
        # # ===== END TEMPORARY DEBUG CODE =====

        # Output projection
        y = self.had(y) # input fp16, output is int8

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # if DEBUG_ENABLED and self._debug_step_count <= 3:
        #     print(f"  After had: y dtype: {y.dtype}, first 3: {y.flatten()[:3].tolist()}")
        # # ===== END TEMPORARY DEBUG CODE =====

        out = self.out_proj(y) # HadW8A8BF16OF16Linear: input int8, output is fp16

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # if DEBUG_ENABLED and self._debug_step_count <= 3:
        #     print(f"  After out_proj: out dtype: {out.dtype}, first 3: {out.flatten()[:3].tolist()}")
        #     print(f"{'='*80}\n")
        # # ===== END TEMPORARY DEBUG CODE =====

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        # Input projection for x, z
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Perform causal conv1d and update conv_state in-place
        x = self.conv1d.update(x, conv_state)

        # Compute dt, B, C 
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt.contiguous())

        # SSM step and update ssm_state in-place
        y, ssm_state = self.selective_scan.update(ssm_state, x.contiguous(), dt, B, C, z=z)

        # Output projection
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = torch.int8
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = torch.int8
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=torch.int8,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=torch.int8,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class W8A8QMamba(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_had_transform=True,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": torch.float16}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        assert bias is False, "Only support bias=False for now"

        self.in_proj = W8A8B8O8Linear(self.d_model, self.d_inner * 2)

        self.conv1d = QCausalConv1D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = W8A8B8O8Linear(self.d_inner, self.dt_rank + self.d_state * 2)
        # we seperate the bias and put the bias in the QSScan
        self.dt_proj = W8A8B8O8Linear(self.dt_rank, self.d_inner)

        # Quantized selective scan
        self.selective_scan = QSScan(d_state=self.d_state, d_inner=self.d_inner, delta_softplus=True)

        # output proj
        if use_had_transform:
            self.had = QHadamard(self.d_inner, x_H_scale=1.0)
        else:
            self.had = QAct(scale=1.0)
        self.out_proj = W8A8B16O16Linear(self.d_inner, self.d_model)

    @classmethod
    def from_fp16(cls, originalLayer: MambaSimple, act_scales: Dict, use_had_transform: bool = True):

        qmixer = cls(
            d_model=originalLayer.d_model,
            d_state=originalLayer.d_state,
            d_conv=originalLayer.d_conv,
            expand=originalLayer.expand,
            dt_rank=originalLayer.dt_rank,
            use_had_transform = use_had_transform,
            use_fast_path=False,  # Fused kernel options
            layer_idx=originalLayer.layer_idx,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

        # input proj
        qmixer.in_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.in_proj),
            input_scale=act_scales["in_proj:input"].item(),
            output_scale=act_scales["in_proj:output"].item(),
        )

        # causal conv
        # no used, silu is fused in causal_conv1d
        qmixer.activation = "silu"
        assert qmixer.activation == "silu"
        qmixer.conv1d = QCausalConv1D.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.conv1d),
            input_scale=act_scales["in_proj:output"].item(),
            output_scale=act_scales["x_proj:input"].item(),            
        )

        # x_proj
        qmixer.x_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.x_proj),
            input_scale=act_scales["x_proj:input"].item(),
            output_scale=act_scales["x_proj:output"].item(),
        )

        # dt_proj
        original_dt_proj = copy.deepcopy(originalLayer.dt_proj)
        dt_proj_bias = originalLayer.dt_proj_bias.clone() # MambaSimple has separated bias 
        # original_dt_proj.bias = None
        qmixer.dt_proj = W8A8B8O8Linear.from_fp16(
            originalLayer=original_dt_proj,
            input_scale=act_scales["x_proj:output"].item(), # use x_proj_scale to avoid additional quantization operations
            output_scale=act_scales["dt_proj:output"].item(),
        )

        # ascan
        qmixer.selective_scan = QSScan.from_fp16(
            originalLayer.d_state, originalLayer.d_inner,
            originalLayer.A_log.clone(), D=originalLayer.D.clone(),
            dt_bias=dt_proj_bias, delta_softplus=True,
            ssm_state_scale=act_scales["ssm_state_act:input"],
            u_scale=act_scales["x_proj:input"],
            dt_scale=act_scales["dt_proj:output"],
            B_scale=act_scales["x_proj:output"],
            C_scale=act_scales["x_proj:output"],
            z_scale=act_scales["in_proj:output"],
        )

        # output proj
        if use_had_transform:
            qmixer.had.x_H_scale = act_scales["out_proj:input"].item()
        else:
            qmixer.had.scale = act_scales["out_proj:input"].item()
        qmixer.out_proj = W8A8B16O16Linear.from_fp16(
            originalLayer=copy.deepcopy(originalLayer.out_proj),
            input_scale=act_scales["out_proj:input"].item(),
        )
        return qmixer

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        #assert hidden_states.shape[0] == 1, "Current only support bsz=1"
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = self.in_proj.to_seqlen_last(hidden_states) #(B, D, L)
        x, z = xz.chunk(2, dim=1) #(B, D, L), #(B, D, L)
        
        # Perform causal conv1d and return conv_state
        if conv_state is not None:
            # store quantized x into conv_state
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        x = self.conv1d.forward(x)

        # Check if FP32 SSM mode is enabled (requires requantization for x_proj)
        import os
        fp32_mode_enabled = (
            os.environ.get('FLOAT_SIM_ASIC_INT8', 'false').lower() == 'true' or
            os.environ.get('CONV1D_MODE24_FP32', 'false').lower() == 'true' or
            os.environ.get('CONV1D_MODE3_FP32', 'false').lower() == 'true'
        )

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # # Only debug last layer (layer_idx == 23 for 24-layer model)
        # DEBUG_ENABLED = self.layer_idx is not None and self.layer_idx == 23
        # if DEBUG_ENABLED:
        #     if not hasattr(self, '_debug_step_count'):
        #         self._debug_step_count = 0
        #     self._debug_step_count += 1
        #     # Only print first 3 calls for this layer
        #     if self._debug_step_count <= 3:
        #         print(f"\n{'='*80}")
        #         print(f"[Layer {self.layer_idx} Call #{self._debug_step_count}] After Conv1D")
        #         print(f"  x dtype: {x.dtype}, shape: {x.shape}")
        #         x_vals = x.flatten()[:10]
        #         print(f"  First 10 values (high precision):")
        #         for i in range(min(10, x_vals.numel())):
        #             print(f"    x[{i}] = {x_vals[i]:.12f}" if x.dtype == torch.float32 else f"    x[{i}] = {x_vals[i]}")
        #         if hasattr(x, '_dual_scale_overflow_mask'):
        #             print(f"  x has dual-scale metadata (outliers: {x._dual_scale_overflow_mask.sum().item()})")

        #         # Print all scales for this layer (first call only)
        #         if self._debug_step_count == 1:
        #             print(f"\n  === Layer {self.layer_idx} All Scales ===")
        #             print(f"  in_proj scales:")
        #             if hasattr(self.in_proj, 'input_scale'):
        #                 print(f"    input_scale: {self.in_proj.input_scale:.10f}")
        #             if hasattr(self.in_proj, 'weight_scale'):
        #                 print(f"    weight_scale: {self.in_proj.weight_scale:.10f}")
        #             if hasattr(self.in_proj, 'output_scale'):
        #                 print(f"    output_scale: {self.in_proj.output_scale:.10f}")

        #             print(f"  conv1d scales:")
        #             print(f"    input_scale: {self.conv1d.input_scale:.10f}")
        #             print(f"    weight_scale: {self.conv1d.weight_scale:.10f}")
        #             if hasattr(self.conv1d, 'bias_scale') and self.conv1d.bias_scale is not None:
        #                 print(f"    bias_scale: {self.conv1d.bias_scale:.10f}")
        #             print(f"    output_scale: {self.conv1d.output_scale:.10f}")

        #             print(f"  x_proj scales:")
        #             if hasattr(self.x_proj, 'input_scale'):
        #                 print(f"    input_scale: {self.x_proj.input_scale:.10f}")
        #             if hasattr(self.x_proj, 'weight_scale'):
        #                 print(f"    weight_scale: {self.x_proj.weight_scale:.10f}")
        #             if hasattr(self.x_proj, 'output_scale'):
        #                 print(f"    output_scale: {self.x_proj.output_scale:.10f}")

        #             print(f"  selective_scan scales:")
        #             if hasattr(self.selective_scan, 'u_scale'):
        #                 print(f"    u_scale: {self.selective_scan.u_scale:.10f}")
        #             if hasattr(self.selective_scan, 'dt_scale'):
        #                 print(f"    dt_scale: {self.selective_scan.dt_scale:.10f}")

        #             print(f"  out_proj scales:")
        #             if hasattr(self.out_proj, 'input_scale'):
        #                 print(f"    input_scale: {self.out_proj.input_scale:.10f}")
        #             if hasattr(self.out_proj, 'weight_scale'):
        #                 print(f"    weight_scale: {self.out_proj.weight_scale:.10f}")
        #             if hasattr(self.out_proj, 'output_scale'):
        #                 print(f"    output_scale: {self.out_proj.output_scale:.10f}")
        # # ===== END TEMPORARY DEBUG CODE =====

        # Convert Conv1D output for use in SSM
        # Different modes have different requirements for SSM input dtype
        if fp32_mode_enabled:
            # Check mode-specific environment variables
            ssm_use_pytorch_int8 = os.environ.get('SSM_USE_PYTORCH_INT8', 'false').lower() == 'true'
            conv1d_mode23_fp32 = os.environ.get('CONV1D_MODE23_FP32', 'false').lower() == 'true'

            if x.dtype == torch.int8:
                # x is INT8 from Conv1D (Mode 2-0, 2-1, 2-2, 2-3 when FLOAT_SIM_ASIC_INT8=true)
                x_for_xproj = x  # INT8 for x_proj

                # Mode 2-1: Keep INT8 for PyTorch INT8 SSM (direct pass, no dequant)
                # Mode 2-0, 2-2: Dequantize to FP32 for SSM
                if ssm_use_pytorch_int8 and not conv1d_mode23_fp32:
                    # Mode 2-1: PyTorch INT8 SSM expects INT8 input directly
                    x_for_ssm = x  # Keep INT8
                else:
                    # Mode 2-0 (CUDA INT8 SSM with requant), Mode 2-2 (FP32 SSM)
                    x_for_ssm = x.float() * self.conv1d.output_scale  # Dequantize to FP32
            elif x.dtype == torch.float32:
                # x is FP32 from Conv1D (Mode 2-3, 2-4)
                # Requantize to INT8 for x_proj, keep FP32 for SSM
                x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
                x_for_ssm = x  # Keep FP32 for SSM
            else:
                raise ValueError(f"Unexpected dtype in fp32_mode_enabled: {x.dtype}")

            # Compute dt, B, C using INT8
            x_reshape = rearrange(x_for_xproj, "b d l -> b l d").contiguous()
            x_dbl = self.x_proj(x_reshape)  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            # Compute dt proj with x_proj_scale
            dt = self.dt_proj.to_seqlen_last(dt.contiguous())
            B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()

            # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
            # if DEBUG_ENABLED and self._debug_step_count <= 3:
            #     print(f"  After dt_proj: dt dtype: {dt.dtype}, first 3: {dt.flatten()[:3].tolist()}")
            #     print(f"  B dtype: {B.dtype}, first 3: {B.flatten()[:3].tolist()}")
            #     print(f"  C dtype: {C.dtype}, first 3: {C.flatten()[:3].tolist()}")
            #     print(f"  z dtype: {z.dtype if z is not None else 'None'}, first 3: {z.flatten()[:3].tolist() if z is not None else 'N/A'}")
            # # ===== END TEMPORARY DEBUG CODE =====

            # Print Layer 24 SSM scales (before SSM forward) - only once
            if self.layer_idx == 23:
                if not hasattr(self, '_ssm_scales_printed'):
                    self._ssm_scales_printed = False

                if not self._ssm_scales_printed:
                    print(f"\n{'='*80}")
                    print(f"[Layer 24 / layer_idx {self.layer_idx}] SSM Scales")
                    print(f"{'='*80}")
                    print(f"  Location: qMambaLayer.py forward() - fp32_mode_enabled branch")
                    print(f"  ")
                    print(f"  SSM Input Data (before SSM.forward):")
                    print(f"    u dtype: {x_for_ssm.dtype}")
                    print(f"    u first 5 values [0,0,:5]: {x_for_ssm[0, 0, :5].tolist()}")
                    print(f"    dt first 5 values [0,0,:5]: {dt[0, 0, :5].tolist()}")
                    print(f"    B first 5 values [0,0,:5]: {B[0, 0, :5].tolist()}")
                    print(f"    C first 5 values [0,0,:5]: {C[0, 0, :5].tolist()}")
                    print(f"    dt/B/C dtype: {dt.dtype}")
                    print(f"  ")
                    print(f"  SSM Scales (from self.selective_scan / QSScan):")
                    print(f"    u_scale          = {self.selective_scan.u_scale.item():.10f}  (for SSM input u)")
                    print(f"    dt_scale         = {self.selective_scan.dt_scale.item():.10f}  (for dt)")
                    print(f"    B_scale          = {self.selective_scan.B_scale.item():.10f}  (for B)")
                    print(f"    C_scale          = {self.selective_scan.C_scale.item():.10f}  (for C)")
                    print(f"    A_scale          = {self.selective_scan.A_scale.item():.10f}  (for A)")
                    print(f"    D_scale          = {self.selective_scan.D_scale.item():.10f}  (for D)")
                    print(f"    z_scale          = {self.selective_scan.z_scale.item():.10f}  (for z)")
                    print(f"    ssm_state_scale  = {self.selective_scan.ssm_state_scale.item():.10f}  (for state)")
                    print(f"    dt_bias_scale    = {self.selective_scan.dt_bias_scale.item():.10f}  (for dt_bias)")
                    print(f"  ")
                    print(f"  ⚠️  Important: Conv1D output_scale should match SSM u_scale")
                    print(f"    (Conv1D output_scale printed above)")
                    print(f"    SSM u_scale = {self.selective_scan.u_scale.item():.10f}")
                    print(f"{'='*80}\n")
                    self._ssm_scales_printed = True

                    # Quick verification mode: exit after printing Layer 24 SSM scales
                    if os.environ.get('QUICK_VERIFY', 'false').lower() == 'true':
                        print("🔍 QUICK_VERIFY mode: Exiting after Layer 24 SSM input data print")
                        import sys
                        sys.exit(0)

            # ===== DUAL MODE DEBUG: Compare Mode 2-0 vs 2-1 =====
            # Run for ALL samples (remove the hasattr check)
            if self.layer_idx == 23 and os.environ.get('DUAL_MODE_DEBUG', 'false').lower() == 'true':
                dual_mode_compare_ssm(self, x, dt, B, C, z, self.conv1d.output_scale)

            # SSM with FP32 input (ONLY u is FP32, dt/B/C are INT8)
            y = self.selective_scan.forward(x_for_ssm, dt, B, C, z=z, return_last_state=ssm_state is not None)
        else:
            # Original INT8 path (completely unchanged)
            x_reshape = rearrange(x, "b d l -> b l d").contiguous()
            x_dbl = self.x_proj(x_reshape)  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
            # if DEBUG_ENABLED and self._debug_step_count <= 3:
            #     print(f"  After x_proj: x_dbl dtype: {x_dbl.dtype}, first 3: {x_dbl.flatten()[:3].tolist()}")
            #     print(f"  dt (before dt_proj) dtype: {dt.dtype}, first 3: {dt.flatten()[:3].tolist()}")
            # # ===== END TEMPORARY DEBUG CODE =====

            # Compute dt proj with x_proj_scale
            dt = self.dt_proj.to_seqlen_last(dt.contiguous())
            B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()

            # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
            # if DEBUG_ENABLED and self._debug_step_count <= 3:
            #     print(f"  After dt_proj: dt dtype: {dt.dtype}, first 3: {dt.flatten()[:3].tolist()}")
            #     print(f"  B dtype: {B.dtype}, first 3: {B.flatten()[:3].tolist()}")
            #     print(f"  C dtype: {C.dtype}, first 3: {C.flatten()[:3].tolist()}")
            #     print(f"  z dtype: {z.dtype if z is not None else 'None'}, first 3: {z.flatten()[:3].tolist() if z is not None else 'N/A'}")
            # # ===== END TEMPORARY DEBUG CODE =====

            # SSM step and return ssm_state
            y = self.selective_scan.forward(x, dt, B, C, z=z, return_last_state=ssm_state is not None)
        if ssm_state is not None:
            y, last_state = y # y: fp16, last_state: fp32
            ssm_state.copy_(last_state) # last_state: fp32 copy to ssm_state: fp16

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # if DEBUG_ENABLED and self._debug_step_count <= 3:
        #     print(f"  After SSM: y dtype: {y.dtype}, first 3: {y.flatten()[:3].tolist()}")
        # # ===== END TEMPORARY DEBUG CODE =====

        # Output projection
        y = self.had(y) # input fp16, output is int8

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # if DEBUG_ENABLED and self._debug_step_count <= 3:
        #     print(f"  After had: y dtype: {y.dtype}, first 3: {y.flatten()[:3].tolist()}")
        # # ===== END TEMPORARY DEBUG CODE =====

        out = self.out_proj(y) # HadW8A8BF16OF16Linear: input int8, output is fp16

        # # ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
        # if DEBUG_ENABLED and self._debug_step_count <= 3:
        #     print(f"  After out_proj: out dtype: {out.dtype}, first 3: {out.flatten()[:3].tolist()}")
        #     print(f"{'='*80}\n")
        # # ===== END TEMPORARY DEBUG CODE =====

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        # Input projection for x, z
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Perform causal conv1d and update conv_state in-place
        x = self.conv1d.update(x, conv_state)

        # Compute dt, B, C 
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt.contiguous())

        # SSM step and update ssm_state in-place
        y, ssm_state = self.selective_scan.update(ssm_state, x.contiguous(), dt, B, C, z=z)

        # Output projection
        y = self.had(y) # input fp16, output is int8
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = torch.int8
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = torch.int8
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=torch.int8,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=torch.int8,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

# ===== DUAL MODE COMPARISON HELPER FUNCTION =====
# Global accumulator for all samples
_dual_mode_stats = {
    'sample_count': 0,
    'total_mean_diff': 0.0,
    'total_max_diff': 0.0,
    'all_diffs': []
}

def dual_mode_compare_ssm(layer_self, x, dt, B, C, z, output_scale):
    """
    Helper function to compare Mode 2-0 vs Mode 2-1 SSM outputs.
    Called when DUAL_MODE_DEBUG=true.
    Accumulates statistics across ALL samples.

    Args:
        layer_self: The MambaLayer instance
        x: Conv1D output (INT8)
        dt, B, C, z: SSM inputs
        output_scale: Conv1D output scale for dequantization
    """
    import torch
    import os
    global _dual_mode_stats

    # Only print details for first sample
    if _dual_mode_stats['sample_count'] == 0:
        print(f"\n{'='*80}")
        print(f"🔬 DUAL MODE COMPARISON - Layer {layer_self.layer_idx}")
        print(f"{'='*80}")
        print(f"📊 Conv1D Output (Sample 1, shared input for both modes):")
        print(f"    dtype: {x.dtype}")
        print(f"    first 5 values [0,0,:5]: {x[0, 0, :5].tolist()}")
        print(f"    output_scale: {output_scale:.10f}")
        print(f"")

    # Prepare inputs for BOTH modes
    x_mode20_fp32 = x.float() * output_scale  # Mode 2-0: dequantize to FP32
    x_mode21_int8 = x  # Mode 2-1: keep INT8

    if _dual_mode_stats['sample_count'] == 0:
        print(f"🔀 Mode 2-0 (CUDA INT8 with FP32 input):")
        print(f"    u dtype: torch.float32")
        print(f"    u first 5 values: {x_mode20_fp32[0, 0, :5].tolist()}")
        print(f"")

        print(f"🔀 Mode 2-1 (PyTorch INT8 direct):")
        print(f"    u dtype: torch.int8")
        print(f"    u first 5 values: {x_mode21_int8[0, 0, :5].tolist()}")
        print(f"    u first 5 (dequantized for ref): {(x_mode21_int8.float() * output_scale)[0, 0, :5].tolist()}")
        print(f"")

        # Run BOTH SSM kernels and compare outputs
        print(f"🚀 Running BOTH SSM kernels for ALL 100 samples...")

    # Save original environment
    original_env = {
        'SSM_USE_PYTORCH_INT8': os.environ.get('SSM_USE_PYTORCH_INT8', 'false'),
        'SSM_USE_CUDA_FOR_FP32': os.environ.get('SSM_USE_CUDA_FOR_FP32', 'false'),
        'DUAL_MODE_DEBUG': 'true'
    }

    # Temporarily disable debug mode to avoid infinite recursion
    os.environ['DUAL_MODE_DEBUG'] = 'false'

    try:
        # Run Mode 2-0: CUDA INT8 SSM with FP32 input
        os.environ['SSM_USE_PYTORCH_INT8'] = 'false'
        os.environ['SSM_USE_CUDA_FOR_FP32'] = 'true'
        y_ssm_mode20 = layer_self.selective_scan.forward(x_mode20_fp32, dt, B, C, z=z, return_last_state=False)

        # Run Mode 2-1: PyTorch INT8 SSM with INT8 input
        os.environ['SSM_USE_PYTORCH_INT8'] = 'true'
        os.environ['SSM_USE_CUDA_FOR_FP32'] = 'false'
        y_ssm_mode21 = layer_self.selective_scan.forward(x_mode21_int8, dt, B, C, z=z, return_last_state=False)

        # Continue through had + out_proj to get final layer output
        y_had_mode20 = layer_self.had(y_ssm_mode20)
        y_had_mode21 = layer_self.had(y_ssm_mode21)

        out_mode20 = layer_self.out_proj(y_had_mode20)
        out_mode21 = layer_self.out_proj(y_had_mode21)

        # Compute SSM output difference
        diff_ssm = (y_ssm_mode20.float() - y_ssm_mode21.float()).abs()
        mean_diff_ssm = diff_ssm.mean().item()
        max_diff_ssm = diff_ssm.max().item()

        # Compute final layer output difference
        diff_out = (out_mode20.float() - out_mode21.float()).abs()
        mean_diff_out = diff_out.mean().item()
        max_diff_out = diff_out.max().item()

        # Accumulate statistics
        _dual_mode_stats['sample_count'] += 1
        _dual_mode_stats['total_mean_diff'] += mean_diff_ssm
        _dual_mode_stats['total_max_diff'] = max(max_diff_ssm, _dual_mode_stats['total_max_diff'])
        _dual_mode_stats['all_diffs'].append(mean_diff_ssm)

        # Also track layer output differences
        if 'total_mean_diff_out' not in _dual_mode_stats:
            _dual_mode_stats['total_mean_diff_out'] = 0.0
            _dual_mode_stats['total_max_diff_out'] = 0.0
            _dual_mode_stats['all_diffs_out'] = []

        _dual_mode_stats['total_mean_diff_out'] += mean_diff_out
        _dual_mode_stats['total_max_diff_out'] = max(max_diff_out, _dual_mode_stats['total_max_diff_out'])
        _dual_mode_stats['all_diffs_out'].append(mean_diff_out)

        # Print details for first sample only
        if _dual_mode_stats['sample_count'] == 1:
            print(f"")
            print(f"📈 SSM Output Comparison (Sample 1):")
            print(f"    Mode 2-0 first 5 values [0,0,:5]: {y_ssm_mode20[0, 0, :5].tolist()}")
            print(f"    Mode 2-1 first 5 values [0,0,:5]: {y_ssm_mode21[0, 0, :5].tolist()}")
            print(f"    Mean absolute diff: {mean_diff_ssm:.6f}")
            print(f"    Max absolute diff: {max_diff_ssm:.6f}")
            print(f"")
            print(f"📈 After Hadamard (Sample 1):")
            print(f"    Mode 2-0 had output [0,0,:5]: {y_had_mode20[0, 0, :5].tolist()}")
            print(f"    Mode 2-1 had output [0,0,:5]: {y_had_mode21[0, 0, :5].tolist()}")
            print(f"")
            print(f"📈 Final Layer Output (Sample 1):")
            print(f"    Mode 2-0 first 5 values [0,0,:5]: {out_mode20[0, 0, :5].tolist()}")
            print(f"    Mode 2-1 first 5 values [0,0,:5]: {out_mode21[0, 0, :5].tolist()}")
            print(f"    Mean absolute diff: {mean_diff_out:.6f}")
            print(f"    Max absolute diff: {max_diff_out:.6f}")
            print(f"")
            print(f"⏳ Continuing to compare remaining samples (progress shown by lm_eval)...")
            print(f"{'='*80}\n")

        # Print summary after all samples
        if _dual_mode_stats['sample_count'] == 100:
            avg_mean_diff_ssm = _dual_mode_stats['total_mean_diff'] / 100
            avg_mean_diff_out = _dual_mode_stats['total_mean_diff_out'] / 100
            print(f"\n{'='*80}")
            print(f"📊 DUAL MODE COMPARISON - FINAL SUMMARY (100 samples)")
            print(f"{'='*80}")
            print(f"")
            print(f"🔹 SSM Output Differences:")
            print(f"  Average mean absolute diff: {avg_mean_diff_ssm:.6f}")
            print(f"  Maximum absolute diff: {_dual_mode_stats['total_max_diff']:.6f}")
            import statistics
            print(f"  Min: {min(_dual_mode_stats['all_diffs']):.6f}")
            print(f"  Max: {max(_dual_mode_stats['all_diffs']):.6f}")
            print(f"  Median: {statistics.median(_dual_mode_stats['all_diffs']):.6f}")
            print(f"")
            print(f"🔹 Final Layer Output Differences (after had + out_proj):")
            print(f"  Average mean absolute diff: {avg_mean_diff_out:.6f}")
            print(f"  Maximum absolute diff: {_dual_mode_stats['total_max_diff_out']:.6f}")
            print(f"  Min: {min(_dual_mode_stats['all_diffs_out']):.6f}")
            print(f"  Max: {max(_dual_mode_stats['all_diffs_out']):.6f}")
            print(f"  Median: {statistics.median(_dual_mode_stats['all_diffs_out']):.6f}")
            print(f"")
            if avg_mean_diff_out < 0.001:
                print(f"✅ Mode 2-0 and Mode 2-1 produce nearly IDENTICAL layer outputs!")
                print(f"   Difference < 0.001 is within floating-point precision.")
            else:
                print(f"⚠️  Mode 2-0 and Mode 2-1 have measurable differences in layer outputs.")
                print(f"   This could lead to different final predictions.")
            print(f"{'='*80}\n")

    finally:
        # Restore original environment
        for key, value in original_env.items():
            os.environ[key] = value

    def forward_mode5(self, hidden_states, inference_params=None):
        """
        Mode 5: Dual-path forward
        - Mode 0 path: Normal INT8 quantization
        - Mode 5 path: Conv1D outputs FP32, SSM receives FP32
        - Both paths start from the SAME initial quantized input
        - Compare outputs at each layer (layers 0, 1, 2, 23)
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda
        
        batch, seqlen, dim = hidden_states.shape
        
        # === Step 1: in_proj (shared by both paths) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)  # (B, D, L) INT8
        x_int8, z = xz.chunk(2, dim=1)
        
        # === Step 2a: Mode 0 Conv1D (INT8 → INT8) ===
        x_mode0 = quant_causal_conv1d_cuda.fwd(
            x_int8, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.output_scale, self.conv1d.bias_scale,
            self.conv1d.bias, None, None, None, True
        )  # INT8
        
        # === Step 2b: Mode 5 Conv1D (INT8 → FP32) ===
        x_mode5_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_int8, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32
        
        # === Step 3a: Mode 0 SSM (INT8 input) ===
        x_mode0_reshape = rearrange(x_mode0, "b d l -> b l d").contiguous()
        x_dbl_mode0 = self.x_proj(x_mode0_reshape)
        dt_mode0, B_mode0, C_mode0 = torch.split(x_dbl_mode0, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_mode0 = self.dt_proj.to_seqlen_last(dt_mode0.contiguous())
        B_mode0 = rearrange(B_mode0, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C_mode0 = rearrange(C_mode0, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        y_mode0 = self.selective_scan.forward(x_mode0, dt_mode0, B_mode0, C_mode0, z=z, return_last_state=False)
        
        # === Step 3b: Mode 5 SSM (FP32 input) ===
        # Requantize FP32 to INT8 for x_proj
        x_mode5_int8_for_xproj = torch.round(x_mode5_fp32 / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
        x_mode5_reshape = rearrange(x_mode5_int8_for_xproj, "b d l -> b l d").contiguous()
        x_dbl_mode5 = self.x_proj(x_mode5_reshape)
        dt_mode5, B_mode5, C_mode5 = torch.split(x_dbl_mode5, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_mode5 = self.dt_proj.to_seqlen_last(dt_mode5.contiguous())
        B_mode5 = rearrange(B_mode5, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C_mode5 = rearrange(C_mode5, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        # Call Mode 5 SSM kernel (FP32 input)
        y_mode5, _ = quant_sscan_cuda.fwd_mode5(
            x_mode5_fp32,  # FP32 input
            dt_mode5, self.selective_scan.dt_scale,
            self.selective_scan.A, self.selective_scan.A_scale,
            B_mode5, self.selective_scan.B_scale,
            C_mode5, self.selective_scan.C_scale,
            self.selective_scan.ssm_state_scale,
            self.selective_scan.D, self.selective_scan.D_scale,
            z, self.selective_scan.z_scale,
            self.selective_scan.dt_bias, self.selective_scan.dt_bias_scale,
            True  # delta_softplus
        )
        
        # === Step 4: Compare outputs (for layers 0, 1, 2, 23) ===
        if self.layer_idx in [0, 1, 2, 23]:
            y_mode0_fp16 = y_mode0.half() if y_mode0.dtype != torch.float16 else y_mode0
            y_mode5_fp16 = y_mode5.half() if y_mode5.dtype != torch.float16 else y_mode5
            
            # Sample 3 values
            sample_indices = [0, y_mode0_fp16.numel() // 2, y_mode0_fp16.numel() - 1]
            y_mode0_flat = y_mode0_fp16.flatten()
            y_mode5_flat = y_mode5_fp16.flatten()
            
            # Statistics
            diff = (y_mode0_flat.float() - y_mode5_flat.float()).abs()
            
            print(f"\n{'='*80}")
            print(f"[Mode 5 Dual-Path] Layer {self.layer_idx} SSM Output Comparison")
            print(f"{'='*80}")
            print(f"Sample values (3 positions):")
            for idx in sample_indices:
                print(f"  [{idx}] Mode 0: {y_mode0_flat[idx].item():.6f}, Mode 5: {y_mode5_flat[idx].item():.6f}, Diff: {diff[idx].item():.6f}")
            print(f"Statistics:")
            print(f"  Mean: Mode 0={y_mode0_flat.float().mean().item():.6f}, Mode 5={y_mode5_flat.float().mean().item():.6f}")
            print(f"  Std:  Mode 0={y_mode0_flat.float().std().item():.6f}, Mode 5={y_mode5_flat.float().std().item():.6f}")
            print(f"  Min:  Mode 0={y_mode0_flat.min().item():.6f}, Mode 5={y_mode5_flat.min().item():.6f}")
            print(f"  Max:  Mode 0={y_mode0_flat.max().item():.6f}, Mode 5={y_mode5_flat.max().item():.6f}")
            print(f"Difference Statistics:")
            print(f"  Mean Abs Diff: {diff.mean().item():.6f}")
            print(f"  Max Abs Diff:  {diff.max().item():.6f}")
            print(f"{'='*80}\n")
        
        # === Step 5: Continue with Mode 5 output ===
        y_mode5_for_output = y_mode5.half() if y_mode5.dtype != torch.float16 else y_mode5
        y = self.had(y_mode5_for_output)
        out = self.out_proj(y)
        
        return out

    def forward_mode6(self, hidden_states, inference_params=None):
        """
        Mode 6: Dual-path forward
        - Mode 0 path: Takes quantized Mode 6 Conv FP32 output as input
        - Mode 6 path: Conv1D outputs FP32, SSM receives FP32
        - Mode 6 output feeds BOTH paths (quantized for Mode 0, FP32 for Mode 6)
        - Compare outputs at each layer (layers 0, 1, 2, 23)
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda
        
        batch, seqlen, dim = hidden_states.shape
        
        # === Step 1: in_proj (shared by both paths) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)  # (B, D, L) INT8
        x_int8, z = xz.chunk(2, dim=1)
        
        # === Step 2: Mode 6 Conv1D (INT8 → FP32) ===
        x_mode6_fp32 = quant_causal_conv1d_cuda.fwd_mode6(
            x_int8, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32
        
        # === Step 3a: Mode 0 Conv1D (quantize Mode 6 output → INT8 → INT8) ===
        x_mode6_int8 = torch.round(x_mode6_fp32 / self.conv1d.input_scale).clamp(-128, 127).to(torch.int8)
        x_mode0 = quant_causal_conv1d_cuda.fwd(
            x_mode6_int8, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.output_scale, self.conv1d.bias_scale,
            self.conv1d.bias, None, None, None, True
        )  # INT8
        
        # === Step 4a: Mode 0 SSM (INT8 input) ===
        x_mode0_reshape = rearrange(x_mode0, "b d l -> b l d").contiguous()
        x_dbl_mode0 = self.x_proj(x_mode0_reshape)
        dt_mode0, B_mode0, C_mode0 = torch.split(x_dbl_mode0, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_mode0 = self.dt_proj.to_seqlen_last(dt_mode0.contiguous())
        B_mode0 = rearrange(B_mode0, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C_mode0 = rearrange(C_mode0, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        y_mode0 = self.selective_scan.forward(x_mode0, dt_mode0, B_mode0, C_mode0, z=z, return_last_state=False)
        
        # === Step 4b: Mode 6 SSM (FP32 input) ===
        # Requantize FP32 to INT8 for x_proj
        x_mode6_int8_for_xproj = torch.round(x_mode6_fp32 / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
        x_mode6_reshape = rearrange(x_mode6_int8_for_xproj, "b d l -> b l d").contiguous()
        x_dbl_mode6 = self.x_proj(x_mode6_reshape)
        dt_mode6, B_mode6, C_mode6 = torch.split(x_dbl_mode6, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_mode6 = self.dt_proj.to_seqlen_last(dt_mode6.contiguous())
        B_mode6 = rearrange(B_mode6, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C_mode6 = rearrange(C_mode6, "b l dstate -> b dstate l", l=seqlen).contiguous()
        
        # Call Mode 6 SSM kernel (FP32 input)
        y_mode6, _ = quant_sscan_cuda.fwd_mode6(
            x_mode6_fp32,  # FP32 input
            dt_mode6, self.selective_scan.dt_scale,
            self.selective_scan.A, self.selective_scan.A_scale,
            B_mode6, self.selective_scan.B_scale,
            C_mode6, self.selective_scan.C_scale,
            self.selective_scan.ssm_state_scale,
            self.selective_scan.D, self.selective_scan.D_scale,
            z, self.selective_scan.z_scale,
            self.selective_scan.dt_bias, self.selective_scan.dt_bias_scale,
            True  # delta_softplus
        )
        
        # === Step 5: Compare outputs (for layers 0, 1, 2, 23) ===
        if self.layer_idx in [0, 1, 2, 23]:
            y_mode0_fp16 = y_mode0.half() if y_mode0.dtype != torch.float16 else y_mode0
            y_mode6_fp16 = y_mode6.half() if y_mode6.dtype != torch.float16 else y_mode6
            
            # Sample 3 values
            sample_indices = [0, y_mode0_fp16.numel() // 2, y_mode0_fp16.numel() - 1]
            y_mode0_flat = y_mode0_fp16.flatten()
            y_mode6_flat = y_mode6_fp16.flatten()
            
            # Statistics
            diff = (y_mode0_flat.float() - y_mode6_flat.float()).abs()
            
            print(f"\n{'='*80}")
            print(f"[Mode 6 Dual-Path] Layer {self.layer_idx} SSM Output Comparison")
            print(f"{'='*80}")
            print(f"Sample values (3 positions):")
            for idx in sample_indices:
                print(f"  [{idx}] Mode 0: {y_mode0_flat[idx].item():.6f}, Mode 6: {y_mode6_flat[idx].item():.6f}, Diff: {diff[idx].item():.6f}")
            print(f"Statistics:")
            print(f"  Mean: Mode 0={y_mode0_flat.float().mean().item():.6f}, Mode 6={y_mode6_flat.float().mean().item():.6f}")
            print(f"  Std:  Mode 0={y_mode0_flat.float().std().item():.6f}, Mode 6={y_mode6_flat.float().std().item():.6f}")
            print(f"  Min:  Mode 0={y_mode0_flat.min().item():.6f}, Mode 6={y_mode6_flat.min().item():.6f}")
            print(f"  Max:  Mode 0={y_mode0_flat.max().item():.6f}, Mode 6={y_mode6_flat.max().item():.6f}")
            print(f"Difference Statistics:")
            print(f"  Mean Abs Diff: {diff.mean().item():.6f}")
            print(f"  Max Abs Diff:  {diff.max().item():.6f}")
            print(f"{'='*80}\n")
        
        # === Step 6: Continue with Mode 6 output ===
        y_mode6_for_output = y_mode6.half() if y_mode6.dtype != torch.float16 else y_mode6
        y = self.had(y_mode6_for_output)
        out = self.out_proj(y)
        
        return out
