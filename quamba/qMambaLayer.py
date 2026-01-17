import math
import copy
from functools import partial
from typing import Optional, Dict

import torch

# ===== DEBUG: 层计数器 =====
_DEBUG_LAYER_COUNTER = {'count': 0, 'total': None}
# ===== END DEBUG =====

# ===== Mode 6-4: α=1.0 scale 缓存 =====
import os as _os
_PA1_SCALES_CACHE = {}  # {model_size: {layer_idx: output_scale}}
_PA1_MODEL_PATHS = {
    '130m': '/workspace/Quamba/pretrained/130mpercentile1125/130mpercentile1125/pa-1/quamba-130m-w8a8',
    '1.4b': '/workspace/Quamba/pretrained/percentile1125/1p4b/pa-1/quamba-1.4b-w8a8',
    '2.8b': '/workspace/Quamba/pretrained/percentile1125/2p8b/pa-1/quamba-2.8b-w8a8',
}
# ===== END Mode 6-4 =====
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

        # Check if FP32 SSM mode is enabled (Mode 4 only)
        import os
        fp32_mode_enabled = os.environ.get('CONV1D_MODE24_FP32', 'false').lower() == 'true'

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

        # Convert Conv1D output for use in SSM (Mode 4 only)
        if fp32_mode_enabled:
            # Mode 4: x is FP32 from Conv1D
            # Requantize to INT8 for x_proj, keep FP32 for SSM
            x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
            x_for_ssm = x  # Keep FP32 for SSM

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

            # ===== DEBUG: Mode 0 统一调试输出 =====
            import os
            if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
                # 对所有层统计 overflow (理论上，反量化后如果用 x_proj scale 重新量化的值)
                x_dequant = x.float() * self.conv1d.output_scale
                x_requant = x_dequant / self.x_proj.a.item()  # 如果用 x_proj scale 重新量化
                overflow_mask = torch.abs(x_requant) > 127
                overflow_count = overflow_mask.sum().item()
                total_count = overflow_mask.numel()
                overflow_ratio = overflow_count / total_count * 100

                # 累积所有层的 overflow 统计
                if not hasattr(self, '_mode0_overflow_stats'):
                    self._mode0_overflow_stats = {'overflow': 0, 'total': 0}
                self._mode0_overflow_stats['overflow'] += overflow_count
                self._mode0_overflow_stats['total'] += total_count

                # 最后一层打印详细信息
                last_layer_idx = 63  # 2.8b=63, 1.4b=47, 130m=23
                if self.layer_idx == last_layer_idx:
                    if not hasattr(self, '_mode0_debug_count'):
                        self._mode0_debug_count = 0
                    self._mode0_debug_count += 1
                    if self._mode0_debug_count == 1:  # 只打印第一次
                        print(f"\n{'='*80}")
                        print(f"[Mode 0 - Layer {last_layer_idx+1}] Debug Output (Baseline INT8)")
                        print(f"{'='*80}")
                        # 全局 overflow 统计
                        global_overflow_ratio = self._mode0_overflow_stats['overflow'] / self._mode0_overflow_stats['total'] * 100
                        print(f"\n--- Overflow Statistics (All Layers) ---")
                        print(f"  Total overflow: {self._mode0_overflow_stats['overflow']} / {self._mode0_overflow_stats['total']} = {global_overflow_ratio:.4f}%")
                        print(f"  NOTE: Mode 0 uses INT8 throughout, overflow is theoretical (if re-quantized with x_proj scale)")
                        # Conv1D 输出
                        print(f"\n--- Conv1D Output (Layer {last_layer_idx+1}) ---")
                        print(f"  x_int8 raw: min={x.min().item()}, max={x.max().item()}")
                        print(f"  x_dequant: min={x_dequant.min().item():.4f}, max={x_dequant.max().item():.4f}, mean={x_dequant.mean().item():.4f}, std={x_dequant.std().item():.4f}")
                        print(f"  x_dequant first 5: {x_dequant[0, 0, :5].tolist()}")
                        print(f"  x_int8 first 5: {x[0, 0, :5].tolist()}")
                        print(f"  Layer overflow (theoretical): {overflow_count} / {total_count} = {overflow_ratio:.4f}%")
                        # Scales
                        print(f"\n--- Scales ---")
                        print(f"  conv1d.input_scale:  {self.conv1d.input_scale:.10f}")
                        print(f"  conv1d.output_scale: {self.conv1d.output_scale:.10f}")
                        print(f"  x_proj.a (input_scale): {self.x_proj.a.item():.10f}")
                        print(f"  selective_scan.u_scale: {self.selective_scan.u_scale.item():.10f}")
                        # x_proj 输出
                        print(f"\n--- x_proj/dt_proj Output ---")
                        print(f"  dt first 5: {dt[0, 0, :5].tolist()}")
                        print(f"  B first 5: {B[0, 0, :5].tolist()}")
                        print(f"  C first 5: {C[0, 0, :5].tolist()}")
            # ===== END DEBUG =====

            # SSM step and return ssm_state
            y = self.selective_scan.forward(x, dt, B, C, z=z, return_last_state=ssm_state is not None)

            # ===== DEBUG: Mode 0 SSM输出 =====
            import os
            if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
                last_layer_idx = 63  # 2.8b=63, 1.4b=47, 130m=23
                if self.layer_idx == last_layer_idx and hasattr(self, '_mode0_debug_count') and self._mode0_debug_count == 1:
                    # y 可能是 tuple (y, last_state) 或者 tensor
                    y_tensor = y[0] if isinstance(y, tuple) else y
                    print(f"\n--- SSM Output ---")
                    print(f"  y: min={y_tensor.min().item():.6f}, max={y_tensor.max().item():.6f}, mean={y_tensor.mean().item():.6f}")
                    print(f"  y first 5: {y_tensor[0, 0, :5].tolist()}")
                    print(f"{'='*80}\n")
            # ===== END DEBUG =====

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

        # Check if FP32 SSM mode is enabled (Mode 4 only)
        import os
        fp32_mode_enabled = os.environ.get('CONV1D_MODE24_FP32', 'false').lower() == 'true'

        # Convert Conv1D output for use in SSM (Mode 4 only)
        if fp32_mode_enabled:
            # Mode 4: x is FP32 from Conv1D
            # Requantize to INT8 for x_proj, keep FP32 for SSM
            x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
            x_for_ssm = x  # Keep FP32 for SSM

            # Compute dt, B, C using INT8
            x_reshape = rearrange(x_for_xproj, "b d l -> b l d").contiguous()
            x_dbl = self.x_proj(x_reshape)  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            # Compute dt proj with x_proj_scale
            dt = self.dt_proj.to_seqlen_last(dt.contiguous())
            B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()

        
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

            # SSM with FP32 input (ONLY u is FP32, dt/B/C are INT8)
            y = self.selective_scan.forward(x_for_ssm, dt, B, C, z=z, return_last_state=ssm_state is not None)
            sys.exit(0)# path 2 not used. Exit to avoid mistakes
        else:
            # Original INT8 path (completely unchanged)
            x_reshape = rearrange(x, "b d l -> b l d").contiguous()
            x_dbl = self.x_proj(x_reshape)  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            # Compute dt proj with x_proj_scale
            dt = self.dt_proj.to_seqlen_last(dt.contiguous())
            B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
 
            # ===== DEBUG: Mode 0 统一调试输出 =====
            import os
            if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
                # 对所有层统计 overflow (理论上，反量化后如果用 x_proj scale 重新量化的值)
                x_dequant = x.float() * self.conv1d.output_scale
                x_requant = x_dequant / self.x_proj.a.item()  # 如果用 x_proj scale 重新量化
                overflow_mask = torch.abs(x_requant) > 127
                overflow_count = overflow_mask.sum().item()
                total_count = overflow_mask.numel()
                overflow_ratio = overflow_count / total_count * 100

                # 累积所有层的 overflow 统计
                if not hasattr(self, '_mode0_overflow_stats'):
                    self._mode0_overflow_stats = {'overflow': 0, 'total': 0}
                self._mode0_overflow_stats['overflow'] += overflow_count
                self._mode0_overflow_stats['total'] += total_count

                # 最后一层打印详细信息
                last_layer_idx = 63  # 2.8b=63, 1.4b=47, 130m=23
                if self.layer_idx == last_layer_idx:
                    if not hasattr(self, '_mode0_debug_count'):
                        self._mode0_debug_count = 0
                    self._mode0_debug_count += 1
                    if self._mode0_debug_count == 1:  # 只打印第一次
                        print(f"\n{'='*80}")
                        print(f"[Mode 0 - Layer {last_layer_idx+1}] Debug Output (Baseline INT8)")
                        print(f"{'='*80}")
                        # 全局 overflow 统计
                        global_overflow_ratio = self._mode0_overflow_stats['overflow'] / self._mode0_overflow_stats['total'] * 100
                        print(f"\n--- Overflow Statistics (All Layers) ---")
                        print(f"  Total overflow: {self._mode0_overflow_stats['overflow']} / {self._mode0_overflow_stats['total']} = {global_overflow_ratio:.4f}%")
                        print(f"  NOTE: Mode 0 uses INT8 throughout, overflow is theoretical (if re-quantized with x_proj scale)")
                        # Conv1D 输出
                        print(f"\n--- Conv1D Output (Layer {last_layer_idx+1}) ---")
                        print(f"  x_int8 raw: min={x.min().item()}, max={x.max().item()}")
                        print(f"  x_dequant: min={x_dequant.min().item():.4f}, max={x_dequant.max().item():.4f}, mean={x_dequant.mean().item():.4f}, std={x_dequant.std().item():.4f}")
                        print(f"  x_dequant first 5: {x_dequant[0, 0, :5].tolist()}")
                        print(f"  x_int8 first 5: {x[0, 0, :5].tolist()}")
                        print(f"  Layer overflow (theoretical): {overflow_count} / {total_count} = {overflow_ratio:.4f}%")
                        # Scales
                        print(f"\n--- Scales ---")
                        print(f"  conv1d.input_scale:  {self.conv1d.input_scale:.10f}")
                        print(f"  conv1d.output_scale: {self.conv1d.output_scale:.10f}")
                        print(f"  x_proj.a (input_scale): {self.x_proj.a.item():.10f}")
                        print(f"  selective_scan.u_scale: {self.selective_scan.u_scale.item():.10f}")
                        # x_proj 输出
                        print(f"\n--- x_proj/dt_proj Output ---")
                        print(f"  dt first 5: {dt[0, 0, :5].tolist()}")
                        print(f"  B first 5: {B[0, 0, :5].tolist()}")
                        print(f"  C first 5: {C[0, 0, :5].tolist()}")
            # ===== END DEBUG =====

            # SSM step and return ssm_state
            y = self.selective_scan.forward(x, dt, B, C, z=z, return_last_state=ssm_state is not None)

            # ===== DEBUG: Mode 0 SSM输出 =====
            import os
            if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
                last_layer_idx = 63  # 2.8b=63, 1.4b=47, 130m=23
                if self.layer_idx == last_layer_idx and hasattr(self, '_mode0_debug_count') and self._mode0_debug_count == 1:
                    # y 可能是 tuple (y, last_state) 或者 tensor
                    y_tensor = y[0] if isinstance(y, tuple) else y
                    print(f"\n--- SSM Output ---")
                    print(f"  y: min={y_tensor.min().item():.6f}, max={y_tensor.max().item():.6f}, mean={y_tensor.mean().item():.6f}")
                    print(f"  y first 5: {y_tensor[0, 0, :5].tolist()}")
                    print(f"{'='*80}\n")
            # ===== END DEBUG =====

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

    def forward_mode5_0(self, hidden_states, inference_params=None):
        """
        Mode 5-0: Real FP32 + Virtual INT8 路径
        - Conv1D: INT8 输入 → FP32 输出 → Virtual INT8 (量化到 INT8 grid 的 FP32 值)
        - x_proj: INT8 输入 (从 Virtual INT8 转换)
        - SSM: FP32 输入 (Virtual INT8 值) via quant_sscan_cuda.fwd_mode5
        - 目标：验证 Virtual INT8 量化是否能达到与真实 INT8 (Mode 0) 相同的精度
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)  # (B, D, L) INT8
        x_5_0, z_5_0 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32 → Virtual INT8) ===
        # 使用 fwd_mode5 获取 FP32 输出
        x_5_0_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_5_0, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32 输出

        # Virtual INT8: 将 FP32 值量化到 INT8 grid (但保持 FP32 dtype)
        # 重要：使用 roundf_like 模拟 C++ roundf()，而不是 torch.round()
        # torch.round() 使用 "round half to even" (banker's rounding): 0.5 → 0, 1.5 → 2
        # C++ roundf() 使用 "round half away from zero": 0.5 → 1, -0.5 → -1
        def roundf_like(x):
            """模拟 C++ roundf() - round half away from zero"""
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

        x_5_0_scaled = x_5_0_fp32 / self.conv1d.output_scale
        x_5_0_int8_values = roundf_like(x_5_0_scaled).clamp(-128, 127)
        x_5_0_virtual_int8 = x_5_0_int8_values * self.conv1d.output_scale  # FP32 on INT8 grid

        # === Step 3: x_proj, dt_proj (需要真实 INT8 输入) ===
        # 复用已计算的 int8 值，避免重复计算
        x_5_0_int8_for_xproj = x_5_0_int8_values.to(torch.int8)
        x_5_0_reshape = rearrange(x_5_0_int8_for_xproj, "b d l -> b l d").contiguous()
        x_dbl_5_0 = self.x_proj(x_5_0_reshape)
        dt_5_0, B_5_0, C_5_0 = torch.split(x_dbl_5_0, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_5_0 = self.dt_proj.to_seqlen_last(dt_5_0.contiguous())
        B_5_0 = rearrange(B_5_0, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_5_0 = rearrange(C_5_0, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 4: SSM (FP32 Virtual INT8 输入) via fwd_mode5 ===
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_5_0, _ = quant_sscan_cuda.fwd_mode5(
            x_5_0_virtual_int8,  # FP32 Virtual INT8 输入
            dt_5_0, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_5_0, ensure_shape_1(self.selective_scan.B_scale),
            C_5_0, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_5_0, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True  # delta_softplus
        )

        # ===== DEBUG: Mode 5-0 vs Mode 0 对比 =====
        import os
        if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
            # 对所有层统计 overflow，但只打印最后一层详细信息
            # 统计 overflow (超出 INT8 范围)
            overflow_mask = torch.abs(x_5_0_scaled) > 127
            overflow_count = overflow_mask.sum().item()
            total_count = overflow_mask.numel()
            overflow_ratio = overflow_count / total_count * 100

            # 累积所有层的 overflow 统计
            if not hasattr(self, '_mode50_overflow_stats'):
                self._mode50_overflow_stats = {'overflow': 0, 'total': 0}
            self._mode50_overflow_stats['overflow'] += overflow_count
            self._mode50_overflow_stats['total'] += total_count

            # 最后一层打印详细信息
            last_layer_idx = 63  # 2.8b=63, 1.4b=47, 130m=23
            if self.layer_idx == last_layer_idx:
                if not hasattr(self, '_mode50_debug_count'):
                    self._mode50_debug_count = 0
                self._mode50_debug_count += 1
                if self._mode50_debug_count == 1:  # 只打印第一次
                    print(f"\n{'='*80}")
                    print(f"[Mode 5-0 - Layer {last_layer_idx+1}] Debug Output (Virtual INT8)")
                    print(f"{'='*80}")
                    # 全局 overflow 统计
                    global_overflow_ratio = self._mode50_overflow_stats['overflow'] / self._mode50_overflow_stats['total'] * 100
                    print(f"\n--- Overflow Statistics (All Layers) ---")
                    print(f"  Total overflow: {self._mode50_overflow_stats['overflow']} / {self._mode50_overflow_stats['total']} = {global_overflow_ratio:.4f}%")
                    # Conv1D 输出
                    print(f"\n--- Conv1D Output (Layer {last_layer_idx+1}) ---")
                    print(f"  x_fp32 raw: min={x_5_0_fp32.min().item():.4f}, max={x_5_0_fp32.max().item():.4f}, mean={x_5_0_fp32.mean().item():.4f}, std={x_5_0_fp32.std().item():.4f}")
                    print(f"  x_fp32 first 5: {x_5_0_fp32[0, 0, :5].tolist()}")
                    print(f"  x_virtual_int8 first 5: {x_5_0_virtual_int8[0, 0, :5].tolist()}")
                    print(f"  x_int8 first 5: {x_5_0_int8_for_xproj[0, 0, :5].tolist()}")
                    print(f"  Layer overflow: {overflow_count} / {total_count} = {overflow_ratio:.4f}%")
                    # Scales
                    print(f"\n--- Scales ---")
                    print(f"  conv1d.input_scale:  {self.conv1d.input_scale:.10f}")
                    print(f"  conv1d.output_scale: {self.conv1d.output_scale:.10f}")
                    print(f"  x_proj.a (input_scale): {self.x_proj.a.item():.10f}")
                    # x_proj 输出
                    print(f"\n--- x_proj/dt_proj Output ---")
                    print(f"  dt first 5: {dt_5_0[0, 0, :5].tolist()}")
                    print(f"  B first 5: {B_5_0[0, 0, 0, :5].tolist()}")
                    print(f"  C first 5: {C_5_0[0, 0, 0, :5].tolist()}")
                    # SSM 输出
                    print(f"\n--- SSM Output ---")
                    print(f"  y: min={y_5_0.min().item():.6f}, max={y_5_0.max().item():.6f}, mean={y_5_0.mean().item():.6f}")
                    print(f"  y first 5: {y_5_0[0, 0, :5].tolist()}")
                    print(f"{'='*80}\n")
        # ===== END DEBUG =====

        # === Step 5: had + out_proj ===
        y_5_0_fp16 = y_5_0.half() if y_5_0.dtype != torch.float16 else y_5_0
        y_5_0 = self.had(y_5_0_fp16)
        out_5_0 = self.out_proj(y_5_0)

        return out_5_0

    def forward_mode5_1(self, hidden_states, inference_params=None):
        """
        Mode 5-1: FP32 路径 (高精度路径)
        - Conv1D: INT8 输入 → FP32 输出
        - SSM: FP32 输入
        - 用于误差累积比较的 FP32 基准路径
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)  # (B, D, L) INT8
        x_5_1, z_5_1 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        # [Scale对比] fwd_mode5 使用和 Mode 0 相同的 scales:
        #   - input_scale: self.conv1d.input_scale (同 Mode 0, qConvLayer.py:174)
        #   - weight_scale: self.conv1d.weight_scale (同 Mode 0, qConvLayer.py:175)
        #   - bias_scale: self.conv1d.bias_scale (同 Mode 0, qConvLayer.py:177)
        #   区别: fwd_mode5 返回 FP32, Mode 0 的 fwd 返回 INT8 (内部用 output_scale 量化)
        x_5_1_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_5_1, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32 输出

        # === Step 3: x_proj, dt_proj (需要 INT8 输入，重新量化) ===
        # [Scale对比] Mode 5-1 使用 self.conv1d.output_scale 做截断
        #   - Mode 0: qConvLayer.py:171-179 CUDA kernel 内部用 output_scale 做量化+截断
        #             quant_causal_conv1d_cuda.fwd(..., self.output_scale, ...) → 返回 INT8
        #   - Mode 5-1: Python 端用同样的 output_scale 做 re-quantize
        #   - 两者用的是同一个 scale 值: self.conv1d.output_scale
        #   - 两者都会把超出 [-128, 127] 的 outlier clamp 截断
        # 使用 roundf_like 模拟 C++ roundf()
        def roundf_like(x):
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
        x_5_1_int8_for_xproj = roundf_like(x_5_1_fp32 / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
        x_5_1_reshape = rearrange(x_5_1_int8_for_xproj, "b d l -> b l d").contiguous()
        x_dbl_5_1 = self.x_proj(x_5_1_reshape)
        dt_5_1, B_5_1, C_5_1 = torch.split(x_dbl_5_1, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_5_1 = self.dt_proj.to_seqlen_last(dt_5_1.contiguous())
        B_5_1 = rearrange(B_5_1, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_5_1 = rearrange(C_5_1, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 4: SSM (FP32 输入) ===
        # [Scale对比] fwd_mode5 使用相同的 selective_scan scales:
        #   - Mode 0: self.selective_scan.forward(x, dt, B, C, z=z) 使用 quant_sscan_cuda.fwd
        #             内部使用同样的 dt_scale, A_scale, B_scale, C_scale, etc.
        #   - Mode 5-1: 直接调用 quant_sscan_cuda.fwd_mode5, 传入相同的 scales
        #   - 区别: fwd_mode5 接受 FP32 输入 x_5_1_fp32 (保留了 outlier 精度)
        #           Mode 0 的 fwd 接受 INT8 输入 (outlier 已被截断)
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_5_1_ssm_out, _ = quant_sscan_cuda.fwd_mode5(
            x_5_1_fp32,  # FP32 输入 (保留 outlier 精度, 与 Mode 0 不同)
            dt_5_1, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_5_1, ensure_shape_1(self.selective_scan.B_scale),
            C_5_1, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_5_1, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True  # delta_softplus
        )

        # === Step 5: had + out_proj ===
        y_5_1_fp16 = y_5_1_ssm_out.half() if y_5_1_ssm_out.dtype != torch.float16 else y_5_1_ssm_out
        y_5_1 = self.had(y_5_1_fp16)
        out_5_1 = self.out_proj(y_5_1)

        # ===== DEBUG: Mode 5-1 数值打印 =====
        import os
        if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
            # 计算 overflow 统计 (与 x_proj 输入相关)
            x_5_1_scaled = x_5_1_fp32 / self.conv1d.output_scale
            overflow_mask = torch.abs(x_5_1_scaled) > 127
            overflow_count = overflow_mask.sum().item()
            total_count = overflow_mask.numel()
            overflow_ratio = overflow_count / total_count * 100

            # 累积所有层的 overflow 统计
            if not hasattr(self, '_mode51_overflow_stats'):
                self._mode51_overflow_stats = {'overflow': 0, 'total': 0}
            self._mode51_overflow_stats['overflow'] += overflow_count
            self._mode51_overflow_stats['total'] += total_count

            # 最后一层打印详细信息
            last_layer_idx = 63  # 2.8b=63, 1.4b=47, 130m=23
            if self.layer_idx == last_layer_idx:
                if not hasattr(self, '_mode51_debug_count'):
                    self._mode51_debug_count = 0
                self._mode51_debug_count += 1
                if self._mode51_debug_count == 1:  # 只打印第一次
                    print(f"\n{'='*80}")
                    print(f"[Mode 5-1 - Layer {last_layer_idx+1}] Debug Output (FP32 Path)")
                    print(f"{'='*80}")
                    # 全局 overflow 统计
                    global_overflow_ratio = self._mode51_overflow_stats['overflow'] / self._mode51_overflow_stats['total'] * 100
                    print(f"\n--- Overflow Statistics (All Layers) ---")
                    print(f"  Total overflow: {self._mode51_overflow_stats['overflow']} / {self._mode51_overflow_stats['total']} = {global_overflow_ratio:.4f}%")
                    print(f"  NOTE: Mode 5-1 SSM uses raw FP32, overflow only affects x_proj input")
                    # Conv1D 输出
                    print(f"\n--- Conv1D Output (Layer {last_layer_idx+1}) ---")
                    print(f"  x_fp32 raw: min={x_5_1_fp32.min().item():.4f}, max={x_5_1_fp32.max().item():.4f}, mean={x_5_1_fp32.mean().item():.4f}, std={x_5_1_fp32.std().item():.4f}")
                    print(f"  x_fp32 first 5: {x_5_1_fp32[0, 0, :5].tolist()}")
                    print(f"  x_int8 (for x_proj) first 5: {x_5_1_int8_for_xproj[0, 0, :5].tolist()}")
                    print(f"  Layer overflow: {overflow_count} / {total_count} = {overflow_ratio:.4f}%")
                    # 关键区别：SSM 输入对比
                    x_5_1_virtual_int8 = roundf_like(x_5_1_scaled).clamp(-128, 127) * self.conv1d.output_scale
                    ssm_input_diff = (x_5_1_fp32 - x_5_1_virtual_int8).abs()
                    print(f"\n--- SSM Input Comparison (Mode 5-1 vs Mode 5-0) ---")
                    print(f"  Mode 5-1 SSM input (raw FP32): first 5 = {x_5_1_fp32[0, 0, :5].tolist()}")
                    print(f"  Mode 5-0 would use (virtual INT8): first 5 = {x_5_1_virtual_int8[0, 0, :5].tolist()}")
                    print(f"  Difference (|FP32 - VirtualINT8|): max={ssm_input_diff.max().item():.6f}, mean={ssm_input_diff.mean().item():.6f}")
                    # Scales
                    print(f"\n--- Scales ---")
                    print(f"  conv1d.input_scale:  {self.conv1d.input_scale:.10f}")
                    print(f"  conv1d.output_scale: {self.conv1d.output_scale:.10f}")
                    print(f"  x_proj.a (input_scale): {self.x_proj.a.item():.10f}")
                    # x_proj 输出
                    print(f"\n--- x_proj/dt_proj Output ---")
                    print(f"  dt first 5: {dt_5_1[0, 0, :5].tolist()}")
                    print(f"  B first 5: {B_5_1[0, 0, 0, :5].tolist()}")
                    print(f"  C first 5: {C_5_1[0, 0, 0, :5].tolist()}")
                    # SSM 输出 (使用 had() 之前的原始输出)
                    print(f"\n--- SSM Output (before had()) ---")
                    print(f"  y_ssm_out dtype: {y_5_1_ssm_out.dtype}")
                    print(f"  y_ssm_out: min={y_5_1_ssm_out.min().item():.6f}, max={y_5_1_ssm_out.max().item():.6f}, mean={y_5_1_ssm_out.mean().item():.6f}")
                    print(f"  y_ssm_out first 5: {y_5_1_ssm_out[0, 0, :5].tolist()}")
                    print(f"{'='*80}\n")
        # ===== END DEBUG =====

        return out_5_1

    def forward_mode5_2(self, hidden_states, inference_params=None):
        """
        Mode 5-2: 虚拟量化 + Outlier 路径
        - Conv1D: INT8 输入 → FP32 输出 → 虚拟量化 + outlier
        - SSM: FP32 输入（混合：网格值 + outlier 保持原 FP32）
        - 用于研究 outlier 保护对精度的影响
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)  # (B, D, L) INT8
        x_5_2, z_5_2 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_5_2_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_5_2, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32 输出

        # === Step 3: 虚拟量化 + Outlier (5-2 独有) ===
        # 使用 roundf_like 模拟 C++ roundf()
        def roundf_like(x):
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
        # 量化到整数网格
        x_quantized = roundf_like(x_5_2_fp32 / self.conv1d.output_scale)
        # 检测 outlier（超出 INT8 范围）
        overflow_mask_5_2 = torch.abs(x_quantized) > 127
        # 网格值 = clamp + 反量化
        x_grid = torch.clamp(x_quantized, -128, 127) * self.conv1d.output_scale
        # 混合输出 - outlier 保持 FP32，正常值用网格
        x_5_2_mixed = torch.where(overflow_mask_5_2, x_5_2_fp32, x_grid)

        # 保存 outlier 统计供打印
        self._mode5_2_overflow_count = overflow_mask_5_2.sum().item()
        self._mode5_2_total_count = overflow_mask_5_2.numel()

        # === Step 4: x_proj, dt_proj (重新量化为 INT8) ===
        x_5_2_int8_for_xproj = roundf_like(x_5_2_mixed / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
        x_5_2_reshape = rearrange(x_5_2_int8_for_xproj, "b d l -> b l d").contiguous()
        x_dbl_5_2 = self.x_proj(x_5_2_reshape)
        dt_5_2, B_5_2, C_5_2 = torch.split(x_dbl_5_2, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_5_2 = self.dt_proj.to_seqlen_last(dt_5_2.contiguous())
        B_5_2 = rearrange(B_5_2, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_5_2 = rearrange(C_5_2, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 5: SSM (FP32 输入，用混合值) ===
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_5_2, _ = quant_sscan_cuda.fwd_mode5(
            x_5_2_mixed,  # FP32 混合输入（网格值 + outlier FP32）
            dt_5_2, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_5_2, ensure_shape_1(self.selective_scan.B_scale),
            C_5_2, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_5_2, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True  # delta_softplus
        )

        # === Step 6: had + out_proj ===
        y_5_2_fp16 = y_5_2.half() if y_5_2.dtype != torch.float16 else y_5_2
        y_5_2 = self.had(y_5_2_fp16)
        out_5_2 = self.out_proj(y_5_2)

        return out_5_2

    def forward_mode5_3(self, hidden_states, inference_params=None):
        """
        Mode 5-3: 双精度 INT8 虚拟量化路径（动态 outlier scale）
        - Conv1D: INT8 输入 → FP32 输出 → 双精度虚拟量化
        - SSM: FP32 输入（所有值都在 INT8 网格上，但用不同 scale）
        - 与 Mode 0 CUDA kernel 保持一致：round + clamp
        - 动态计算 outlier scale：根据 outlier 最大值，让其刚好填满 [-127, 127]
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)  # (B, D, L) INT8
        x_5_3, z_5_3 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_5_3_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_5_3, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32 输出

        # === Step 3: 双精度 INT8 虚拟量化 (5-3 独有，动态 scale) ===
        # 使用 roundf_like 模拟 C++ roundf()
        def roundf_like(x):
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

        # Step 3.1: 使用原始 scale 量化
        x_quantized = roundf_like(x_5_3_fp32 / self.conv1d.output_scale)

        # Step 3.2: 检测 outlier（超出 INT8 范围）
        overflow_mask_1x = torch.abs(x_quantized) > 127

        # Step 3.3: 正常值 → 原始 scale 的 INT8 网格
        x_normal = torch.clamp(x_quantized, -128, 127) * self.conv1d.output_scale

        # Step 3.4: 动态计算 outlier scale（关键改进）
        # 目标：让 outlier 刚好填满 [-127, 127] 范围，最大化精度
        outlier_values = x_5_3_fp32[overflow_mask_1x]
        if outlier_values.numel() > 0:
            outlier_max = outlier_values.abs().max()
            # 让 outlier 刚好填满 [-127, 127] 范围
            outlier_scale = (outlier_max / 127.0).clamp(min=self.conv1d.output_scale)
            scale_factor = (outlier_scale / self.conv1d.output_scale).item()
        else:
            outlier_scale = self.conv1d.output_scale  # 无 outlier，fallback
            scale_factor = 1.0

        # Step 3.5: Outlier → 动态 scale 的 INT8 网格
        x_outlier_quantized = roundf_like(x_5_3_fp32 / outlier_scale)

        # Step 3.6: 检测二次溢出（理论上不应该有，因为 scale 是动态计算的）
        overflow_mask_2x = torch.abs(x_outlier_quantized) > 127

        # Step 3.7: Outlier 值 → 动态 scale 的 INT8 网格
        x_outlier = torch.clamp(x_outlier_quantized, -128, 127) * outlier_scale

        # Step 3.8: 混合输出（全部是 FP32 dtype，但值在不同精度的 INT8 网格上）
        x_5_3_mixed = torch.where(overflow_mask_1x, x_outlier, x_normal)

        # 保存统计信息供打印
        self._mode5_3_overflow_1x_count = overflow_mask_1x.sum().item()
        self._mode5_3_overflow_2x_count = (overflow_mask_1x & overflow_mask_2x).sum().item()
        self._mode5_3_total_count = overflow_mask_1x.numel()
        self._mode5_3_scale_factor = scale_factor

        # === Step 4: x_proj, dt_proj (重新量化为 INT8) ===
        x_5_3_int8_for_xproj = roundf_like(x_5_3_mixed / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
        x_5_3_reshape = rearrange(x_5_3_int8_for_xproj, "b d l -> b l d").contiguous()
        x_dbl_5_3 = self.x_proj(x_5_3_reshape)
        dt_5_3, B_5_3, C_5_3 = torch.split(x_dbl_5_3, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_5_3 = self.dt_proj.to_seqlen_last(dt_5_3.contiguous())
        B_5_3 = rearrange(B_5_3, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_5_3 = rearrange(C_5_3, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 5: SSM (FP32 输入，用混合值) ===
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_5_3, _ = quant_sscan_cuda.fwd_mode5(
            x_5_3_mixed,  # FP32 混合输入（全部在 INT8 网格上，但不同 scale）
            dt_5_3, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_5_3, ensure_shape_1(self.selective_scan.B_scale),
            C_5_3, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_5_3, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True  # delta_softplus
        )

        # === Step 6: had + out_proj ===
        y_5_3_fp16 = y_5_3.half() if y_5_3.dtype != torch.float16 else y_5_3
        y_5_3 = self.had(y_5_3_fp16)
        out_5_3 = self.out_proj(y_5_3)

        return out_5_3

    def forward_mode5_4(self, hidden_states, inference_params=None):
        """
        Mode 5-4: QuarterScale 4× Precision for Small Values
        - Conv1D: INT8 → FP32 输出
        - QuarterScale: 小值 (|q| < 32) 用 scale/4 (4× 精度), 大值用正常 scale
        - x_proj: 重新量化回 INT8 (与 Mode 5-0 一致)
        - SSM: FP32 mixed 输入 (保留小值 4× 精度)

        关键设计:
        - 小值占 INT8 范围的 1/4 (|q| < 32)
        - 小值用 quarter_scale = scale/4，获得 4× 精度
        - x_proj 精度与 Mode 5-0 完全一致（退化回原始精度）
        - 只有 SSM 输入获得小值的 4× 精度
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x_5_4, z_5_4 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_5_4_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_5_4, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )

        # === Step 3: QuarterScale Virtual Quantization ===
        # 使用 roundf_like 模拟 C++ roundf()
        def roundf_like(x):
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

        scale = self.conv1d.output_scale
        quarter_scale = scale / 4.0

        # 用原始 scale 判断值的大小
        x_quantized = roundf_like(x_5_4_fp32 / scale)

        # 小值: |q| < 32 (范围的 1/4)
        is_small = (x_quantized.abs() < 32)

        # 小值用 quarter_scale (4× 精度)
        x_small = roundf_like(x_5_4_fp32 / quarter_scale).clamp(-127, 127) * quarter_scale

        # 大值用原始 scale
        x_normal = x_quantized.clamp(-128, 127) * scale

        # 组合
        x_5_4_mixed = torch.where(is_small, x_small, x_normal)

        # === Step 4: x_proj (重新量化回 INT8，与 Mode 5-0 一致) ===
        x_5_4_int8 = roundf_like(x_5_4_mixed / scale).clamp(-128, 127).to(torch.int8)
        x_5_4_reshape = rearrange(x_5_4_int8, "b d l -> b l d").contiguous()
        x_dbl_5_4 = self.x_proj(x_5_4_reshape)
        dt_5_4, B_5_4, C_5_4 = torch.split(x_dbl_5_4, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_5_4 = self.dt_proj.to_seqlen_last(dt_5_4.contiguous())
        B_5_4 = rearrange(B_5_4, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_5_4 = rearrange(C_5_4, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 5: SSM (FP32 mixed 输入，保留小值 4× 精度) ===
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_5_4, _ = quant_sscan_cuda.fwd_mode5(
            x_5_4_mixed,  # FP32 输入 (小值 4× 精度)
            dt_5_4, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_5_4, ensure_shape_1(self.selective_scan.B_scale),
            C_5_4, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_5_4, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True  # delta_softplus
        )

        # ===== DEBUG: Mode 5-4 统一调试输出 =====
        import os
        if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
            # 统计 overflow (理论上，用原始 scale 量化超出 INT8 范围的值)
            overflow_mask = torch.abs(x_quantized) > 127
            overflow_count = overflow_mask.sum().item()
            total_count = overflow_mask.numel()
            overflow_ratio = overflow_count / total_count * 100

            # 统计 QuarterScale 小值比例
            small_value_count = is_small.sum().item()
            small_value_ratio = small_value_count / total_count * 100

            # 累积所有层的统计
            if not hasattr(self, '_mode54_overflow_stats'):
                self._mode54_overflow_stats = {'overflow': 0, 'total': 0, 'small': 0}
            self._mode54_overflow_stats['overflow'] += overflow_count
            self._mode54_overflow_stats['total'] += total_count
            self._mode54_overflow_stats['small'] += small_value_count

            # 最后一层打印详细信息
            last_layer_idx = 47  # 1.4b=47, 2.8b=63, 130m=23
            if self.layer_idx == last_layer_idx:
                if not hasattr(self, '_mode54_debug_count'):
                    self._mode54_debug_count = 0
                self._mode54_debug_count += 1
                if self._mode54_debug_count == 1:  # 只打印第一次
                    print(f"\n{'='*80}")
                    print(f"[Mode 5-4 - Layer {last_layer_idx+1}] Debug Output (QuarterScale)")
                    print(f"{'='*80}")
                    # 全局统计
                    global_overflow_ratio = self._mode54_overflow_stats['overflow'] / self._mode54_overflow_stats['total'] * 100
                    global_small_ratio = self._mode54_overflow_stats['small'] / self._mode54_overflow_stats['total'] * 100
                    print(f"\n--- Overflow & Small Value Statistics (All Layers) ---")
                    print(f"  Total overflow: {self._mode54_overflow_stats['overflow']} / {self._mode54_overflow_stats['total']} = {global_overflow_ratio:.4f}%")
                    print(f"  Small values (|q|<32, 4x precision): {self._mode54_overflow_stats['small']} / {self._mode54_overflow_stats['total']} = {global_small_ratio:.2f}%")
                    # Conv1D 输出
                    print(f"\n--- Conv1D Output (Layer {last_layer_idx+1}) ---")
                    print(f"  x_fp32 raw: min={x_5_4_fp32.min().item():.4f}, max={x_5_4_fp32.max().item():.4f}, mean={x_5_4_fp32.mean().item():.4f}, std={x_5_4_fp32.std().item():.4f}")
                    print(f"  x_fp32 first 5: {x_5_4_fp32[0, 0, :5].tolist()}")
                    print(f"  x_mixed (after QuarterScale) first 5: {x_5_4_mixed[0, 0, :5].tolist()}")
                    print(f"  x_int8 (for x_proj) first 5: {x_5_4_int8[0, 0, :5].tolist()}")
                    print(f"  Layer overflow: {overflow_count} / {total_count} = {overflow_ratio:.4f}%")
                    print(f"  Layer small values: {small_value_count} / {total_count} = {small_value_ratio:.2f}%")
                    # Scales
                    print(f"\n--- Scales ---")
                    print(f"  conv1d.input_scale:  {self.conv1d.input_scale:.10f}")
                    print(f"  conv1d.output_scale: {scale:.10f}")
                    print(f"  quarter_scale: {quarter_scale:.10f}")
                    print(f"  x_proj.a (input_scale): {self.x_proj.a.item():.10f}")
                    # x_proj 输出
                    print(f"\n--- x_proj/dt_proj Output ---")
                    print(f"  dt first 5: {dt_5_4[0, 0, :5].tolist()}")
                    print(f"  B first 5: {B_5_4[0, 0, 0, :5].tolist()}")
                    print(f"  C first 5: {C_5_4[0, 0, 0, :5].tolist()}")
                    # SSM 输出
                    print(f"\n--- SSM Output ---")
                    print(f"  y: min={y_5_4.min().item():.6f}, max={y_5_4.max().item():.6f}, mean={y_5_4.mean().item():.6f}")
                    print(f"  y first 5: {y_5_4[0, 0, :5].tolist()}")
                    print(f"{'='*80}\n")
        # ===== END DEBUG =====

        # === Step 6: had + out_proj ===
        y_5_4_fp16 = y_5_4.half() if y_5_4.dtype != torch.float16 else y_5_4
        y_5_4 = self.had(y_5_4_fp16)
        out_5_4 = self.out_proj(y_5_4)

        return out_5_4

    def forward_mode5_6(self, hidden_states, inference_params=None):
        """
        Mode 5-6: QuarterScale + VirtualBiasedLog2Quant
        - Conv1D: INT8 → FP32 输出
        - QuarterLog2: 小值 (|q| < 32) 用 scale/4 (4× 精度), 大值用 8-bit Log2 (可表示极大值)
        - x_proj: 重新量化回 INT8 (与 Mode 5-0 一致)
        - SSM: FP32 mixed 输入

        关键设计:
        - 高精度区域: |x| <= 32 * (scale/4) = 8*scale，用 scale/4 (4× 精度)
        - Log2区域: |x| > 8*scale，用 x_quant = sign(x) * (max_quarter + 2^k * log_scale)
        - k 是 8 bits (0-255)
        - log_scale = scale/4 (与 quarter_scale 相同，保证平滑过渡)
        - x_proj 精度与 Mode 5-0 完全一致（退化回原始精度）
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x_5_6, z_5_6 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_5_6_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_5_6, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )

        # === Step 3: QuarterLog2 Virtual Quantization ===
        # 使用 roundf_like 模拟 C++ roundf()
        def roundf_like(x):
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

        scale = self.conv1d.output_scale
        quarter_scale = scale / 4.0
        max_quarter = 32 * quarter_scale  # = 8 * scale (高精度区域边界)
        log_scale = quarter_scale  # 与 quarter_scale 相同，保证平滑过渡

        abs_x = torch.abs(x_5_6_fp32)
        sign_x = torch.sign(x_5_6_fp32)

        # 高精度区域: |x| <= max_quarter
        is_small = abs_x <= max_quarter
        x_small = roundf_like(x_5_6_fp32 / quarter_scale).clamp(-127, 127) * quarter_scale

        # Log2区域: |x| > max_quarter
        # x_quant = sign(x) * (max_quarter + 2^k × log_scale)
        # 其中 k = floor(log2((|x| - max_quarter) / log_scale))
        residual = abs_x - max_quarter
        # 避免 log2(0) 或 log2(负数)，对于 is_small 的元素，residual 可能为负
        residual_safe = torch.clamp(residual, min=log_scale)
        k = torch.floor(torch.log2(residual_safe / log_scale))
        k = torch.clamp(k, min=0, max=255)  # 8 bits
        x_log2 = sign_x * (max_quarter + (2.0 ** k) * log_scale)

        # 组合: 小值用 QuarterScale，大值用 Log2
        x_5_6_mixed = torch.where(is_small, x_small, x_log2)

        # === Step 4: x_proj (重新量化回 INT8，与 Mode 5-0 一致) ===
        x_5_6_int8 = roundf_like(x_5_6_mixed / scale).clamp(-128, 127).to(torch.int8)
        x_5_6_reshape = rearrange(x_5_6_int8, "b d l -> b l d").contiguous()
        x_dbl_5_6 = self.x_proj(x_5_6_reshape)
        dt_5_6, B_5_6, C_5_6 = torch.split(x_dbl_5_6, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_5_6 = self.dt_proj.to_seqlen_last(dt_5_6.contiguous())
        B_5_6 = rearrange(B_5_6, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_5_6 = rearrange(C_5_6, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 5: SSM (FP32 mixed 输入) ===
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_5_6, _ = quant_sscan_cuda.fwd_mode5(
            x_5_6_mixed,  # FP32 输入 (QuarterLog2 量化后)
            dt_5_6, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_5_6, ensure_shape_1(self.selective_scan.B_scale),
            C_5_6, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_5_6, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True  # delta_softplus
        )

        # ===== DEBUG: Mode 5-6 统一调试输出 =====
        import os
        if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
            # 用原始 scale 判断 overflow (理论上，因为 Log2 可表示任意大值，不应该有 overflow)
            x_quantized_orig = roundf_like(x_5_6_fp32 / scale)
            overflow_mask = torch.abs(x_quantized_orig) > 127
            overflow_count = overflow_mask.sum().item()
            total_count = overflow_mask.numel()
            overflow_ratio = overflow_count / total_count * 100

            # 统计 QuarterScale 小值比例
            small_value_count = is_small.sum().item()
            small_value_ratio = small_value_count / total_count * 100

            # 统计 Log2 区域的 k 值分布
            log2_mask = ~is_small
            log2_count = log2_mask.sum().item()
            if log2_count > 0:
                k_values = k[log2_mask]
                k_mean = k_values.mean().item()
                k_max = k_values.max().item()
            else:
                k_mean = 0
                k_max = 0

            # 累积所有层的统计
            if not hasattr(self, '_mode56_overflow_stats'):
                self._mode56_overflow_stats = {'overflow': 0, 'total': 0, 'small': 0, 'log2': 0, 'k_sum': 0}
            self._mode56_overflow_stats['overflow'] += overflow_count
            self._mode56_overflow_stats['total'] += total_count
            self._mode56_overflow_stats['small'] += small_value_count
            self._mode56_overflow_stats['log2'] += log2_count

            # 最后一层打印详细信息
            last_layer_idx = 47  # 1.4b=47, 2.8b=63, 130m=23
            if self.layer_idx == last_layer_idx:
                if not hasattr(self, '_mode56_debug_count'):
                    self._mode56_debug_count = 0
                self._mode56_debug_count += 1
                if self._mode56_debug_count == 1:  # 只打印第一次
                    print(f"\n{'='*80}")
                    print(f"[Mode 5-6 - Layer {last_layer_idx+1}] Debug Output (QuarterLog2)")
                    print(f"{'='*80}")
                    # 全局统计
                    global_overflow_ratio = self._mode56_overflow_stats['overflow'] / self._mode56_overflow_stats['total'] * 100
                    global_small_ratio = self._mode56_overflow_stats['small'] / self._mode56_overflow_stats['total'] * 100
                    global_log2_ratio = self._mode56_overflow_stats['log2'] / self._mode56_overflow_stats['total'] * 100
                    print(f"\n--- QuarterLog2 Statistics (All Layers) ---")
                    print(f"  Original overflow (if using scale): {self._mode56_overflow_stats['overflow']} / {self._mode56_overflow_stats['total']} = {global_overflow_ratio:.4f}%")
                    print(f"  QuarterScale region (|x|<=8*scale): {self._mode56_overflow_stats['small']} / {self._mode56_overflow_stats['total']} = {global_small_ratio:.2f}%")
                    print(f"  Log2 region (|x|>8*scale): {self._mode56_overflow_stats['log2']} / {self._mode56_overflow_stats['total']} = {global_log2_ratio:.2f}%")
                    # Conv1D 输出
                    print(f"\n--- Conv1D Output (Layer {last_layer_idx+1}) ---")
                    print(f"  x_fp32 raw: min={x_5_6_fp32.min().item():.4f}, max={x_5_6_fp32.max().item():.4f}, mean={x_5_6_fp32.mean().item():.4f}, std={x_5_6_fp32.std().item():.4f}")
                    print(f"  x_fp32 first 5: {x_5_6_fp32[0, 0, :5].tolist()}")
                    print(f"  x_mixed (after QuarterLog2) first 5: {x_5_6_mixed[0, 0, :5].tolist()}")
                    print(f"  x_int8 (for x_proj) first 5: {x_5_6_int8[0, 0, :5].tolist()}")
                    print(f"  Layer overflow: {overflow_count} / {total_count} = {overflow_ratio:.4f}%")
                    print(f"  Layer small values: {small_value_count} / {total_count} = {small_value_ratio:.2f}%")
                    print(f"  Layer log2 values: {log2_count} / {total_count}, k_mean={k_mean:.2f}, k_max={k_max:.0f}")
                    # Scales
                    print(f"\n--- Scales ---")
                    print(f"  conv1d.output_scale: {scale:.10f}")
                    print(f"  quarter_scale (scale/4): {quarter_scale:.10f}")
                    print(f"  max_quarter (32*quarter_scale): {max_quarter:.10f}")
                    print(f"  log_scale (=quarter_scale): {log_scale:.10f}")
                    # x_proj 输出
                    print(f"\n--- x_proj/dt_proj Output ---")
                    print(f"  dt first 5: {dt_5_6[0, 0, :5].tolist()}")
                    print(f"  B first 5: {B_5_6[0, 0, 0, :5].tolist()}")
                    print(f"  C first 5: {C_5_6[0, 0, 0, :5].tolist()}")
                    # SSM 输出
                    print(f"\n--- SSM Output ---")
                    print(f"  y: min={y_5_6.min().item():.6f}, max={y_5_6.max().item():.6f}, mean={y_5_6.mean().item():.6f}")
                    print(f"  y first 5: {y_5_6[0, 0, :5].tolist()}")
                    print(f"{'='*80}\n")
        # ===== END DEBUG =====

        # === Step 6: had + out_proj ===
        y_5_6_fp16 = y_5_6.half() if y_5_6.dtype != torch.float16 else y_5_6
        y_5_6 = self.had(y_5_6_fp16)
        out_5_6 = self.out_proj(y_5_6)

        return out_5_6

    def forward_mode5_7(self, hidden_states, inference_params=None):
        """
        Mode 5-7: Three-Segment Uniform Quantization (三段均匀量化)
        - Conv1D: INT8 → FP32 输出
        - Three-Segment:
          - High precision: scale/4 (0.25x), ±63 levels, range [-15.75*s, +15.75*s] where s=scale
          - Mid precision:  scale (1x),     ±127 levels, range ±[16*s, 143*s]
          - Low precision:  scale*4 (4x),   ±63 levels, range ±[144*s, 396*s]
        - x_proj: 重新量化回 INT8 (与 Mode 5-0 一致)
        - SSM: FP32 mixed 输入

        设计思路:
        - 三段均匀步长，每段都是均匀量化
        - 总量化级别: 63 + 127 + 63 = 253，可用 8-bit 编码
        - 小值区域 4x 精度，中值区域原始精度，大值区域 1/4 精度但覆盖更大范围
        - 相比 5-6 的 Log2，5-7 中大值区域是均匀步长而非指数增长
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x_5_7, z_5_7 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_5_7_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_5_7, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )

        # === Step 3: Three-Segment Virtual Quantization ===
        def roundf_like(x):
            return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

        scale = self.conv1d.output_scale
        high_scale = scale / 4.0   # 高精度区域步长
        mid_scale = scale          # 中精度区域步长 (原始)
        low_scale = scale * 4.0    # 低精度区域步长

        # 区域边界 (以 scale=1 为例):
        # - 高精度: |x| <= 15.75 (即 63 * 0.25), q ∈ [-63, 63]
        # - 中精度: 16 <= |x| <= 143 (即 q ∈ [16, 143], 共 128 级)
        # - 低精度: |x| > 143, q ∈ [0, 63], 偏移从 mid_max 开始
        high_max = 63 * high_scale   # = 15.75 * scale
        mid_max = 143 * mid_scale    # = 143 * scale

        abs_x = torch.abs(x_5_7_fp32)
        sign_x = torch.sign(x_5_7_fp32)

        # 高精度区域: |x| <= high_max
        is_high = abs_x <= high_max
        x_high = roundf_like(x_5_7_fp32 / high_scale).clamp(-63, 63) * high_scale

        # 中精度区域: high_max < |x| <= mid_max
        # 直接用原始 scale 量化，q 范围自然在 [16, 143]
        is_mid = (abs_x > high_max) & (abs_x <= mid_max)
        q_mid_raw = roundf_like(abs_x / mid_scale)  # 直接用原始 scale 量化
        q_mid = q_mid_raw.clamp(16, 143)  # 限制在中精度区域范围
        x_mid = sign_x * q_mid * mid_scale

        # 低精度区域: |x| > mid_max
        # 偏移从 mid_max 开始，保证连续性
        is_low = abs_x > mid_max
        offset_from_mid_max = abs_x - mid_max
        q_low = roundf_like(offset_from_mid_max / low_scale).clamp(0, 63)  # 0-63 levels
        x_low = sign_x * (mid_max + q_low * low_scale)

        # 组合三段
        x_5_7_mixed = torch.where(is_high, x_high,
                     torch.where(is_mid, x_mid, x_low))

        # === Step 4: x_proj (重新量化回 INT8，与 Mode 5-0 一致) ===
        x_5_7_int8 = roundf_like(x_5_7_mixed / scale).clamp(-128, 127).to(torch.int8)
        x_5_7_reshape = rearrange(x_5_7_int8, "b d l -> b l d").contiguous()
        x_dbl_5_7 = self.x_proj(x_5_7_reshape)
        dt_5_7, B_5_7, C_5_7 = torch.split(x_dbl_5_7, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_5_7 = self.dt_proj.to_seqlen_last(dt_5_7.contiguous())
        B_5_7 = rearrange(B_5_7, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_5_7 = rearrange(C_5_7, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 5: SSM (FP32 mixed 输入) ===
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_5_7, _ = quant_sscan_cuda.fwd_mode5(
            x_5_7_mixed,  # FP32 输入 (ThreeSegment 量化后)
            dt_5_7, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_5_7, ensure_shape_1(self.selective_scan.B_scale),
            C_5_7, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_5_7, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True  # delta_softplus
        )

        # ===== DEBUG: Mode 5-7 统一调试输出 =====
        import os
        if os.environ.get('DEBUG_MODE50_VS_MODE51', 'false').lower() == 'true':
            # 统计各区域比例
            high_count = is_high.sum().item()
            mid_count = is_mid.sum().item()
            low_count = is_low.sum().item()
            total_count = abs_x.numel()

            # 用原始 scale 判断 overflow
            x_quantized_orig = roundf_like(x_5_7_fp32 / scale)
            overflow_mask = torch.abs(x_quantized_orig) > 127
            overflow_count = overflow_mask.sum().item()

            # 累积所有层的统计
            if not hasattr(self, '_mode57_stats'):
                self._mode57_stats = {'high': 0, 'mid': 0, 'low': 0, 'total': 0, 'overflow': 0}
            self._mode57_stats['high'] += high_count
            self._mode57_stats['mid'] += mid_count
            self._mode57_stats['low'] += low_count
            self._mode57_stats['total'] += total_count
            self._mode57_stats['overflow'] += overflow_count

            # 最后一层打印详细信息
            last_layer_idx = 23  # 130m=23, 1.4b=47, 2.8b=63
            if self.layer_idx == last_layer_idx:
                if not hasattr(self, '_mode57_debug_count'):
                    self._mode57_debug_count = 0
                self._mode57_debug_count += 1
                if self._mode57_debug_count == 1:  # 只打印第一次
                    print(f"\n{'='*80}")
                    print(f"[Mode 5-7 - Layer {last_layer_idx+1}] Debug Output (ThreeSegment)")
                    print(f"{'='*80}")
                    # 全局统计
                    global_high_ratio = self._mode57_stats['high'] / self._mode57_stats['total'] * 100
                    global_mid_ratio = self._mode57_stats['mid'] / self._mode57_stats['total'] * 100
                    global_low_ratio = self._mode57_stats['low'] / self._mode57_stats['total'] * 100
                    global_overflow_ratio = self._mode57_stats['overflow'] / self._mode57_stats['total'] * 100
                    print(f"\n--- ThreeSegment Statistics (All Layers) ---")
                    print(f"  Original overflow (if using scale): {global_overflow_ratio:.4f}%")
                    print(f"  High precision region (|x|<=15.75*s): {global_high_ratio:.2f}%")
                    print(f"  Mid precision region (16s<=|x|<=143s): {global_mid_ratio:.2f}%")
                    print(f"  Low precision region (|x|>143s): {global_low_ratio:.2f}%")
                    # Scales
                    print(f"\n--- Scales ---")
                    print(f"  scale: {scale:.10f}")
                    print(f"  high_scale (scale/4): {high_scale:.10f}")
                    print(f"  mid_scale (scale): {mid_scale:.10f}")
                    print(f"  low_scale (scale*4): {low_scale:.10f}")
                    # 边界
                    low_max = mid_max + 63 * low_scale  # 最大可表示值
                    print(f"\n--- Boundaries ---")
                    print(f"  high_max: {high_max:.6f} (63 * scale/4)")
                    print(f"  mid_max: {mid_max:.6f} (143 * scale)")
                    print(f"  low_max: {low_max:.6f} (mid_max + 63 * scale*4)")
                    print(f"{'='*80}\n")
        # ===== END DEBUG =====

        # === Step 6: had + out_proj ===
        y_5_7_fp16 = y_5_7.half() if y_5_7.dtype != torch.float16 else y_5_7
        y_5_7 = self.had(y_5_7_fp16)
        out_5_7 = self.out_proj(y_5_7)

        return out_5_7

    def forward_mode6_0_eval(self, hidden_states, inference_params=None):
        """
        Mode 6-0: Virtual INT8 (模拟 Mode 0 的量化误差)
        所有 FP32 值都模拟 INT8 grid：round(x/scale).clamp(-128,127)*scale
        - Conv1D: INT8 kernel → FP32 输出 → Virtual INT8
        - x_proj: FP32 F.linear → Virtual INT8
        - dt_proj: FP32 F.linear → Virtual INT8
        - SSM: mamba_ssm selective_scan_fn (FP32 kernel with Virtual INT8 values)

        注意：由于 SSM 使用 FP32 kernel（mamba_ssm），内部累加精度与 Mode 0 的 INT8 kernel 不同，
        因此结果不会完全一致。6-0 只模拟了输入的量化误差，不模拟 kernel 内部的精度差异。
        预期：6-0 的 perplexity 应介于 Mode 0 和 6-1 之间。
        """
        import quant_causal_conv1d_cuda
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        batch, seqlen, dim = hidden_states.shape

        # === 获取所有需要的 scale ===
        conv1d_output_scale = self.conv1d.output_scale
        x_proj_output_scale = self.selective_scan.B_scale  # = C_scale = x_proj:output
        dt_proj_output_scale = self.selective_scan.dt_scale
        z_scale = self.selective_scan.z_scale
        D_scale = self.selective_scan.D_scale
        dt_bias_scale = self.selective_scan.dt_bias_scale
        A_scale = self.selective_scan.A_scale

        # === Step 1: in_proj (保持 INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x, z = xz.chunk(2, dim=1)  # x, z 都是 INT8

        # === Step 2: Conv1D → FP32，然后 Virtual INT8 ===
        x_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32 输出
        x_vq = torch.round(x_fp32 / conv1d_output_scale).clamp(-128, 127) * conv1d_output_scale  # Virtual INT8

        # Debug: 验证 Virtual INT8 (前 5 次)
        if self.layer_idx == 0:
            if not hasattr(self, '_vq_debug_count'):
                self._vq_debug_count = 0
            if self._vq_debug_count < 5:
                residual = (x_vq / conv1d_output_scale) - torch.round(x_vq / conv1d_output_scale)
                print(f"[6-0 VQ Check] Layer 0, Call {self._vq_debug_count}: max_residual = {residual.abs().max().item():.10f}")
                self._vq_debug_count += 1

        # === Step 3: x_proj (FP32 F.linear) ===
        x_reshape = rearrange(x_vq, "b d l -> (b l) d")
        x_dbl_fp32 = self.x_proj.forward_mode6(x_reshape)  # FP32 输出

        # Virtual INT8 for x_proj output
        x_dbl_vq = torch.round(x_dbl_fp32 / x_proj_output_scale).clamp(-128, 127) * x_proj_output_scale
        x_dbl_vq = x_dbl_vq.view(batch, seqlen, -1)

        # === Step 4: split dt, B, C (都是 Virtual INT8) ===
        dt_raw, B_raw, C_raw = torch.split(x_dbl_vq, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # === Step 5: dt_proj (FP32 F.linear) + Virtual INT8 ===
        dt_fp32 = self.dt_proj.to_seqlen_last_mode6(dt_raw.contiguous())
        dt_vq = torch.round(dt_fp32 / dt_proj_output_scale).clamp(-128, 127) * dt_proj_output_scale

        # === Step 6: B, C reshape (保持 Virtual INT8，已经在 split 时做过 VQ) ===
        B_vq = rearrange(B_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_vq = rearrange(C_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 7: z Virtual INT8 ===
        # z 从 in_proj 出来是 INT8，dequant 后 (z.float() * z_scale) 已经在 INT8 grid 上
        z_fp32 = z.float() * z_scale  # 这就是 Virtual INT8

        # === Step 8: A, D, dt_bias dequant ===
        # A_log 是 INT8 存的 log 值，dequant 后再 exp
        A_fp32 = -torch.exp(self.selective_scan.A_log.float() * A_scale)

        # D 和 dt_bias 是 INT8 weight buffers，dequant 后已经在 INT8 grid 上
        D_fp32 = None
        if self.selective_scan.D is not None:
            D_fp32 = self.selective_scan.D.float() * D_scale  # 已经是 Virtual INT8

        dt_bias_fp32 = None
        if self.selective_scan.dt_bias is not None:
            dt_bias_fp32 = self.selective_scan.dt_bias.float() * dt_bias_scale  # 已经是 Virtual INT8

        # === Step 9: SSM (FP32 kernel with Virtual INT8 values) ===
        y = selective_scan_fn(
            x_vq, dt_vq, A_fp32, B_vq, C_vq,
            D=D_fp32, z=z_fp32, delta_bias=dt_bias_fp32,
            delta_softplus=True, return_last_state=False
        )

        # === Step 10: had + out_proj ===
        y = rearrange(y, "b d l -> b l d")
        y_fp16 = y.half() if y.dtype != torch.float16 else y
        y = self.had(y_fp16)
        out = self.out_proj(y)

        return out

    def forward_mode6_1_eval(self, hidden_states, inference_params=None):
        """
        Mode 6-1: INT8 Kernel + FP32 输出 (无 Virtual INT8)
        - Conv1D: INT8 kernel → FP32 输出 (不做 Virtual INT8!)
        - x_proj: FP32 F.linear → FP32 输出 (不 clamp!)
        - dt_proj: FP32 F.linear → FP32 输出 (不 clamp!)
        - SSM: mamba_ssm selective_scan_fn (FP32 kernel)
        与 6-0 的区别：6-0 每步输出都做 VQ 到 INT8 grid，6-1 直接用 FP32 值
        预期：6-1 的 Acc 应该比 6-0 高（6-1 是 upper bound）
        """
        import quant_causal_conv1d_cuda
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x_6_1, z_6_1 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_6_1_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_6_1, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32 输出

        # === Step 3: x_proj (FP32) - 用与 SSM 完全相同的值！===
        x_6_1_reshape = rearrange(x_6_1_fp32, "b d l -> (b l) d")
        x_dbl_6_1 = self.x_proj.forward_mode6(x_6_1_reshape)
        x_dbl_6_1 = x_dbl_6_1.view(batch, seqlen, -1)
        dt_6_1_raw, B_6_1_raw, C_6_1_raw = torch.split(x_dbl_6_1, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt_proj: FP32 输出, shape (batch, dim, seqlen)
        dt_6_1 = self.dt_proj.to_seqlen_last_mode6(dt_6_1_raw.contiguous())

        # B, C: selective_scan_fn 期望 shape (batch, n_groups, dstate, seqlen)
        # 原始 shape: (batch, seqlen, dstate), 需要转换
        B_6_1 = rearrange(B_6_1_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_6_1 = rearrange(C_6_1_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 4: SSM (FP32 输入) - 使用 mamba_ssm 原始 FP32 kernel ===
        # 获取 A (dequantize A_log: INT8 -> FP32, 然后 exp)
        A_fp32 = -torch.exp(self.selective_scan.A_log.float() * self.selective_scan.A_scale)  # (dim, dstate)

        # z 需要 dequantize: INT8 -> FP32
        z_6_1_fp32 = z_6_1.float() * self.selective_scan.z_scale

        # D: dequantize if needed
        D_fp32 = None
        if self.selective_scan.D is not None:
            D_fp32 = self.selective_scan.D.float() * self.selective_scan.D_scale

        # dt_bias: dequantize
        dt_bias_fp32 = None
        if self.selective_scan.dt_bias is not None:
            dt_bias_fp32 = self.selective_scan.dt_bias.float() * self.selective_scan.dt_bias_scale

        # selective_scan_fn expects:
        #   u: (batch, dim, seqlen), delta: (batch, dim, seqlen)
        #   A: (dim, dstate), B: (batch, n_groups, dstate, seqlen), C: (batch, n_groups, dstate, seqlen)
        #   D: (dim,), z: (batch, dim, seqlen)
        y_6_1 = selective_scan_fn(
            x_6_1_fp32,      # u: (batch, dim, seqlen) FP32
            dt_6_1,          # delta: (batch, dim, seqlen) FP32
            A_fp32,          # A: (dim, dstate) FP32
            B_6_1,           # B: (batch, 1, dstate, seqlen) FP32
            C_6_1,           # C: (batch, 1, dstate, seqlen) FP32
            D=D_fp32,        # D: (dim,) FP32 or None
            z=z_6_1_fp32,    # z: (batch, dim, seqlen) FP32
            delta_bias=dt_bias_fp32,
            delta_softplus=True,
            return_last_state=False
        )

        # === Step 5: had + out_proj ===
        # selective_scan_fn 返回 (batch, dim, seqlen), 需要转换为 (batch, seqlen, dim)
        y_6_1 = rearrange(y_6_1, "b d l -> b l d")
        y_6_1_fp16 = y_6_1.half() if y_6_1.dtype != torch.float16 else y_6_1
        y_6_1 = self.had(y_6_1_fp16)
        out_6_1 = self.out_proj(y_6_1)

        return out_6_1

    def forward_mode6_2_eval(self, hidden_states, inference_params=None):
        """
        Mode 6-2: FP32 + Outlier 保护 (虚拟量化 + outlier 保留原值)
        - Conv1D: INT8 输入 → FP32 输出
        - Virtual Quant: 正常值映射到 INT8 网格，outlier 保留 FP32 原值
        - x_proj & SSM: 用相同的 mixed 值！(使用原始 mamba_ssm selective_scan_fn)
        关键改进：解决 Mode 5-2 的不一致问题
        """
        import quant_causal_conv1d_cuda
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x_6_2, z_6_2 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_6_2_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_6_2, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )

        # === Step 3: Virtual quantization + Outlier 保护 ===
        x_quantized = torch.round(x_6_2_fp32 / self.conv1d.output_scale)
        is_outlier = (x_quantized.abs() > 127)
        x_normal = x_quantized.clamp(-128, 127) * self.conv1d.output_scale
        x_6_2_mixed = torch.where(is_outlier, x_6_2_fp32, x_normal)

        # === Step 4: x_proj (FP32) - 用与 SSM 相同的 mixed 值！===
        x_6_2_reshape = rearrange(x_6_2_mixed, "b d l -> (b l) d")
        x_dbl_6_2 = self.x_proj.forward_mode6(x_6_2_reshape)
        x_dbl_6_2 = x_dbl_6_2.view(batch, seqlen, -1)
        dt_6_2_raw, B_6_2_raw, C_6_2_raw = torch.split(x_dbl_6_2, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt_proj: FP32 输出, shape (batch, dim, seqlen)
        dt_6_2 = self.dt_proj.to_seqlen_last_mode6(dt_6_2_raw.contiguous())

        # B, C: selective_scan_fn 期望 shape (batch, n_groups, dstate, seqlen)
        B_6_2 = rearrange(B_6_2_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_6_2 = rearrange(C_6_2_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 5: SSM (FP32 输入) - 使用 mamba_ssm 原始 FP32 kernel ===
        # 获取 A (dequantize A_log)
        A_fp32 = -torch.exp(self.selective_scan.A_log.float())  # (dim, dstate)

        # z 需要 dequantize: INT8 -> FP32
        z_6_2_fp32 = z_6_2.float() * self.selective_scan.z_scale

        # D: dequantize if needed
        D_fp32 = None
        if self.selective_scan.D is not None:
            D_fp32 = self.selective_scan.D.float() * self.selective_scan.D_scale

        # dt_bias: dequantize
        dt_bias_fp32 = None
        if self.selective_scan.dt_bias is not None:
            dt_bias_fp32 = self.selective_scan.dt_bias.float() * self.selective_scan.dt_bias_scale

        y_6_2 = selective_scan_fn(
            x_6_2_mixed,     # u: (batch, dim, seqlen) FP32
            dt_6_2,          # delta: (batch, dim, seqlen) FP32
            A_fp32,          # A: (dim, dstate) FP32
            B_6_2,           # B: (batch, 1, dstate, seqlen) FP32
            C_6_2,           # C: (batch, 1, dstate, seqlen) FP32
            D=D_fp32,        # D: (dim,) FP32 or None
            z=z_6_2_fp32,    # z: (batch, dim, seqlen) FP32
            delta_bias=dt_bias_fp32,
            delta_softplus=True,
            return_last_state=False
        )

        # === Step 6: had + out_proj ===
        # selective_scan_fn 返回 (batch, dim, seqlen), 需要转换为 (batch, seqlen, dim)
        y_6_2 = rearrange(y_6_2, "b d l -> b l d")
        y_6_2_fp16 = y_6_2.half() if y_6_2.dtype != torch.float16 else y_6_2
        y_6_2 = self.had(y_6_2_fp16)
        out_6_2 = self.out_proj(y_6_2)

        return out_6_2

    def forward_mode6_3_eval(self, hidden_states, inference_params=None):
        """
        Mode 6-3: HalfScale 2x Precision for Small Values
        - Conv1D: INT8 输入 → FP32 输出
        - HalfScale Virtual Quant: 小值 (|q| < 64) 用 half scale (2x 精度), 大值用正常 scale
        - x_proj & SSM: 用相同的 HalfScale 值
        关键: 小值使用 scale/2，量化步长减半，精度翻倍
        """
        import quant_causal_conv1d_cuda
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x_6_3, z_6_3 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_6_3_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_6_3, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )

        # === Step 3: HalfScale Virtual Quantization ===
        scale = self.conv1d.output_scale
        half_scale = scale / 2.0

        # 用正常 scale 判断值的大小
        x_quantized = torch.round(x_6_3_fp32 / scale)

        # 判断小值 vs 大值 vs outlier
        is_small = (x_quantized.abs() < 64)      # 小值: 用 half scale (2x 精度)
        # is_outlier = (x_quantized.abs() > 127)   # outlier: 用正常 scale VQ (clamp)
        # is_normal = ~is_small & ~is_outlier    # 大值: 用正常 scale

        # 小值用 half scale (更精确)
        x_small_quantized = torch.round(x_6_3_fp32 / half_scale).clamp(-127, 127)
        x_small = x_small_quantized * half_scale

        # 大值和 outlier 用正常 scale
        x_normal_vq = x_quantized.clamp(-128, 127) * scale

        # 组合: 小值用 half scale, 其余用正常 scale
        x_6_3_mixed = torch.where(is_small, x_small, x_normal_vq)

        # === Step 4: x_proj (FP32) - 用与 SSM 相同的 HalfScale 值！===
        x_6_3_reshape = rearrange(x_6_3_mixed, "b d l -> (b l) d")
        x_dbl_6_3 = self.x_proj.forward_mode6(x_6_3_reshape)  # FP32 output
        x_dbl_6_3 = x_dbl_6_3.view(batch, seqlen, -1)
        dt_6_3_raw, B_6_3_raw, C_6_3_raw = torch.split(x_dbl_6_3, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt_proj: FP32 输出
        dt_6_3 = self.dt_proj.to_seqlen_last_mode6(dt_6_3_raw.contiguous())

        # B, C: reshape for selective_scan_fn
        B_6_3 = rearrange(B_6_3_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_6_3 = rearrange(C_6_3_raw, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # === Step 5: SSM (FP32) - 使用 mamba_ssm 原始 FP32 kernel ===
        A_fp32 = -torch.exp(self.selective_scan.A_log.float() * self.selective_scan.A_scale)

        z_6_3_fp32 = z_6_3.float() * self.selective_scan.z_scale

        D_fp32 = None
        if self.selective_scan.D is not None:
            D_fp32 = self.selective_scan.D.float() * self.selective_scan.D_scale

        dt_bias_fp32 = None
        if self.selective_scan.dt_bias is not None:
            dt_bias_fp32 = self.selective_scan.dt_bias.float() * self.selective_scan.dt_bias_scale

        y_6_3 = selective_scan_fn(
            x_6_3_mixed,     # u: (batch, dim, seqlen) FP32 HalfScale mixed
            dt_6_3,          # delta: (batch, dim, seqlen) FP32
            A_fp32,          # A: (dim, dstate) FP32
            B_6_3,           # B: (batch, 1, dstate, seqlen) FP32
            C_6_3,           # C: (batch, 1, dstate, seqlen) FP32
            D=D_fp32,
            z=z_6_3_fp32,
            delta_bias=dt_bias_fp32,
            delta_softplus=True,
            return_last_state=False
        )

        # === Step 6: had + out_proj ===
        y_6_3 = rearrange(y_6_3, "b d l -> b l d")
        y_6_3_fp16 = y_6_3.half() if y_6_3.dtype != torch.float16 else y_6_3
        y_6_3 = self.had(y_6_3_fp16)
        out_6_3 = self.out_proj(y_6_3)

        return out_6_3

    def _get_pa1_output_scale(self, layer_idx):
        """
        获取当前层对应的 α=1.0 模型的 output_scale
        使用模块级变量缓存，只加载一次
        """
        global _PA1_SCALES_CACHE, _PA1_MODEL_PATHS

        # 根据模型 d_model 判断模型大小
        if self.d_model == 768:
            model_size = '130m'
        elif self.d_model == 2048:
            model_size = '1.4b'
        elif self.d_model == 2560:
            model_size = '2.8b'
        else:
            raise ValueError(f"Unknown model size for d_model={self.d_model}")

        # 检查缓存
        if model_size not in _PA1_SCALES_CACHE:
            # 加载 α=1.0 模型
            pa1_path = _PA1_MODEL_PATHS[model_size]
            model_file = _os.path.join(pa1_path, 'pytorch_model.bin')

            if not _os.path.exists(model_file):
                print(f"[Mode 6-4] Warning: α=1.0 model not found at {model_file}, using current scale")
                return self.conv1d.output_scale

            pa1_state = torch.load(model_file, map_location='cpu')

            # 提取所有层的 output_scale
            _PA1_SCALES_CACHE[model_size] = {}
            for key, value in pa1_state.items():
                if 'conv1d.output_scale' in key:
                    # key 格式: backbone.layers.{idx}.mixer.conv1d.output_scale
                    parts = key.split('.')
                    idx = int(parts[2])  # 提取 layer index
                    # Handle both tensor and scalar values
                    if hasattr(value, 'item'):
                        _PA1_SCALES_CACHE[model_size][idx] = value.item()
                    else:
                        _PA1_SCALES_CACHE[model_size][idx] = float(value)

            print(f"[Mode 6-4] Loaded α=1.0 scales for {model_size} from {pa1_path}")

        return _PA1_SCALES_CACHE[model_size].get(layer_idx, self.conv1d.output_scale)

    def forward_mode6_4_eval(self, hidden_states, inference_params=None):
        """
        Mode 6-4: Calibrated DualScale INT8 虚拟量化 + x_proj 一致
        - Conv1D: INT8 输入 → FP32 输出
        - DualScale Virtual Quant:
            - 正常值用当前校准 scale (α=0.9995/0.9999)
            - Outlier 用预校准 scale (α=1.0)
        - Outlier 判定: 用当前 scale
        - x_proj & SSM: 用相同的 DualScale 值！

        与 Mode 6-3 的区别:
        - 6-3: outlier_scale = max(outlier)/127 (动态计算)
        - 6-4: outlier_scale = α=1.0 模型的 output_scale (预校准)
        """
        import quant_causal_conv1d_cuda
        import quant_sscan_cuda

        batch, seqlen, dim = hidden_states.shape

        # === Step 1: in_proj (INT8) ===
        xz = self.in_proj.to_seqlen_last(hidden_states)
        x_6_4, z_6_4 = xz.chunk(2, dim=1)

        # === Step 2: Conv1D (INT8 → FP32) ===
        x_6_4_fp32 = quant_causal_conv1d_cuda.fwd_mode5(
            x_6_4, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )

        # === Step 3: Calibrated DualScale Virtual Quantization ===
        output_scale = self.conv1d.output_scale

        # 获取 α=1.0 预校准的 outlier scale
        outlier_scale = self._get_pa1_output_scale(self.layer_idx)

        # Step 3.1: 用当前 scale 判定 outlier
        x_quantized = torch.round(x_6_4_fp32 / output_scale)
        is_outlier = (x_quantized.abs() > 127)

        # Step 3.2: 正常值用当前 scale
        x_normal = x_quantized.clamp(-128, 127) * output_scale

        # Step 3.3: Outlier 用预校准的 α=1.0 scale
        outlier_values = x_6_4_fp32[is_outlier]
        if outlier_values.numel() > 0:
            x_outlier_quantized = torch.round(outlier_values / outlier_scale)
            x_outlier = x_outlier_quantized.clamp(-128, 127) * outlier_scale
            x_6_4_mixed = x_normal.clone()
            x_6_4_mixed[is_outlier] = x_outlier
        else:
            x_6_4_mixed = x_normal

        # === Step 4: x_proj (FP32) - 用与 SSM 相同的 DualScale 值！===
        x_6_4_reshape = rearrange(x_6_4_mixed, "b d l -> (b l) d")
        x_dbl_6_4 = self.x_proj.forward_mode6(x_6_4_reshape)  # FP32 output
        x_dbl_6_4 = x_dbl_6_4.view(batch, seqlen, -1)
        dt_6_4_fp32, B_6_4_fp32, C_6_4_fp32 = torch.split(x_dbl_6_4, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt_proj: FP32 -> FP32 (via forward_mode6), then requantize to INT8 for SSM kernel
        dt_6_4_proj = self.dt_proj.to_seqlen_last_mode6(dt_6_4_fp32.contiguous())  # (B, D, L) FP32
        # Requantize dt to INT8 for SSM kernel - need contiguous for CUDA kernel
        dt_6_4 = torch.round(dt_6_4_proj / self.selective_scan.dt_scale).clamp(-128, 127).to(torch.int8).contiguous()

        # Requantize B, C to INT8 for SSM kernel
        B_6_4_rearranged = rearrange(B_6_4_fp32, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        B_6_4 = torch.round(B_6_4_rearranged / self.selective_scan.B_scale).clamp(-128, 127).to(torch.int8).contiguous()

        C_6_4_rearranged = rearrange(C_6_4_fp32, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_6_4 = torch.round(C_6_4_rearranged / self.selective_scan.C_scale).clamp(-128, 127).to(torch.int8).contiguous()

        # === Step 5: SSM (FP32 输入) - 用与 x_proj 相同的 DualScale 值！===
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_6_4, _ = quant_sscan_cuda.fwd_mode5(
            x_6_4_mixed,  # 与 x_proj 相同的 DualScale 值
            dt_6_4, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_6_4, ensure_shape_1(self.selective_scan.B_scale),
            C_6_4, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z_6_4, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
            True
        )

        # === Step 6: had + out_proj ===
        y_6_4_fp16 = y_6_4.half() if y_6_4.dtype != torch.float16 else y_6_4
        y_6_4 = self.had(y_6_4_fp16)
        out_6_4 = self.out_proj(y_6_4)

        return out_6_4

    def forward_mode5(self, hidden_states, inference_params=None):
        """
        Mode 5: Dual-path forward (旧实现，保留兼容性)
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
        B_mode5 = rearrange(B_mode5, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_mode5 = rearrange(C_mode5, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # Call Mode 5 SSM kernel (FP32 input)
        # Helper to ensure scale tensors have shape (1,) instead of scalar
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_mode5, _ = quant_sscan_cuda.fwd_mode5(
            x_mode5_fp32,  # FP32 input
            dt_mode5, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_mode5, ensure_shape_1(self.selective_scan.B_scale),
            C_mode5, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
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
            print(f"[Mode 5 Dual-Path] Layer {self.layer_idx}")
            print(f"{'='*80}")

            # === Input Statistics ===
            hs_flat = hidden_states.flatten().float()
            print(f"INPUT (hidden_states):")
            print(f"  Shape: {hidden_states.shape}, Dtype: {hidden_states.dtype}")
            print(f"  Mean: {hs_flat.mean().item():.6f}, Std: {hs_flat.std().item():.6f}")
            print(f"  Min: {hs_flat.min().item():.6f}, Max: {hs_flat.max().item():.6f}")

            # === Scale Values ===
            def get_scale_val(s):
                return s.item() if hasattr(s, 'item') else float(s)
            print(f"\nSCALE VALUES:")
            print(f"  Conv1D: input={get_scale_val(self.conv1d.input_scale):.8f}, output={get_scale_val(self.conv1d.output_scale):.8f}")
            print(f"  Conv1D: weight={get_scale_val(self.conv1d.weight_scale):.8f}, bias={get_scale_val(self.conv1d.bias_scale):.8f}")
            print(f"  SSM: dt={get_scale_val(self.selective_scan.dt_scale):.8f}, A={get_scale_val(self.selective_scan.A_scale):.8f}")
            print(f"  SSM: B={get_scale_val(self.selective_scan.B_scale):.8f}, C={get_scale_val(self.selective_scan.C_scale):.8f}")
            print(f"  SSM: D={get_scale_val(self.selective_scan.D_scale):.8f}, z={get_scale_val(self.selective_scan.z_scale):.8f}")

            # === Output Comparison ===
            print(f"\nOUTPUT COMPARISON:")
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

        # === Step 2: Mode 6 Conv1D (INT8 → FP32) - 共享起点 ===
        x_mode6_fp32 = quant_causal_conv1d_cuda.fwd_mode6(
            x_int8, self.conv1d.input_scale,
            self.conv1d.weight, self.conv1d.weight_scale,
            self.conv1d.bias_scale, self.conv1d.bias, True
        )  # FP32 - 这是两条路径的共享起点

        # === Step 3: 从共享的 x_mode6_fp32 分叉 ===
        # Mode 0: 量化 FP32 Conv 输出为 INT8，直接用于 x_proj 和 SSM (不再做第二次 Conv1D!)
        x_mode0_int8 = torch.round(x_mode6_fp32 / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)

        # === Step 4a: Mode 0 SSM (用量化后的 x_mode0_int8) ===
        x_mode0_reshape = rearrange(x_mode0_int8, "b d l -> b l d").contiguous()
        x_dbl_mode0 = self.x_proj(x_mode0_reshape)
        dt_mode0, B_mode0, C_mode0 = torch.split(x_dbl_mode0, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_mode0 = self.dt_proj.to_seqlen_last(dt_mode0.contiguous())
        B_mode0 = rearrange(B_mode0, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C_mode0 = rearrange(C_mode0, "b l dstate -> b dstate l", l=seqlen).contiguous()

        y_mode0 = self.selective_scan.forward(x_mode0_int8, dt_mode0, B_mode0, C_mode0, z=z, return_last_state=False)

        # === Step 4b: Mode 6 SSM (FP32 input) ===
        # Requantize FP32 to INT8 for x_proj
        x_mode6_int8_for_xproj = torch.round(x_mode6_fp32 / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
        x_mode6_reshape = rearrange(x_mode6_int8_for_xproj, "b d l -> b l d").contiguous()
        x_dbl_mode6 = self.x_proj(x_mode6_reshape)
        dt_mode6, B_mode6, C_mode6 = torch.split(x_dbl_mode6, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_mode6 = self.dt_proj.to_seqlen_last(dt_mode6.contiguous())
        B_mode6 = rearrange(B_mode6, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()
        C_mode6 = rearrange(C_mode6, "b l dstate -> b 1 dstate l", l=seqlen).contiguous()

        # Call Mode 6 SSM kernel (FP32 input)
        # Helper to ensure scale tensors have shape (1,) instead of scalar
        def ensure_shape_1(t):
            return t.view(1) if t.dim() == 0 else t

        y_mode6, _ = quant_sscan_cuda.fwd_mode6(
            x_mode6_fp32,  # FP32 input
            dt_mode6, ensure_shape_1(self.selective_scan.dt_scale),
            self.selective_scan.A_log, ensure_shape_1(self.selective_scan.A_scale),
            B_mode6, ensure_shape_1(self.selective_scan.B_scale),
            C_mode6, ensure_shape_1(self.selective_scan.C_scale),
            ensure_shape_1(self.selective_scan.ssm_state_scale),
            self.selective_scan.D, ensure_shape_1(self.selective_scan.D_scale),
            z, ensure_shape_1(self.selective_scan.z_scale),
            self.selective_scan.dt_bias, ensure_shape_1(self.selective_scan.dt_bias_scale),
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
            print(f"[Mode 6 Dual-Path] Layer {self.layer_idx}")
            print(f"{'='*80}")

            # === Input Statistics ===
            hs_flat = hidden_states.flatten().float()
            print(f"INPUT (hidden_states):")
            print(f"  Shape: {hidden_states.shape}, Dtype: {hidden_states.dtype}")
            print(f"  Mean: {hs_flat.mean().item():.6f}, Std: {hs_flat.std().item():.6f}")
            print(f"  Min: {hs_flat.min().item():.6f}, Max: {hs_flat.max().item():.6f}")

            # === Shared Starting Point (x_mode6_fp32) ===
            fp32_flat = x_mode6_fp32.flatten()
            print(f"\nSHARED STARTING POINT (x_mode6_fp32 - Conv1D FP32 output):")
            print(f"  Shape: {x_mode6_fp32.shape}, Dtype: {x_mode6_fp32.dtype}")
            print(f"  Mean: {fp32_flat.mean().item():.6f}, Std: {fp32_flat.std().item():.6f}")
            print(f"  Min: {fp32_flat.min().item():.6f}, Max: {fp32_flat.max().item():.6f}")

            # === Scale Values ===
            def get_scale_val(s):
                return s.item() if hasattr(s, 'item') else float(s)
            print(f"\nSCALE VALUES:")
            print(f"  Conv1D: input={get_scale_val(self.conv1d.input_scale):.8f}, output={get_scale_val(self.conv1d.output_scale):.8f}")
            print(f"  Conv1D: weight={get_scale_val(self.conv1d.weight_scale):.8f}, bias={get_scale_val(self.conv1d.bias_scale):.8f}")
            print(f"  SSM: dt={get_scale_val(self.selective_scan.dt_scale):.8f}, A={get_scale_val(self.selective_scan.A_scale):.8f}")
            print(f"  SSM: B={get_scale_val(self.selective_scan.B_scale):.8f}, C={get_scale_val(self.selective_scan.C_scale):.8f}")
            print(f"  SSM: D={get_scale_val(self.selective_scan.D_scale):.8f}, z={get_scale_val(self.selective_scan.z_scale):.8f}")

            # === Quantization Comparison ===
            x_mode0_flat = x_mode0_int8.flatten().float()
            x_mode6_int8_flat = x_mode6_int8_for_xproj.flatten().float()
            quant_diff = (x_mode0_flat - x_mode6_int8_flat).abs()
            print(f"\nQUANTIZATION CHECK (x_mode0_int8 vs x_mode6_int8_for_xproj):")
            print(f"  Should be identical since both come from same FP32: {quant_diff.max().item() == 0}")
            print(f"  Max diff: {quant_diff.max().item():.6f}")

            # === Output Comparison ===
            print(f"\nOUTPUT COMPARISON:")
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

