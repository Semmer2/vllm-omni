# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional, Any
import torch
import torch_npu

from vllm_ascend.ops.fused_moe.fused_moe import AscendSharedFusedMoE
import vllm.forward_context as _vllm_fc

if hasattr(torch, 'npu') and torch.npu.is_available():
    if not hasattr(_vllm_fc.ForwardContext, 'moe_comm_method'):
        # modify __annotations__
        _vllm_fc.ForwardContext.__annotations__['moe_comm_method'] = Optional[Any]
        _vllm_fc.ForwardContext.__annotations__['moe_comm_type'] = Optional[Any]
        _vllm_fc.ForwardContext.__annotations__['sp_enabled'] = bool
        _vllm_fc.ForwardContext.__annotations__['in_profile_run'] = bool
        # default
        _vllm_fc.ForwardContext.moe_comm_method = None
        _vllm_fc.ForwardContext.moe_comm_type = None
        _vllm_fc.ForwardContext.sp_enabled = False
        _vllm_fc.ForwardContext.in_profile_run = False

logger = logging.getLogger(__name__)


class HunyuanFusedMoE(AscendSharedFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs):
        from vllm_omni.diffusion.forward_context import get_forward_context as omni_get_ctx
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods, MoECommType
        #omni_ctx = omni_get_ctx()
        #vllm_config = omni_ctx.vllm_config
        #init_ascend_config(vllm_config)
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix
        self._init_hook_handle = self.register_forward_pre_hook(self._initialize_kernel_hook, with_kwargs=True)
        from vllm_ascend.ascend_config import get_ascend_config
        ascend_config = get_ascend_config()
        if getattr(ascend_config, 'moe_comm_type', None) == "mc2":
            self._moe_comm_type = MoECommType.MC2
        else:
            self._moe_comm_type = MoECommType.ALLTOALL
        self._moe_comm_method = _MoECommMethods.get(self._moe_comm_type)


    def _initialize_kernel_hook(self, module, args, kwargs):
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(self, hidden_states, router_logits):
        from vllm.model_executor.layers.fused_moe.layer import get_forward_context

        ctx = get_forward_context()
        if not ctx.remaining_moe_layers:
            import re

            moe_names = [name for name in ctx.no_compile_layers.keys() if ".mlp.experts" in name]

            def get_layer_num(name):
                match = re.search(r"layers\.(\d+)\.mlp", name)
                return int(match.group(1)) if match else -1

            moe_names.sort(key=get_layer_num, reverse=True)
            ctx.remaining_moe_layers.extend(moe_names)
        return super().forward(hidden_states, router_logits)
