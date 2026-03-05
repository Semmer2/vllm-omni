# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Any, Optional

from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

_impl_class: type | None = None


def _get_impl_class() -> type:
    global _impl_class
    if _impl_class is not None:
        return _impl_class
    if current_omni_platform.is_npu():
        _impl_class = _get_npu_impl_class()
    elif current_omni_platform.is_cuda():
        _impl_class = _get_cuda_impl_class()
    else:
        raise NotImplementedError(
            f"HunyuanFusedMoE is not implemented for current_omni_platform: "
            f"{current_omni_platform!r}"
        )
    return _impl_class


def _get_cuda_impl_class() -> type:
    from vllm.model_executor.layers.fused_moe import SharedFusedMoE

    class HunyuanFusedMoECuda(SharedFusedMoE):
        def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
            super().__init__(prefix=prefix, **kwargs)
            self._prefix = prefix
            self._init_hook_handle = self.register_forward_pre_hook(
                self._initialize_kernel_hook, with_kwargs=True
            )

        def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
            if self.quant_method:
                self.quant_method.process_weights_after_loading(self)
            self._init_hook_handle.remove()

        def forward(self, hidden_states: Any, router_logits: Any) -> Any:
            return super().forward(hidden_states, router_logits)

    return HunyuanFusedMoECuda


def _get_npu_impl_class() -> type:
    import torch
    import vllm.forward_context as _vllm_fc
    from vllm_ascend.ops.fused_moe.fused_moe import AscendSharedFusedMoE
    from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
    from vllm.config import VllmConfig
    from vllm.distributed import get_ep_group
    from vllm_omni.diffusion.forward_context import get_forward_context as omni_get_ctx


    if hasattr(torch, "npu") and torch.npu.is_available():
        if not hasattr(_vllm_fc.ForwardContext, "moe_comm_method"):
            _vllm_fc.ForwardContext.__annotations__["in_profile_run"] = bool
            _vllm_fc.ForwardContext.in_profile_run = False

    def select_moe_comm_method(vllm_config: VllmConfig) -> MoECommType | None:
        soc_version = get_ascend_device_type()
        if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
            moe_comm_type = MoECommType.ALLGATHER
        elif soc_version in {AscendDeviceType.A2}:
            moe_comm_type = MoECommType.ALLGATHER
        elif soc_version in {AscendDeviceType.A3}:
            moe_comm_type =  MoECommType.ALLTOALL
        elif soc_version in {AscendDeviceType._310P}:
            moe_comm_type = MoECommType.ALLGATHER
        elif soc_version in {AscendDeviceType.A5}:
                moe_comm_type = MoECommType.ALLTOALL
        else:
            raise ValueError(f"Unsupported soc_version: {soc_version}")
        return moe_comm_type

    class HunyuanFusedMoENPU(AscendSharedFusedMoE):
        def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
            super().__init__(prefix=prefix, **kwargs)
            self._prefix = prefix
            self._init_hook_handle = self.register_forward_pre_hook(
                self._initialize_kernel_hook, with_kwargs=True
            )

            _vllm_fc.ForwardContext.moe_comm_type = select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
            _vllm_fc.ForwardContext.moe_comm_method=_MoECommMethods.get(_vllm_fc.ForwardContext.moe_comm_type)
            _vllm_fc.ForwardContext.flash_comm_v1_enabled=False


        def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
            if self.quant_method:
                self.quant_method.process_weights_after_loading(self)
            self._init_hook_handle.remove()

        def forward(self, hidden_states: Any, router_logits: Any) -> Any:
            return super().forward(hidden_states, router_logits)

    return HunyuanFusedMoENPU


class HunyuanFusedMoE:
    def __new__(cls, *, prefix: str = "", **kwargs: Any) -> Any:
        impl = _get_impl_class()
        return impl(prefix=prefix, **kwargs)

    @classmethod
    def make_expert_params_mapping(
        cls,
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        return _get_impl_class().make_expert_params_mapping(
            model,
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )