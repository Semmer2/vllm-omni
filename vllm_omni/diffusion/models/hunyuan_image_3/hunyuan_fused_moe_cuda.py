# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from vllm.model_executor.layers.fused_moe import SharedFusedMoE

logger = logging.getLogger(__name__)

class HunyuanFusedMoE(SharedFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs):
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix

        self._init_hook_handle = self.register_forward_pre_hook(self._initialize_kernel_hook, with_kwargs=True)

    def _initialize_kernel_hook(self, module, args, kwargs):
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(self, hidden_states, router_logits):
        return super().forward(hidden_states, router_logits)