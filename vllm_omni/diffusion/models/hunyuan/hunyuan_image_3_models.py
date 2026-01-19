import logging
from dataclasses import dataclass
from typing import Callable, Dict
import inspect
from einops import rearrange
from typing import Optional, Tuple, Any, List, Union, Iterable, cast
import numpy as np
from PIL import Image
import regex as re
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# from vllm.attention.layer import Attention
from vllm_omni.diffusion.attention.layer import Attention
from vllm.attention.backends.abstract import AttentionType
from vllm.config import CacheConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput
from transformers.activations import ACT2FN
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .hunyuan_image3_utils import (
    HunYuanImageAttentionMeta,
    HunYuanRotary2DEmbedder,
    ImageKVCacheManager,
)
# from vllm_omni.diffusion.data import OmniDiffusionConfig

try:
    import flashinfer
except Exception as e:
    flashinfer = None

logger = logging.getLogger(__name__)

def topkgating(
        logits: Tensor,
        topk: int,
        group_limited_greedy: bool = False,
        n_group: int | None = None,
        topk_group: int | None = None,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        capacity_factor: float = 1.0,
        drop_tokens: bool = False,
):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)

    if group_limited_greedy:
        group_shape = list(gates.shape[:-1]) + [n_group, gates.shape[-1] // n_group]
        group_scores = (
            gates.reshape(group_shape).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(
            group_scores, topk_group, dim=-1, sorted=False
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                group_shape
            )
            .reshape(list(gates.shape))
        )  # [n, e]
        gates = gates.masked_fill(~score_mask.bool(), 0.0)

    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts ** 2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)

    if drop_tokens:
        expert_capacity = int(max(topk, topk * gates.shape[0] // gates.shape[1]) * capacity_factor)
    else:
        expert_index_flat = expert_index.flatten()
        tokens_per_expert = torch.bincount(expert_index_flat, minlength=num_experts)
        expert_capacity = torch.max(tokens_per_expert).item()

    if norm_topk_prob and topk > 1:
        gates_s = torch.clamp(
            torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1), min=torch.finfo(gates.dtype).eps
        )
        router_probs = gates / gates_s
    else:
        router_probs = gates * routed_scaling_factor
    # Make num_selected_experts the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3 choices,
    # etc.
    expert_index = torch.transpose(expert_index, 0, 1)
    # Shape: [num_selected_experts * tokens_per_group]
    expert_index = expert_index.reshape(-1)

    # Create mask out of indices.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    # Shape: [num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape((topk, -1, num_experts))
    # Shape: [tokens_per_group, num_selected_experts, num_experts].
    token_priority = torch.transpose(token_priority, 0, 1)
    # For each token, across all selected experts, select the only non-negative
    # (unmasked) priority. Now, for group G routing to expert E, token T has
    # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
    # is its targeted expert.
    # Shape: [tokens_per_group, num_experts].
    token_priority = torch.max(token_priority, dim=1)[0]

    # Token T can only be routed to expert E if its priority is positive and
    # less than the expert capacity. One-hot matrix will ignore indices outside
    # the range [0, expert_capacity).
    # Shape: [tokens_per_group, num_experts, expert_capacity].
    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
    # expert_capacity].
    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


@dataclass
class CausalMMOutputWithPast(CausalLMOutputWithPast):
    diffusion_prediction: Optional[torch.Tensor] = None


class HunyuanImage3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HunyuanImage3Model`]. It is used to instantiate
    an Hunyuan model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Hunyuan-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Hunyuan Image 3 model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`HunyuanImage3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations or shared MLP representations.
        moe_intermediate_size (`int` or `List`, *optional*, defaults to 11008):
            Dimension of the MLP representations in MoE. Use a list if you want a different size per layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether query and key in attention use norm
        use_cla (`bool`, *optional*, defaults to `False`):
            Whether to use CLA in attention
        cla_share_factor (`int`, *optional*, defaults to 1):
            The share factor of CLA
        num_experts (`int` or `List`, *optional*, defaults to 1):
            The number of experts for moe. If it is a list, it will be used as the number of experts for each layer.
        num_shared_expert (`int` or `List`, *optional*, defaults to 1):
            The number of shared experts for moe. If it is a list, it will be used as the number of shared experts
            for each layer.
        moe_topk (`int` or `List`, *optional*, defaults to 1):
            The topk value for moe. If it is a list, it will be used as the topk value for each layer.
        capacity_factor (Not used) (`float` or `List`, *optional*, defaults to 1.0):
            The capacity factor for moe. If it is a list, it will be used as the capacity factor for each layer.
        moe_layer_num_skipped (`int`, *optional*, defaults to 0):
            First moe_layer_num_skipped layers do not use MoE.
    """

    model_type = "Hunyuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=290943,
            hidden_size=4096,
            intermediate_size: int=11008,
            moe_intermediate_size: Union[int, List]=None,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            attention_head_dim=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            eod_token_id=3,
            im_start_id=4,
            im_end_id=5,
            text_start_id=6,
            text_end_id=7,
            image_token_id=8,
            video_start_id=9,
            video_end_id=10,
            im_newline_id=11,
            mask_init_id=12,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            use_qk_norm=False,
            use_rotary_pos_emb=True,
            use_cla=False,
            cla_share_factor=1,
            norm_type="hf_rms",
            num_experts: Union[int, List] = 1,
            use_mixed_mlp_moe=False,
            num_shared_expert: Union[int, List] = 1,
            moe_topk: Union[int, List] = 1,
            capacity_factor: int = 1.0,
            moe_drop_tokens=False,
            moe_random_routing_dropped_token=False,
            use_mla=False,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            v_head_dim=128,
            qk_nope_head_dim=128,
            moe_layer_num_skipped=0,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            group_limited_greedy=False,
            n_group=None,
            topk_group=None,
            add_classification_head=False,
            class_num=0,
            pool_type="last",
            pad_id=-1,
            # Added
            moe_impl="eager",
            vae_downsample_factor=(16, 16),     # (h, w)
            img_proj_type="unet",
            patch_size=1,
            patch_embed_hidden_dim=1024,
            image_base_size=1024,
            vae=None,
            vit=None,
            vit_processor=None,
            vit_aligner=None,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.moe_impl = moe_impl
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        self.capacity_factor = capacity_factor
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_random_routing_dropped_token = moe_random_routing_dropped_token

        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        else:
            self.attention_head_dim = self.hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.use_cla = use_cla
        self.cla_share_factor = cla_share_factor
        self.norm_type = norm_type
        # MLA args
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # DeepSeek related args
        self.moe_layer_num_skipped = moe_layer_num_skipped
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.group_limited_greedy = group_limited_greedy
        self.n_group = n_group
        self.topk_group = topk_group
        self.add_classification_head = add_classification_head
        self.class_num = class_num
        self.pool_type = pool_type
        self.pad_id = pad_id

        if self.class_num is not None:
            self.dense_list = [self.hidden_size, self.class_num]

        # ViT args
        self.vit = vit
        self.vit_processor = vit_processor
        self.vit_aligner = vit_aligner

        # Image Gen args
        self.vae = vae
        self.vae_downsample_factor = vae_downsample_factor
        self.img_proj_type = img_proj_type
        self.patch_size = patch_size
        self.patch_embed_hidden_dim = patch_embed_hidden_dim
        self.image_base_size = image_base_size

        # token id
        self.eod_token_id = eod_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.text_start_id = text_start_id
        self.text_end_id = text_end_id
        self.image_token_id = image_token_id
        self.video_start_id = video_start_id
        self.video_end_id = video_end_id
        self.im_newline_id = im_newline_id
        self.mask_init_id = mask_init_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class HunyuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HunyuanRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HunyuanImage3SDPAAttention(nn.Module):
    """PyTorch SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention"""

    def __init__(self, config: HunyuanImage3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = 'self'

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.use_qk_norm = config.use_qk_norm
        self.use_rotary_pos_emb = config.use_rotary_pos_emb
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        # define layers
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size_q, self.hidden_size, bias=config.attention_bias)

        if self.use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.use_rotary_pos_emb:
            self._init_rope()

    def _init_rope(self):
        scaling_type = self.config.rope_scaling["type"]
        if scaling_type == "custom":
            # Using custom rotary embedding
            self.rotary_emb = None
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if output_attentions:
            raise NotImplementedError(
                'HunyuanImage3Model is using HunyuanImage3SDPAAttention,'
                'but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`.'
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                        self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if cos.squeeze().dim() == 2:
                query_states = rotary_position_embedding(query_states, cos.squeeze(), sin.squeeze(), head_first=True)
                key_states   = rotary_position_embedding(key_states, cos.squeeze(), sin.squeeze(), head_first=True)
            else:
                query_states = rotary_position_embedding(query_states, cos.squeeze()[0], sin.squeeze()[0], head_first=True)
                key_states   = rotary_position_embedding(key_states, cos.squeeze()[0], sin.squeeze()[0], head_first=True)



        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            query_states = query_states.to(key_states.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class HunyuanImage3FlashAttention2(HunyuanImage3SDPAAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                        self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if cos.squeeze().dim() == 2:
                query_states = rotary_position_embedding(query_states, cos.squeeze(), sin.squeeze(), head_first=True)
                key_states   = rotary_position_embedding(key_states, cos.squeeze(), sin.squeeze(), head_first=True)
            else:
                query_states = rotary_position_embedding(query_states, cos.squeeze()[0], sin.squeeze()[0], head_first=True)
                key_states   = rotary_position_embedding(key_states, cos.squeeze()[0], sin.squeeze()[0], head_first=True)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        target_dtype = key_states.dtype if key_states.dtype in [torch.bfloat16, torch.float16] else torch.bfloat16

        q_fa = query_states.to(target_dtype).transpose(1, 2).contiguous()
        k_fa = key_states.to(target_dtype).transpose(1, 2).contiguous()
        v_fa = value_states.to(target_dtype).transpose(1, 2).contiguous()

        mode = kwargs.get("mode", "gen_text")
        # For gen_text and gen_image, we need to handle the attention differently
        if mode == "gen_text":
            if attention_mask is None:
                attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=False)   # decode attention
            else:
                attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=True)    # prefill attention
        else:  # image attention
            gen_timestep_scatter_index: Optional[torch.Tensor] = kwargs.get("gen_timestep_scatter_index", None)
            assert gen_timestep_scatter_index is not None, \
                "When gen_image, `gen_timestep_scatter_index` must be provided."
            # TODO: batchify
            timestep_index = gen_timestep_scatter_index[0, 0].item()
            # When image generation, different attention implementations for the first step and the following steps
            # help to improve the inference speed.
            first_step = kwargs.get("first_step", None)
            if first_step is None:
                raise ValueError("When gen_image, `first_step` must be provided.")
            if first_step:
                casual_len = timestep_index + 1
                text_query_states = q_fa[:, :casual_len, :, :]
                text_key_states = k_fa[:, :casual_len, :, :]
                text_value_states = v_fa[:, :casual_len, :, :]
                text_attn_output = flash_attn_func(
                    text_query_states, text_key_states, text_value_states, causal=True)
                image_query_states = q_fa[:, casual_len:, :, :]
                image_attn_output = flash_attn_func(image_query_states, k_fa, v_fa, causal=False)
                attn_output = torch.cat((text_attn_output, image_attn_output), dim=1)
            else:
                casual_len = timestep_index + 1
                timestep_query_states = q_fa[:, 0:1, :, :]
                timestep_key_states = k_fa[:, :casual_len, :, :]
                timestep_value_states = v_fa[:, :casual_len, :, :]
                timestep_attn_output = flash_attn_func(
                    timestep_query_states, timestep_key_states, timestep_value_states, causal=True)
                image_query_states = q_fa[:, 1:, :, :]
                image_attn_output = flash_attn_func(image_query_states, k_fa, v_fa, causal=False)
                attn_output = torch.cat((timestep_attn_output, image_attn_output), dim=1)

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


Hunyuan_ATTENTION_CLASSES = {
    "eager": HunyuanImage3SDPAAttention,
    "sdpa": HunyuanImage3SDPAAttention,
    "flash_attention_2": HunyuanImage3FlashAttention2,
}


class HunyuanMLP(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx=None, is_shared_mlp=False, is_moe=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act

        self.intermediate_size = config.intermediate_size
        if is_shared_mlp or is_moe:
            # 如果是 moe 的话，优先用 moe_intermediate_size
            if config.moe_intermediate_size is not None:
                self.intermediate_size = config.moe_intermediate_size \
                    if isinstance(config.moe_intermediate_size, int) else config.moe_intermediate_size[layer_idx]

            if is_shared_mlp:
                num_shared_expert = config.num_shared_expert \
                    if isinstance(config.num_shared_expert, int) else config.num_shared_expert[layer_idx]
                self.intermediate_size *= num_shared_expert

        self.act_fn = ACT2FN[config.hidden_act]
        if self.hidden_act == "silu":
            self.intermediate_size *= 2  # SwiGLU
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=config.mlp_bias)
        elif self.hidden_act == "gelu":
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        else:
            assert False, "other hidden_act are not supported"

    def forward(self, x):
        if self.hidden_act == "silu":
            gate_and_up_proj = self.gate_and_up_proj(x)
            x1, x2 = gate_and_up_proj.chunk(2, dim=2)
            down_proj = self.down_proj(x1 * self.act_fn(x2))
            return down_proj
        elif self.hidden_act == "gelu":
            intermediate = self.gate_and_up_proj(x)
            intermediate = self.act_fn(intermediate)
            output = self.down_proj(intermediate)
            return output
        else:
            assert False, "other hidden_act are not supported"


class HunyuanTopKGate(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[layer_idx]
        self.drop_tokens = config.moe_drop_tokens
        self.min_capacity = 8
        self.random_routing_dropped_token = config.moe_random_routing_dropped_token
        num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        self.wg = nn.Linear(config.hidden_size, num_experts, bias=False, dtype=torch.float32)

        # DeepSeek gating args
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.group_limited_greedy = config.group_limited_greedy

    def forward(self, hidden_states, topk_impl='default'):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        if topk_impl == 'default':
            gate_output = topkgating(logits, self.moe_topk, group_limited_greedy=self.group_limited_greedy,
                                     n_group=self.n_group, topk_group=self.topk_group,
                                     norm_topk_prob=self.norm_topk_prob,
                                     routed_scaling_factor=self.routed_scaling_factor,
                                     capacity_factor=self.config.capacity_factor,
                                     drop_tokens=self.drop_tokens)
        elif topk_impl == 'easy':
            gate_output = self.easy_topk(logits, self.moe_topk)
        else:
            raise ValueError(f"Unsupported topk_impl: {topk_impl}")

        return gate_output

    @staticmethod
    def easy_topk(logits, moe_topk):
        gates = F.softmax(logits, dim=1)
        topk_weight_1, expert_index = torch.topk(gates, moe_topk)
        weight_sums = topk_weight_1.sum(dim=1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        topk_weight = topk_weight_1 / weight_sums

        return topk_weight, expert_index


class HunyuanMoE(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = config.moe_topk
        self.num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        if config.use_mixed_mlp_moe:
            self.shared_mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True)
        self.gate = HunyuanTopKGate(config, layer_idx=layer_idx)
        self.experts = nn.ModuleList(
            [HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True) for _ in range(self.num_experts)]
        )

        self._moe_impl = config.moe_impl
        # For FlashInfer
        self.moe_weight = None
        self.moe_weight_2 = None
        self._weights_initialized = False

    @property
    def moe_impl(self):
        return self._moe_impl

    @moe_impl.setter
    def moe_impl(self, value):
        self._moe_impl = value
        if self._moe_impl == "flashinfer":
            assert flashinfer is not None, "When using fused_moe, flashinfer must be installed."

    def forward(self, hidden_states):
        torch.cuda.set_device(hidden_states.device.index)
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        reshaped_input = hidden_states.reshape(-1, hidden_size) # [bsz*seq_len, hidden_size]

        if self._moe_impl == "flashinfer":
            # Get expert weights
            if not self._weights_initialized:
                self._initialize_weights_on_device(hidden_states.device)
            topk_weight, topk_index = self.gate(hidden_states, topk_impl='easy')

            combined_output = torch.zeros_like(reshaped_input)
            _ = flashinfer.fused_moe.cutlass_fused_moe(     # noqa
                reshaped_input.contiguous(),
                topk_index.to(torch.int).contiguous(),
                topk_weight.to(torch.float).contiguous(),
                self.moe_weight,
                self.moe_weight_2,
                torch.bfloat16,
                output=combined_output,
                quant_scales=None,
            )
        else:
            # Original implementation - fallback for compatibility
            l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states, topk_impl='default')
            dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
            chunks = dispatched_input.chunk(self.num_experts, dim=0)
            expert_outputs = []
            for chunk, expert in zip(chunks, self.experts):
                expert_outputs.append(expert(chunk))

            expert_output = torch.cat(expert_outputs, dim=0)
            combined_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)

        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output    # noqa
        else:
            output = combined_output

        return output

    def _initialize_weights_on_device(self, device):
        expert_weights_gate_up = []
        expert_weights_down = []

        for expert in self.experts:
            expert.to(device)
            expert_weights_gate_up.append(expert.gate_and_up_proj.weight.to(device))
            expert_weights_down.append(expert.down_proj.weight.to(device))

        self.moe_weight = torch.stack(expert_weights_gate_up).contiguous()
        self.moe_weight_2 = torch.stack(expert_weights_down).contiguous()
        # empty the expert weights
        for expert in self.experts:
            expert.gate_and_up_proj.weight.data = torch.empty(0, device=device)
            if expert.gate_and_up_proj.bias is not None:
                expert.gate_and_up_proj.bias.data = torch.empty(0, device=device)
            expert.down_proj.weight.data = torch.empty(0, device=device)
            if expert.down_proj.bias is not None:
                expert.down_proj.bias.data = torch.empty(0, device=device)

        self._weights_initialized = True


class HunYuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HunYuanSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = -1,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Get layer_id topk if config.moe_topk is a list
        if isinstance(config.moe_topk, list):
            assert layer_id >= 0
            assert len(config.moe_topk) > layer_id
            top_k = config.moe_topk[layer_id]
        else:
            top_k = config.moe_topk

        # If it is moe, moe_intermediate_size is preferred
        intermediate_size = config.intermediate_size
        if config.moe_intermediate_size is not None:
            intermediate_size = (
                config.moe_intermediate_size
                if isinstance(config.moe_intermediate_size, int)
                else config.moe_intermediate_size[layer_id]
            )

        # Load balancing settings.
        # vllm_config = get_current_vllm_config()
        # eplb_config = vllm_config.parallel_config.eplb_config
        # 
        self.enable_eplb = False

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.use_mixed_mlp_moe > 0:
            # Get layer_id num_shared_expert if config.num_shared_expert is
            # a list.
            if isinstance(config.num_shared_expert, list):
                assert layer_id >= 0
                assert len(config.num_shared_expert) > layer_id
                num_shared_expert = config.num_shared_expert[layer_id]
            else:
                num_shared_expert = config.num_shared_expert

            self.shared_mlp = HunYuanMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size * num_shared_expert,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_mlp",
            )
        else:
            self.shared_mlp = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_mlp,
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            renormalize=top_k > 1,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            pcp_size=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.shared_mlp is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


def _get_cla_factor(config: PretrainedConfig) -> int:
    if not getattr(config, "use_cla", False):
        return 1
    return getattr(config, "cla_share_factor", 1)


class HunYuanAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_key_value_heads = config.num_key_value_heads // tp_size if config.num_key_value_heads else self.num_heads
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if hasattr(config, "head_dim") and config.head_dim:
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_scaling is not None:
            # for t2t, fallback rotary_emb type from 'custom' to default.
            rope_scaling["rope_type"] = "default"
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_scaling,
            is_neox_style=True,
        )
        # which attention, from vllm or omni?
        # try use Atten from diffusion
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            # cache_config=cache_config,
            # quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # default image_token_len = timestamp + 4096*image_tokes
        self.image_attn = ImageKVCacheManager(image_token_len=4097)
        self.image_rope2d_emb = HunYuanRotary2DEmbedder(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: tuple[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        custom_pos_emb: tuple[torch.FloatTensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        past_key_value: Optional[Cache] = kwargs.get("past_key_value", None)
        if past_key_value is not None:
            position_ids = kwargs.get("position_ids")
            key_states = k.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = v.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_id, cache_kwargs)
        # for image_generation
        if kwargs.get("mode", "gen_text") == "gen_image":
            # assert positions is None, "positions should be None for image attention"
            q, k = self.image_rope2d_emb(q, k, hidden_states, custom_pos_emb, **kwargs)
        else:
            q, k = self.rotary_emb(positions, q, k)
        ori_k = k
        if self.use_qk_norm:
            q = self.query_layernorm(
                q.view(-1, self.num_heads, self.head_dim).contiguous()
            )
            k = self.key_layernorm(
                k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
            )
        # for image_generation
        if kwargs.get("mode", "gen_text") == "gen_image":
            attn_output = self.image_attn(
                q, k, v, attention_mask=attention_mask, **kwargs
            )
        else:
            attn_output = self.attn(q, k, v)
        # For o_proj
        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        output = output.reshape(bsz, q_len, -1)
        return output, None, past_key_value


class HunYuanCrossAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_scaling,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: tuple[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        assert kv_states is not None
        ori_k, v = kv_states  # use last layer kv,
        k = ori_k
        q, _ = self.q_proj(hidden_states)
        k_tmp = torch.empty_like(k)  # Todo: reduant rotary embedding
        q, _ = self.rotary_emb(positions, q, k_tmp)
        if self.use_qk_norm:
            q = self.query_layernorm(
                q.view(-1, self.num_heads, self.head_dim).contiguous()
            )
            k = self.key_layernorm(
                k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
            )

        attn_output = self.attn(q, k, v)
        # For o_proj
        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output, (ori_k, v)


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self, config: HunyuanImage3Config, layer_idx: int, prefix:str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.intermediate_size = (
            config.intermediate_size
            if isinstance(config.intermediate_size, int)
            else config.intermediate_size[layer_idx]
        )
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )

        # origin attn
        # attn_impl = config._attn_implementation     # noqa
        # if attn_impl in Hunyuan_ATTENTION_CLASSES:
        #     self.self_attn = Hunyuan_ATTENTION_CLASSES[attn_impl](config=config, layer_idx=layer_idx)
        # else:
        #     raise ValueError(f"Unsupported attention implementation: {attn_impl}")
        cla_factor = _get_cla_factor(config)
        attention_type = (
            AttentionType.ENCODER_DECODER
            if layer_idx >= 0 and layer_idx % cla_factor != 0
            else AttentionType.DECODER
        )
        if attention_type == AttentionType.DECODER:
            self.self_attn = HunYuanAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=None,
                bias=attention_bias,
                cache_config=None,
                prefix=f"{prefix}.self_attn",
                layer_id=layer_idx,
            )
        elif attention_type == AttentionType.ENCODER_DECODER:
            self.self_attn = HunYuanCrossAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=None,
                bias=attention_bias,
                cache_config=None,
                prefix=f"{prefix}.self_attn",
                layer_id=layer_idx,
            )
        else:
            raise RuntimeError(f"Unsupported attention type: {attention_type}")

        if ((isinstance(config.num_experts, int) and config.num_experts > 1) or (
                isinstance(config.num_experts, list) and max(
                config.num_experts) > 1)) and layer_idx >= config.moe_layer_num_skipped:
            self.mlp = HunYuanSparseMoeBlock(config, layer_id=layer_idx, prefix=f"{prefix}.mlp")
        else:
            self.mlp = HunYuanMLP(self.hidden_size, self.intermediate_size,
                                  config.hidden_act)
        if config.norm_type == 'hf_rms' or config.norm_type == 'rms':
            self.input_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif config.norm_type == 'fused' or config.norm_type == 'torch_nn':
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor | Any]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            custom_pos_emb (`Tuple[torch.FloatTensor]`, *optional*): custom position embedding for rotary
                position embedding
        """
        if "padding_mask" in kwargs:
            logger.warning(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
                "`attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)


        # Self Attention
        # hidden_states, ori_kv_states = self.self_attn(
        #     positions=position_ids,
        #     hidden_states=hidden_states,
        #     kv_states
        # )
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            custom_pos_emb=custom_pos_emb,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # print(f"---------residual:{residual.shape}, hidden_states:{hidden_states.shape}")
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class HunyuanImage3PreTrainedModel(PreTrainedModel):
    config_class = HunyuanImage3Config
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunyuanImage3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def _is_moe(config: PretrainedConfig) -> bool:
    num_experts = getattr(config, "num_experts", None)
    if isinstance(num_experts, int):
        return num_experts > 1
    if isinstance(num_experts, list) and num_experts:
        # Ensure all elements are integers before calling max.
        if all(isinstance(e, int) for e in num_experts):
            return max(num_experts) > 1
        else:
            return False
    return False


def _get_cla_factor(config: PretrainedConfig) -> int:
    if not getattr(config, "use_cla", False):
        return 1
    return getattr(config, "cla_share_factor", 1)


class HunyuanImage3Model(nn.Module):
    def __init__(self, config: HunyuanImage3Config, prefix: str = ""):
        super().__init__()

        config = config
        cache_config = None
        quant_config = None
        lora_config = None
        eplb_config = None
        enable_eplb = False
        self.num_redundant_experts = 0

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        # maybe should not use make_layers()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HunyuanImage3DecoderLayer(
                config=config,
                layer_idx=int(prefix.split(".")[-1]),
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def _split_qkv_weight(self, qkv: torch.Tensor):
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size

        if hasattr(self.config, "head_dim"):
            attention_head_dim = self.config.head_dim
        elif hasattr(self.config, "attention_head_dim"):
            attention_head_dim = self.config.attention_head_dim
        else:
            attention_head_dim = self.config.hidden_size // num_attention_heads

        qkv = qkv.reshape(
            num_kv_heads, num_key_value_groups + 2, attention_head_dim, hidden_size
        )
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
        return torch.concat((q, k, v))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        if _is_moe(self.config):
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            fused_moe_expert_mapping = SharedFusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )
            expert_weights_remaping = {
                "gate_proj": ("gate_and_up_proj", 1, 2),
                "up_proj": ("gate_and_up_proj", 0, 2),
            }
            return fused_moe_expert_mapping, expert_weights_remaping
        else:
            return [], {}

    # rename for delay load
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        tp_rank = get_tensor_model_parallel_rank()
        print(f"load tp: {tp_rank}")
        cla_factor = _get_cla_factor(self.config)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj",
                ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],
                self._split_qkv_weight,
            ),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        pass
        expert_params_mapping, expert_weights_remapping = self.get_expert_mapping()

        # List of unexpected keywords in weight names
        unexpected_keywords = [
            "vae",
            "vision_aligner",
            "vision_model",
            "final_layer",
            "patch_embed",
            "timestep_emb",
            "time_embed",
            "time_embed_2",
            "guidance_emb",
            "timestep_r_emb",
        ]

        def contains_unexpected_keyword(name, keywords):
            """Check if the name contains any unexpected keywords"""
            for keyword in keywords:
                if keyword in name:
                    return True
            return False

        for name, loaded_weight in weights:
            # print(f"Loading weight name: {name}, tp_rank: {tp_rank}", flush=True)
            if contains_unexpected_keyword(name, unexpected_keywords):
                print(f"Skipping unexpected weight name: {name}")
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                # cross layer only have q_proj, skip qkv pack
                if weight_name == ".q_proj":
                    match = re.search(r"layers\.\d+", name)
                    if match:
                        layer_id = int(match.group(0).split(".")[-1])
                        if cla_factor > 1 and layer_id % cla_factor != 0:
                            continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for (
                param_name,
                weight_name,
                den,
                split_param,
                func,
            ) in split_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den

                param = params_dict[name]
                weight_loader = param.weight_loader
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    if func:
                        weight_loader(
                            param, func(loaded_weight)[offset:new_offset], shard_id
                        )
                    else:
                        weight_loader(param, loaded_weight[offset:new_offset], shard_id)
                    offset = new_offset

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                is_expert_weight = False
                is_found = False
                found_num = 0
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    offset = 0
                    den = 1
                    # if tp_rank == 0:
                    #     print(f"origin weight_name: {weight_name}, param_name: {param_name}, name: {name}")
                    for (
                        mapped_weight_substr,
                        origin_weight_info,
                    ) in expert_weights_remapping.items():
                        if mapped_weight_substr in weight_name:
                            origin_weight_name, offset, den = origin_weight_info
                            weight_name = weight_name.replace(
                                mapped_weight_substr, origin_weight_name
                            )
                            break
                    # if tp_rank == 0:
                    #     print(f"remapped weight_name: {weight_name}, offset: {offset}, den: {den}")
                    if weight_name not in name:
                        continue
                    # this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)
                    found_num += 1
                    # if tp_rank == 0:
                    #     print(f"name_mapped: {name_mapped}, found_num: {found_num}")
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = cast(
                        Callable[..., bool], param.weight_loader
                    )
                    assert loaded_weight.shape[0] % den == 0
                    units = loaded_weight.shape[0] // den

                    success = weight_loader(
                        param,
                        loaded_weight[offset * units : offset * units + units],
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        is_found = True
                        if found_num == den:
                            break
                if is_found:
                    continue
                if is_expert_weight:
                    # We've checked that this is an expert weight
                    # However it's not mapped locally to this rank
                    # So we simply skip it
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if "mlp.gate.wg." in name:
                    name = name.replace("wg.", "")
                if name == "ln_f.weight":
                    name = "norm.weight"
                if name == "wte.weight":
                    name = "embed_tokens.weight"
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            mode: str = "gen_text",
            first_step: Optional[bool] = None,
            query_lens: Optional[list[int]] = None,
            seq_lens: Optional[list[int]] = None,
            num_image_tokens: Optional[int] = None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        print("call HunyuanImage3 HunyuanImage3Model forward")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            
            layer_outputs = decoder_layer(
                positions=None,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                mode=mode,
                first_step=first_step,
                query_lens=query_lens,
                seq_lens=seq_lens,
                num_image_tokens=num_image_tokens,
                gen_timestep_scatter_index=gen_timestep_scatter_index,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # if not self.add_classification_head:
        #     # Do ln_f outside of the model for compatibility with image generation.
        #     pass
            # hidden_states = self.ln_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        reverse (`bool`, defaults to `True`):
            Whether to reverse the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: float = 1.0,
            reverse: bool = True,
            solver: str = "euler",
            use_flux_shift: bool = False,
            flux_base_shift: float = 0.5,
            flux_max_shift: float = 1.15,
            n_tokens: Optional[int] = None,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self.timesteps_full = (sigmas * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = [
            "euler",
            "heun-2", "midpoint-2",
            "kutta-4",
        ]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")

        # empty dt and derivative (for heun)
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    @property
    def state_in_first_order(self):
        return self.derivative_1 is None

    @property
    def state_in_second_order(self):
        return self.derivative_2 is None

    @property
    def state_in_third_order(self):
        return self.derivative_3 is None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None,
                      n_tokens: int = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
        """
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        # Apply timestep shift
        if self.config.use_flux_shift:
            assert isinstance(n_tokens, int), "n_tokens should be provided for flux shift"
            mu = self.get_lin_function(y1=self.config.flux_base_shift, y2=self.config.flux_max_shift)(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif self.config.shift != 1.:
            sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self.timesteps_full = (sigmas * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)

        # empty dt and derivative (for kutta)
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

        # Reset step index
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    @staticmethod
    def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            pred_uncond: torch.FloatTensor = None,
            generator: Optional[torch.Generator] = None,
            n_tokens: Optional[int] = None,
            return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
        pred_uncond = pred_uncond.to(torch.float32) if pred_uncond is not None else None

        # dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        last_inner_step = True
        if self.config.solver == "euler":
            derivative, dt, sample, last_inner_step = self.first_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver in ["heun-2", "midpoint-2"]:
            derivative, dt, sample, last_inner_step = self.second_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver == "kutta-4":
            derivative, dt, sample, last_inner_step = self.fourth_order_method(model_output, sigma, sigma_next, sample)
        else:
            raise ValueError(f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}")

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        # prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        if last_inner_step:
            self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def first_order_method(self, model_output, sigma, sigma_next, sample):
        derivative = model_output
        dt = sigma_next - sigma
        return derivative, dt, sample, True

    def second_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            # store for 2nd order step
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample

            derivative = model_output
            if self.config.solver == 'heun-2':
                dt = self.dt
            elif self.config.solver == 'midpoint-2':
                dt = self.dt / 2
            else:
                raise NotImplementedError(f"Solver {self.config.solver} not supported.")
            last_inner_step = False

        else:
            if self.config.solver == 'heun-2':
                derivative = 0.5 * (self.derivative_1 + model_output)
            elif self.config.solver == 'midpoint-2':
                derivative = model_output
            else:
                raise NotImplementedError(f"Solver {self.config.solver} not supported.")

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample
            last_inner_step = True

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.derivative_1 = None
            self.dt = None
            self.sample = None

        return derivative, dt, sample, last_inner_step

    def fourth_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False

        elif self.state_in_second_order:
            self.derivative_2 = model_output
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False

        elif self.state_in_third_order:
            self.derivative_3 = model_output
            derivative = model_output
            dt = self.dt
            last_inner_step = False

        else:
            derivative = (1/6 * self.derivative_1 + 1/3 * self.derivative_2 + 1/3 * self.derivative_3 +
                          1/6 * model_output)

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample
            last_inner_step = True

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.derivative_1 = None
            self.derivative_2 = None
            self.derivative_3 = None
            self.dt = None
            self.sample = None

        return derivative, dt, sample, last_inner_step

    def __len__(self):
        return self.config.num_train_timesteps


class ClassifierFreeGuidance:
    def __init__(
        self,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()
        self.use_original_formulation = use_original_formulation

    def __call__(
            self,
            pred_cond: torch.Tensor,
            pred_uncond: Optional[torch.Tensor],
            guidance_scale: float,
            step: int,
        ) -> torch.Tensor:

        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + guidance_scale * shift

        return pred


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def real_batched_index_select(t, dim, idx):
    """ index_select for batched index and batched t """
    assert t.ndim >= 2 and idx.ndim >= 2, f"{t.ndim=} {idx.ndim=}"
    assert len(t) == len(idx), f"{len(t)=} != {len(idx)=}"
    return torch.stack([torch.index_select(t[i], dim - 1, idx[i]) for i in range(len(t))])


@dataclass
class HunyuanImage3Text2ImagePipelineOutput(BaseOutput):
    samples: Union[List[Any], np.ndarray]


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.
    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class HunyuanImage3Text2ImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for condition-to-sample generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        model ([`ModelMixin`]):
            A model to denoise the diffused latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `diffusion_model` to denoise the diffused latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    model_cpu_offload_seq = ""
    _optional_components = []
    _exclude_from_cpu_offload = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        model,
        scheduler: SchedulerMixin,
        vae,
        progress_bar_config: Dict[str, Any] | None = None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, '_progress_bar_config'):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)
        # ==========================================================================================

        self.register_modules(
            model=model,
            scheduler=scheduler,
            vae=vae,
        )

        # should be a tuple or a list corresponding to the size of latents (batch_size, channel, *size)
        # if None, will be treated as a tuple of 1
        self.latent_scale_factor = self.model.config.vae_downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)

        # Must start with APG_mode_
        self.cfg_operator = ClassifierFreeGuidance()

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_kwargs[k] = v
        return extra_kwargs

    def prepare_latents(self, batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
        if self.latent_scale_factor is None:
            latent_scale_factor = (1,) * len(image_size)
        elif isinstance(self.latent_scale_factor, int):
            latent_scale_factor = (self.latent_scale_factor,) * len(image_size)
        elif isinstance(self.latent_scale_factor, tuple) or isinstance(self.latent_scale_factor, list):
            assert len(self.latent_scale_factor) == len(image_size), \
                "len(latent_scale_factor) shoudl be the same as len(image_size)"
            latent_scale_factor = self.latent_scale_factor
        else:
            raise ValueError(
                f"latent_scale_factor should be either None, int, tuple of int, or list of int, "
                f"but got {self.latent_scale_factor}"
            )

        latents_shape = (
            batch_size,
            latent_channel,
            *[int(s) // f for s, f in zip(image_size, latent_scale_factor)],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def set_scheduler(self, new_scheduler):
        self.register_modules(scheduler=new_scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        image_size: List[int],
        num_inference_steps: int = 50,
        timesteps: List[int] | None = None,
        sigmas: List[float] | None = None,
        guidance_scale: float = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_rescale: float = 0.0,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        model_kwargs: Dict[str, Any] | None = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The text to guide image generation.
            image_size (`Tuple[int]` or `List[int]`):
                The size (height, width) of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate samples closely linked to the
                `condition` at the expense of lower sample quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for sample
                generation. Can be used to tweak the same generation with different conditions. If not provided,
                a latents tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~DiffusionPipelineOutput`] instead of a
                plain tuple.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~DiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~DiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated samples.
        """

        print("call HunyuanImage3 HunyuanImage3Text2ImagePipeline forward")
        callback_steps = kwargs.pop("callback_steps", None)
        pbar_steps = kwargs.pop("pbar_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        cfg_factor = 1 + self.do_classifier_free_guidance

        # Define call parameters
        device = self._execution_device

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas,
        )

        # Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size,
            latent_channel=self.model.config.vae["latent_channels"],
            image_size=image_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
            latents=latents,
        )

        # Prepare extra step kwargs.
        _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator}
        )

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        b, _, q_len1, seq_len = attention_mask.shape
        query_lens=[q_len1] * b
        seq_lens=[seq_len] * b
        model_kwargs["query_lens"] = query_lens
        model_kwargs["seq_lens"] = seq_lens
        model_kwargs["attention_mask"] = attention_mask.to(latents.device)

        # Sampling loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * cfg_factor)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])

                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids,
                    images=latent_model_input,
                    timestep=t_expand,
                    **model_kwargs,
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    model_output = self.model.forward_call(**model_inputs, first_step=(i == 0))
                    pred = model_output["diffusion_prediction"]
                pred = pred.to(dtype=torch.float32)

                # perform guidance
                if self.do_classifier_free_guidance:
                    pred_cond, pred_uncond = pred.chunk(2)
                    pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False)[0]

                if i != len(timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    if input_ids.shape[1] != model_kwargs["position_ids"].shape[1]:
                        input_ids = torch.gather(input_ids, 1, index=model_kwargs["position_ids"])
                    attention_mask = model_kwargs.get("attention_mask")
                    b, _, q_len1, seq_len = attention_mask.shape
                    query_lens=[q_len1] * b
                    seq_lens=[seq_len] * b
                    model_kwargs["query_lens"] = query_lens
                    model_kwargs["seq_lens"] = seq_lens

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor

        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        # torch.cuda.synchronize()
        # t_vae_s = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        # torch.cuda.synchronize()
        # t_vae_e = time.time()
        # print("Vae time: ", t_vae_e - t_vae_s)

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert image.shape[2] == 1, "image should have shape [B, C, T, H, W] and T should be 1"
            image = image.squeeze(2)

        # torch.cuda.synchronize()
        # t_img_post_s = time.time()

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # torch.cuda.synchronize()
        # t_img_post_e = time.time()
        # print("Image post process time: ", t_img_post_e - t_img_post_s)

        if not return_dict:
            return (image,)

        return HunyuanImage3Text2ImagePipelineOutput(samples=image)


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self,
                 hidden_size,
                 act_layer=nn.GELU,
                 frequency_embedding_size=256,
                 max_period=10000,
                 out_size=None,
                 dtype=None,
                 device=None
                 ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels, **kwargs):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: a nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels, **kwargs)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        # assert num are all integers
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), f"num should be int, but got {num}"
        num = num_int
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)       # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)      # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)       # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")   # dim x [H, W]
    grid = torch.stack(grid, dim=0)     # [dim, H, W]

    return grid


def build_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):
    """
    Reference: https://kexue.fm/archives/10352

    Start from 1, we have
        beta_y = L + (wh - h)/2
        beta_x = L + (wh - w)/2

    Returns
    -------
    cos: torch.Tensor with shape of [seq_len, n_elem]
    sin: torch.Tensor with shape of [seq_len, n_elem]
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)    # [1, half_d, 2]

    # position indices
    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start   # start from 0, so image_slice.start is just L
            # previous text
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L))
                x_sections.append(torch.arange(last_pos, L))
            elif h is None:
                # Interleave data has overlapped positions for <boi> <size> <ratio> <timestep> <eoi> tokens.
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                continue
            else:
                # Interleave data has overlapped positions for noised image and the successive clean image,
                # leading to last_pos (= last text end L + noise w * h) > L (last text end L).
                pass
            # current image
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w))  # [2, h, w]
            grid = grid.reshape(2, -1)  # (y, x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            # step
            last_pos = L + w * h
        # final text
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)    # [seq_len, 1, 2]

    # calc rope
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)

    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def build_batch_2d_rope(
        seq_len: int, n_elem: int, image_infos: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
        device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0,
        return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len, n_elem, image_infos=image_info, device=device,
            base=base, base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param in_channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels, **factory_kwargs)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)
            ),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs
            )
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1. + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    """
    patch_size: one of [1, 2 ,4 ,8]
    in_channels: vae latent dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: transformer model dim
    """
    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels,
                 dropout=0.0, device=None, dtype=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [conv_nd(
                2,
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                **factory_kwargs
            )]
        )

        if self.patch_size == 1:
            self.model.append(ResBlock(
                in_channels=hidden_channels,
                emb_channels=emb_channels,
                out_channels=out_channels,
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    in_channels=hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels if (i + 1) * 2 != self.patch_size else out_channels,
                    dropout=dropout,
                    down=True,
                    **factory_kwargs
                ))

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, token_h, token_w


class UNetUp(nn.Module):
    """
    patch_size: one of [1, 2 ,4 ,8]
    in_channels: transformer model dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: vae latent dim
    """
    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels,
                 dropout=0.0, device=None, dtype=None, out_norm=False):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(ResBlock(
                in_channels=in_channels,
                emb_channels=emb_channels,
                out_channels=hidden_channels,
                dropout=dropout,
                **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    in_channels=in_channels if i == 0 else hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    up=True,
                    **factory_kwargs
                ))

        if out_norm:
            self.model.append(nn.Sequential(
                normalization(hidden_channels, **factory_kwargs),
                nn.SiLU(),
                conv_nd(
                    2,
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs
                ),
            ))
        else:
            self.model.append(conv_nd(
                2,
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                **factory_kwargs
            ))

    # batch_size, seq_len, model_dim
    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


class HunyuanStaticCache(StaticCache):
    """
    A custom static cache for multi-modal models that supports dynamic extension of the cache
    and inplace updates of the cache.

    This cache supports batch cache_position updates.
    """
    def __init__(self, *args, **kwargs):
        self.dynamic = kwargs.pop("dynamic", False)
        super().__init__(*args, **kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            if self.key_cache[layer_idx].device != key_states.device:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            k_out = self.key_cache[layer_idx]
            v_out = self.value_cache[layer_idx]
            key_states = key_states.to(k_out.dtype)
            value_states = value_states.to(v_out.dtype)
        else:
            if self.layers[layer_idx].keys is None:
                self.layers[layer_idx].lazy_initialization(key_states)
            k_out = self.layers[layer_idx].keys
            v_out = self.layers[layer_idx].values

        if k_out.shape != key_states.shape:
            print("cache tensor not equal!")

        if cache_position is None:
            # print(f"----updating static cache: key_states: {key_states.shape.shape}, k_out: {k_out.shape}, value_states: {value_states.shape}, v_out: {v_out.shape}")
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            if cache_position.dim() == 1:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)

                if self.dynamic:
                    end = cache_position[-1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]
            else:
                assert cache_position.dim() == 2, f"multiple batch dims not yet {cache_position.shape=}"
                batch_size, idx_size = cache_position.shape
                assert batch_size == k_out.size(0)
                assert batch_size == v_out.size(0)
                assert batch_size == key_states.size(0)
                assert batch_size == value_states.size(0)
                for i in range(batch_size):
                    unbatched_dim = 1
                    k_out[i].index_copy_(unbatched_dim, cache_position[i], key_states[i])
                    v_out[i].index_copy_(unbatched_dim, cache_position[i], value_states[i])

                if self.dynamic:
                    assert len(cache_position) == 1
                    end = cache_position[0, -1].item() + 1
                    k_out = k_out[:, :, :end]
                    v_out = v_out[:, :, :end]

        return k_out, v_out
