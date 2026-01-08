import logging
from typing import Optional, Tuple, Any, List, Union

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
# from vllm_omni.diffusion.distributed.parallel_state import get_ep_group

from mindiesd import rotary_position_embedding

from transformers.activations import ACT2FN
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

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

        attn_impl = config._attn_implementation     # noqa
        if attn_impl in Hunyuan_ATTENTION_CLASSES:
            self.self_attn = Hunyuan_ATTENTION_CLASSES[attn_impl](config=config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unsupported attention implementation: {attn_impl}")

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
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
            **kwargs,
        )
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


class HunyuanImage3Model(HunyuanImage3PreTrainedModel):
    def __init__(self, config: HunyuanImage3Config, prefix: str = ""):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.add_classification_head = config.add_classification_head
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HunyuanImage3DecoderLayer(config, layer_idx, f"{prefix}.layers.{layer_idx}") for layer_idx in range(config.num_hidden_layers)]
        )
        if not config.add_classification_head:
            self.ln_f = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        self.shared_tensor = None

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
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                mode=mode,
                first_step=first_step,
                gen_timestep_scatter_index=gen_timestep_scatter_index,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if not self.add_classification_head:
            # Do ln_f outside of the model for compatibility with image generation.
            pass
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
