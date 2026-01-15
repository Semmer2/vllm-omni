# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
from collections.abc import Iterable
from typing import Any
import random

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from dataclasses import fields
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

from .config.generation_config import generation_config
from .config.HunyuanImage3Config import HunyuanImage3Config_dict
from .autoencoder_kl_3d import AutoencoderKLConv3D
from .configuration_hunyuan import HunyuanImage3Config
from .image_processor import HunyuanImage3ImageProcessor
from .tokenizer_wrapper import TokenizerWrapper, ImageInfo, JointImageInfo
logger = logging.getLogger(__name__)

#CONDITION_IMAGE_SIZE = 384 * 384
#VAE_IMAGE_SIZE = 1024 * 1024
from typing import TYPE_CHECKING, List, Union, Optional, Dict, Any, Tuple, Callable
from transformers.cache_utils import Cache, StaticCache
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

        if cache_position is None:
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

@staticmethod
def prepare_seed(seed, batch_size):
    if isinstance(seed, torch.Tensor):
        seed = seed.tolist()
    if seed is None:
        seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
    elif isinstance(seed, int):
        seeds = [seed for _ in range(batch_size)]
    elif isinstance(seed, (list, tuple)):
        if len(seed) == batch_size:
            seeds = [int(seed[i]) for i in range(batch_size)]
        else:
            raise ValueError(f"Length of seed must be equal to the batch_size({batch_size}), got {seed}.")
    else:
        raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
    return seeds 

def vae_encode(initialconfig, image, cfg_factor=1):
        vae = AutoencoderKLConv3D.from_config(initialconfig.vae)
        config = vae.config

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            vae_encode_result = vae.encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(config, 'shift_factor') and config.shift_factor:
                latents.sub_(config.shift_factor)
            if hasattr(config, 'scaling_factor') and config.scaling_factor:
                latents.mul_(config.scaling_factor)

        if hasattr(vae, "ffactor_temporal"):
            assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents = latents.squeeze(2)

        # Here we always use t=0 to declare it is a clean conditional image
        t = torch.zeros((latents.shape[0],))

        if cfg_factor > 1:
            t = t.repeat(cfg_factor)
            latents = latents.repeat(cfg_factor, 1, 1, 1)

        return t, latents

def _encode_cond_image(
            initialconfig,
            device,
            batch_cond_image_info_list: List[List[JointImageInfo]],
            cfg_factor: int = 1,
    ):
        # VAE encode one by one, as we assume cond images have different sizes
        batch_cond_vae_images, batch_cond_t, batch_cond_vit_images = [], [], []
        for cond_image_info_list in batch_cond_image_info_list:
            cond_vae_image_list, cond_t_list, cond_vit_image_list = [], [], []
            for image_info in cond_image_info_list:
                cond_t_, cond_vae_image_ = vae_encode(
                    initialconfig, image_info.vae_image_info.image_tensor.to(device),
                )
                cond_vit_image_list.append(image_info.vision_image_info.image_tensor)
                cond_vae_image_list.append(cond_vae_image_.squeeze(0))
                cond_t_list.append(cond_t_)
            batch_cond_vae_images.append(cond_vae_image_list)
            batch_cond_t.append(cond_t_list)
            batch_cond_vit_images.append(torch.cat(cond_vit_image_list, dim=0))

        # If only one cond image for each sample and all have the same size, we can batch them together
        # In this case, cond_vae_images is a 4-D tensor.
        if all([len(items) == 1 for items in batch_cond_vae_images]) and all(
                items[0].shape == batch_cond_vae_images[0][0].shape for items in batch_cond_vae_images):
            cond_vae_images = torch.stack([items[0] for items in batch_cond_vae_images], dim=0)
            cond_t = torch.cat([items[0] for items in batch_cond_t], dim=0)
            if cfg_factor > 1:
                cond_t = cond_t.repeat(cfg_factor)
                cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
        else:
            # In this case, cond_vae_images is a list of 4-D tensors or a list of lists of 3-D tensors.
            cond_t = [torch.cat(item, dim=0) for item in batch_cond_t]
            cond_vae_images = []
            for items in batch_cond_vae_images:
                if all(items[0].shape == item.shape for item in items):
                    cond_vae_images.append(torch.stack(items, dim=0))
                else:
                    cond_vae_images.append(items)
            if cfg_factor > 1:
                cond_t = cond_t * cfg_factor
                cond_vae_images = cond_vae_images * cfg_factor

        if cfg_factor > 1:
            batch_cond_vit_images = batch_cond_vit_images * cfg_factor

        return cond_vae_images, cond_t, batch_cond_vit_images

def build_batch_rope_image_info(output, sections):
        rope_image_info = []
        for image_slices, sections_i in zip(output.all_image_slices, sections):
            image_shapes = []
            for section in sections_i:
                if 'image' in section['type']:
                    if isinstance(section['token_height'], list):
                        assert len(section['token_height']) == len(section['token_height']), \
                            (f"token_height and token_width should have the same length, "
                             f"but got {len(section['token_height'])} and {len(section['token_width'])}")
                        image_shapes.extend(list(zip(section['token_height'], section['token_width'])))
                    else:
                        image_shapes.append((section['token_height'], section['token_width']))
            assert len(image_slices) == len(image_shapes), (
                f"Size miss matching: Image slices({len(image_slices)}) != image shapes({len(image_shapes)})"
            )
            rope_image_info.append(list(zip(image_slices, image_shapes)))
        return rope_image_info


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


def default(val, d):
    return val if val is not None else d


def to_device(data, device):
    if device is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        return data
    
def prepare_model_inputs(
            prompt=None,
            mode="gen_image",
            system_prompt=None,
            cot_text=None,
            image_size="auto",
            message_list=None,
            device=None,
            max_new_tokens=None,
            tokenizer = "/data1/s00957182/hy/weights/HunyuanImage-3.0"
    ):
        # 1. Sanity check
        #self.check_inputs(prompt, message_list)
        #device = default(device, self.device)

        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt
        batch_cot_text = cot_text
        batch_system_prompt = system_prompt
        batch_gen_image_info = None
        # TODO: construct with user input images
        batch_cond_image_info = None
        config = HunyuanImage3Config(**HunyuanImage3Config_dict)
        image_processor = HunyuanImage3ImageProcessor(config)
        #   -- 2.1 message_list
        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)

            batch_gen_image_info = [
                [message['content'] for message in message_list_ if message['type'] == 'gen_image']
                for message_list_ in batch_message_list
            ]
            # At most one gen_image is allowed for each message_list
            batch_gen_image_info = [info[-1] if len(info) > 0 else None for info in batch_gen_image_info]
            # Multiple cond images are allowed.
            batch_cond_image_info = [
                [message['content'] for message in message_list_ if message['type'] == 'joint_image']
                for message_list_ in batch_message_list
            ]

        #   -- 2.2 Prompt, cot text, system prompt
        else:
            if isinstance(batch_prompt, str):
                batch_prompt = [batch_prompt]
            batch_size = len(batch_prompt)

            if batch_cot_text is not None:
                if isinstance(batch_cot_text, str):
                    batch_cot_text = [batch_cot_text]
                else:
                    assert isinstance(batch_cot_text, list) and len(batch_cot_text) == batch_size, \
                        "`cot_text` should be a string or a list of strings with the same length as `prompt`."

            if batch_system_prompt is not None:
                if isinstance(batch_system_prompt, str):
                    batch_system_prompt = [batch_system_prompt]
                else:
                    assert isinstance(batch_system_prompt, list) and len(batch_system_prompt) == batch_size, \
                        "`system_prompts` should be a string or a list of strings with the same length as `prompt`."

            if mode == "gen_image":
                batch_gen_image_info = [image_processor.build_image_info(image_size) for _ in range(batch_size)]

        #   -- 2.3 seed
        seeds = prepare_seed(seed=None, batch_size=batch_size)
        generator = [torch.Generator(device).manual_seed(seed) for seed in seeds]

        # 3. apply chat template
        cfg_factor = {"gen_text": 1, "gen_image": 2}
        bot_task = "auto"#hard code
        # If `drop_think` enabled, always drop <think> parts in the context.
        drop_think = False
        # Apply batched prompt or batched message_list to build input sequence with associated info.

        tkwrapper = TokenizerWrapper(tokenizer)
        out = tkwrapper.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_image_info=batch_cond_image_info,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=None,
            bot_task=bot_task,
            image_base_size=1024,
            sequence_template=generation_config['sequence_template'],
            cfg_factor=cfg_factor[mode],
            drop_think=drop_think,
        )
        output, sections = out['output'], out['sections']

        # 4. Encode conditional images
        if batch_cond_image_info is not None and len(batch_cond_image_info[0]) > 0:
            cond_vae_images, cond_timestep, cond_vit_images = _encode_cond_image(
                config,device,batch_cond_image_info, cfg_factor[mode]
            )
            # Pack vit kwargs. Siglip2-so requires spatial_shapes and attention_mask for inference.
            vit_kwargs = {"spatial_shapes": [], "attention_mask": []}
            for cond_image_info in batch_cond_image_info:
                vit_kwargs["spatial_shapes"].append(
                    torch.stack([item.vision_encoder_kwargs["spatial_shapes"] for item in cond_image_info]))
                vit_kwargs["attention_mask"].append(
                    torch.stack([item.vision_encoder_kwargs["pixel_attention_mask"] for item in cond_image_info]))
            if cfg_factor[mode] > 1:
                vit_kwargs["spatial_shapes"] = vit_kwargs["spatial_shapes"] * cfg_factor[mode]
                vit_kwargs["attention_mask"] = vit_kwargs["attention_mask"] * cfg_factor[mode]
        else:
            cond_vae_images, cond_timestep, cond_vit_images = None, None, None
            vit_kwargs = None

        # 5. Build position embeddings
        rope_image_info = build_batch_rope_image_info(output, sections)
        if mode == "gen_text":
            seq_len = generation_config["max_length"]
        else:
            seq_len = output.tokens.shape[1]
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len,
            n_elem=config.attention_head_dim,
            device=device,
            base=config.rope_theta,
        )

        # 6. Build kv cache
        if bot_task == "img_ratio":
            max_new_tokens = 1
        if mode == "gen_image":
            # Image generation will not extend sequence length, using token length as max_cache_len is enough.
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + default(max_new_tokens, generation_config["max_length"])
        cache = HunyuanStaticCache(
            config=config,
            batch_size=batch_size * cfg_factor[mode],
            max_cache_len=max_cache_len,
            dtype=torch.bfloat16,
            dynamic=mode == "gen_text",
        )

        # 7. Build position ids
        batch_input_pos = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=device)[None].expand(
            batch_size * cfg_factor[mode], -1)  # use expand to share indices to save memory

        # 8. Build model input kwargs
        tkw = tkwrapper
        if image_size == "auto":
            extra_auto_stops = [tkw.special_token_map[f"<img_ratio_{i}>"] for i in range(33)]
        else:
            extra_auto_stops = [tkw.boi_token_id]
        stop_token_id = dict(
            auto=[tkw.eos_token_id] + extra_auto_stops,
            image=[tkw.eos_token_id],
            recaption=[tkw.end_recaption_token_id, tkw.end_answer_token_id, tkw.eos_token_id],
            think=[tkw.end_recaption_token_id, tkw.end_answer_token_id, tkw.eos_token_id],
            img_ratio=extra_auto_stops,
        )
        model_input_kwargs = dict(
            input_ids=output.tokens.to(device),
            position_ids=batch_input_pos,
            past_key_values=cache,
            custom_pos_emb=(cos, sin),
            mode=mode,
            image_mask=to_device(output.gen_image_mask, device),
            gen_timestep_scatter_index=to_device(output.gen_timestep_scatter_index, device),
            cond_vae_images=to_device(cond_vae_images, device),
            cond_timestep=to_device(cond_timestep, device),
            cond_vae_image_mask=to_device(output.cond_vae_image_mask, device),
            cond_vit_images=to_device(cond_vit_images, device),
            cond_vit_image_mask=to_device(output.cond_vit_image_mask, device),
            vit_kwargs={
                k: to_device(v, device) for k, v in vit_kwargs.items()
            } if vit_kwargs is not None else None,
            cond_timestep_scatter_index=to_device(output.cond_timestep_scatter_index, device),
            # for inner usage
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            generator=generator,
            # generation config
            eos_token_id=stop_token_id[bot_task],
            max_new_tokens=max_new_tokens,
        )

        from transformers import TextStreamer
        streamer = TextStreamer(tkwrapper.tokenizer, skip_prompt=True, skip_special_tokens=False)
        return model_input_kwargs, streamer

def prepare_requests(prompt: str | list[str], **kwargs):
    field_names = {f.name for f in fields(OmniDiffusionRequest)}

    init_kwargs = {"prompt": prompt}

    for key, value in kwargs.items():
        if key in field_names:
            init_kwargs[key] = value

    return OmniDiffusionRequest(**init_kwargs)

def get_hunyuan_image_3_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for Hunyuanimage3Pipeline."""
    #model_name = od_config.model
    #if os.path.exists(model_name):
    #    model_path = model_name
    #else:
    #    model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    #vae_config_path = os.path.join(model_path, "vae/config.json")

    def pre_process_func(
        requests: list[OmniDiffusionRequest],
        prompt: str | list[str],
    ):
        """Pre-process requests for Hunyuanimage3Pipeline."""
        prompts = []
        if isinstance(prompt, str):
            prompts.append(prompt)
        elif isinstance(prompt, list):
            prompts.extend(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings")


        # Check if request_id is provided in kwargs
        request_id = None
        # prepare parameters for Hunyuanimage3 
        device = get_local_device()
        #(height, width) set default=(1024,1024),  tokenizer set default
        model_kwargs, streamer_set = prepare_model_inputs(
            prompt=prompt, cot_text=None, system_prompt=None, mode="gen_image", 
            image_size=(1024, 1024), device=device, tokenizer = "/data1/s00957182/hy/weights/HunyuanImage-3.0"
        )
        model_kwargs["verbose"] = 0 # set default
        for i, p in enumerate(prompts):
            req_kwargs = {}
            if request_id is None:
                # Generate default ID consistent with OmniLLM: "{i}_{uuid}"
                req_kwargs["request_id"] = f"{i}"
                request =  prepare_requests(
                    p,
                    **req_kwargs,
                )
                request.extra.update(model_kwargs)
            requests.append(request)
        return requests

    return pre_process_func


# add for self
#self._guidance_scale = req.guidance_scale
#self._guidance_rescale = req.guidance_rescale
#self.do_classifier_free_guidance
def self_do_classifier_free_guidance():
    do_classifier_free_guidance = False
    if guidance_scale > 1.0:
        do_classifier_free_guidance = True
    return do_classifier_free_guidance
#self.scheduler no change
#self._execution_device device
#self.prepare_latents
#latents None batch_gen_image_info: List[ImageInfo] = kwargs.get("batch_gen_image_info") batch_size=len(batch_gen_image_info)
"""
latents = prepare_latents(
    batch_size=batch_size,
    latent_channel=32,
    image_size=(height, width),
    dtype=torch.bfloat16,
    device=device,
    generator=generator,
    latents=latents,
    )
"""
def self_prepare_latents(batch_size, latent_channel, image_size, dtype, device, generator, latents=None, config = HunyuanImage3Config(**HunyuanImage3Config_dict), generation_config=generation_config):
        scheduler = FlowMatchDiscreteScheduler( shift=generation_config['flow_shift'], reverse=True, solver="euler",)
        latent_scale_factor = config.vae_downsample_factor
        if latent_scale_factor is None:
            latent_scale_factor = (1,) * len(image_size)
        elif isinstance(latent_scale_factor, int):
            latent_scale_factor = (latent_scale_factor,) * len(image_size)
        elif isinstance(latent_scale_factor, tuple) or isinstance(latent_scale_factor, list):
            assert len(latent_scale_factor) == len(image_size), \
                "len(latent_scale_factor) shoudl be the same as len(image_size)"
            latent_scale_factor = latent_scale_factor
        else:
            raise ValueError(
                f"latent_scale_factor should be either None, int, tuple of int, or list of int, "
                f"but got {latent_scale_factor}"
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
        if hasattr(scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * scheduler.init_noise_sigma

        return latents

#timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, 1)
#self._num_timesteps = len(timesteps)
#self.prepare_extra_func_kwargs( self.scheduler.step, {"generator": generator})
# _scheduler_step_extra_kwargs = prepare_extra_func_kwargs( self.scheduler.step, {"generator": generator})
def self_prepare_extra_func_kwargs(func, kwargs):
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
#attention_mask = self.model._prepare_attention_mask_for_generation(input_ids, model_kwargs=model_kwargs,)
def self_prepare_attention_mask_for_generation(
            inputs_tensor: torch.Tensor,
            model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        # create `4d` bool attention mask (b, 1, seqlen, seqlen) using this implementation to bypass the 2d requirement
        # in the `transformers.generation_utils.GenerationMixin.generate`.
        # This implementation can handle sequences with text and image modalities, where text tokens use causal
        # attention and image tokens use full attention.
        bsz, seq_len = inputs_tensor.shape #input_ids
        tokenizer_output = model_kwargs["tokenizer_output"]
        batch_image_slices = [
            tokenizer_output.joint_image_slices[i] + tokenizer_output.gen_image_slices[i]
            for i in range(bsz)
        ]
        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for j, image_slice in enumerate(batch_image_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask
#self.progress_bar(total=num_inference_steps)
from tqdm.auto import tqdm
@torch.compiler.disable
def self_progress_bar(self, iterable=None, total=None):
    if not hasattr(self, "_progress_bar_config"):
        self._progress_bar_config = {}
    elif not isinstance(self._progress_bar_config, dict):
        raise ValueError(
            f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
        )

    if iterable is not None:
        return tqdm(iterable, **self._progress_bar_config)
    elif total is not None:
        return tqdm(total=total, **self._progress_bar_config)
    else:
        raise ValueError("Either `total` or `iterable` has to be defined.")

#model_inputs = prepare_inputs_for_generation(input_ids,images=latent_model_input,timestep=t_expand,**model_kwargs,)
def self_prepare_inputs_for_generation(
        input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            tokenizer_output=None, batch_gen_image_info=None, generator=None, **kwargs
    ):
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            if input_ids.shape[1] != kwargs["position_ids"].shape[1]:    # in decode steps
                input_ids = torch.gather(input_ids, dim=1, index=kwargs["position_ids"])
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": kwargs["position_ids"],
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "custom_pos_emb": kwargs["custom_pos_emb"],
                "mode": kwargs["mode"],
                "images": kwargs.get("images"),
                "image_mask": kwargs.get("image_mask"),
                "timestep": kwargs.get("timestep"),
                "gen_timestep_scatter_index": kwargs.get("gen_timestep_scatter_index"),
                "cond_vae_images": kwargs.get("cond_vae_images"),
                "cond_timestep": kwargs.get("cond_timestep"),
                "cond_vae_image_mask": kwargs.get("cond_vae_image_mask"),
                "cond_vit_images": kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": kwargs.get("cond_vit_image_mask"),
                "vit_kwargs": kwargs.get("vit_kwargs"),
                "cond_timestep_scatter_index": kwargs.get("cond_timestep_scatter_index"),
            }
        )
        return model_inputs
#self.model(**model_inputs, first_step=(i == 0))
#self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i) self.cfg_operator = ClassifierFreeGuidance()
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
#self.model._update_model_kwargs_for_generation(model_output,model_kwargs,)
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
)
def self_update_model_kwargs_for_generation(
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        mode = model_kwargs["mode"]

        updated_model_kwargs = {
            "mode": mode,
            "custom_pos_emb": model_kwargs["custom_pos_emb"],
        }

        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                updated_model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "tokenizer_output" in model_kwargs:
            if mode == "gen_text":
                # When enable batching, we use right padding, which requires a real_pos to index the valid
                # end position of the sequence. If tokenizer_output in model_kwargs, it means we are in the
                # prefill step of generation.
                #real_pos = to_device(model_kwargs["tokenizer_output"].real_pos, self.device)
                #updated_model_kwargs["position_ids"] = real_pos
                pass
            else:
                # position ids
                image_mask = model_kwargs["image_mask"]
                bsz, seq_len = image_mask.shape
                index = torch.arange(seq_len, device=image_mask.device).unsqueeze(0).repeat(bsz, 1)
                position_ids = index.masked_select(image_mask.bool()).reshape(bsz, -1)
                timestep_position_ids = \
                    index[torch.arange(bsz), model_kwargs["gen_timestep_scatter_index"][:, -1]].unsqueeze(-1)
                updated_model_kwargs["position_ids"] = torch.cat([timestep_position_ids, position_ids], dim=1)

                # attention mask
                mask_list = []
                for attention_mask_i, position_ids_i in zip(
                        model_kwargs["attention_mask"], updated_model_kwargs["position_ids"]):
                    mask_list.append(torch.index_select(attention_mask_i, dim=1, index=position_ids_i.reshape(-1)))
                attention_mask = torch.stack(mask_list, dim=0)
                updated_model_kwargs["attention_mask"] = attention_mask
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs["gen_timestep_scatter_index"]

        else:
            if mode == "gen_text":
                # Now we are in the decode steps.
                #updated_model_kwargs["position_ids"] = model_kwargs["position_ids"] + 1
                pass
            else:
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs["gen_timestep_scatter_index"]

        return updated_model_kwargs
#self.vae
#tmpconfig = HunyuanImage3Config(**HunyuanImage3Config_dict)
#vae = AutoencoderKLConv3D.from_config(tmpconfig.vae)