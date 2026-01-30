import logging
from typing import List, Optional, Callable, Dict, Any, Union, Tuple, TYPE_CHECKING
from collections.abc import Iterable
import torch
import torch.nn as nn

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerationConfig, GenerationMixin, ALL_CACHE_NAMES
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.integrations.accelerate import init_empty_weights
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.config.vllm import get_current_vllm_config

from .hunyuan_image_3_models import (
    HunyuanImage3Model,
    HunyuanImage3PreTrainedModel,
    FlowMatchDiscreteScheduler,
    HunyuanImage3Text2ImagePipeline,
    TimestepEmbedder,
    UNetDown,
    UNetUp,
    build_batch_2d_rope,
    real_batched_index_select,
    HunyuanStaticCache,
    CausalMMOutputWithPast,
)
from .autoencoder_kl_3d import AutoencoderKLConv3D
from .image_processor import HunyuanImage3ImageProcessor
from .siglip2 import Siglip2VisionTransformer, LightProjector
from .tokenizer_wrapper import TokenizerWrapper, ImageInfo

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.getLogger(__name__)

BatchRaggedImages = Union[torch.Tensor, List[Union[torch.Tensor, List[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, List[torch.Tensor]]

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


class HunyuanImage3Pipeline(HunyuanImage3PreTrainedModel, GenerationMixin):
    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        self.hf_config = get_config(od_config.model, trust_remote_code=True)
        super().__init__(self.hf_config)
        # update diffusion config
        self.generation_config = GenerationConfig.from_pretrained(od_config.model)
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=True,
            )
        ]
        # self.model = None
        # model_init_context = self.get_init_context(False, False)
        # init_empty_weights can accelerate weights loading, but also can cause monkey patch error
        # with init_empty_weights():
        self.model = HunyuanImage3Model(self.hf_config)
        self.vae = AutoencoderKLConv3D.from_config(self.hf_config.vae)
        self._pipeline = None
        self._tkwrapper = TokenizerWrapper(od_config.model)
        self.image_processor = HunyuanImage3ImageProcessor(self.hf_config)
        self.vision_model = Siglip2VisionTransformer(self.hf_config.vit)
        self.vision_aligner = LightProjector(self.hf_config.vit_aligner)
        self.timestep_emb = TimestepEmbedder(hidden_size=self.hf_config.hidden_size)
        if self.hf_config.img_proj_type != "unet":
            raise ValueError(f"Unknown img_proj_type: {self.hf_config.img_proj_type}")

        self.patch_embed = UNetDown(
            patch_size=self.hf_config.patch_size,
            emb_channels=self.hf_config.hidden_size,
            in_channels=self.hf_config.vae["latent_channels"],
            hidden_channels=self.hf_config.patch_embed_hidden_dim,
            out_channels=self.hf_config.hidden_size,
        )
        self.time_embed = TimestepEmbedder(hidden_size=self.hf_config.hidden_size)
        self.final_layer = UNetUp(
            patch_size=self.hf_config.patch_size,
            emb_channels=self.hf_config.hidden_size,
            in_channels=self.hf_config.hidden_size,
            hidden_channels=self.hf_config.patch_embed_hidden_dim,
            out_channels=self.hf_config.vae["latent_channels"],
            out_norm=True,
        )
        self.time_embed_2 = TimestepEmbedder(hidden_size=self.hf_config.hidden_size)
        self.lm_head = nn.Linear(self.hf_config.hidden_size, self.hf_config.vocab_size, bias=False)
        self.vllm_config = get_current_vllm_config()
        self.post_init()

    @staticmethod
    def _create_selective_device_map():
        device_map = {}

        # call vllm transformer_blocks, filter hf weights loading.
        device_map["model.layers"] = "cpu"

        # others keep on cuda:0
        device_map["vae"] = "cuda:0"
        device_map["vision_model"] = "cuda:0"
        device_map["vision_aligner"] = "cuda:0"
        device_map["timestep_emb"] = "cuda:0"
        device_map["patch_embed"] = "cuda:0"
        device_map["time_embed"] = "cuda:0"
        device_map["time_embed_2"] = "cuda:0"
        device_map["final_layer.model"] = "cuda:0"
        device_map["model.wte"] = "cuda:0"
        device_map["model.ln_f"] = "cuda:0"
        device_map["lm_head"] = "cuda:0"

        return device_map

    @classmethod
    def load_multi_moda(cls, path):
        device_map = cls._create_selective_device_map()
        kwargs = dict(
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map=device_map,
            moe_impl="eager",
            low_cpu_mem_usage=True,
        )

        model = cls.from_pretrained(path, **kwargs)
        model.load_tokenizer(path)
        print("layer device_map:")
        print(model.hf_device_map)

        return model, model.tokenizer

    def pre_load(self):
        from .hunyuan_image3_utils import get_full_state_dict
        from vllm.distributed import get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()
        print(f"pre load rank:{tp_rank}")
        state_dict = get_full_state_dict(self.od_config.model)
        non_layer_prefixes = [
            "vae", "vision_model", "vision_aligner", "lm_head",
            "patch_embed", "timestep_emb", "model.wte", "model.ln_f",
            "time_embed", "time_embed_2", "final_layer.model"
        ]
        filtered_sd = {}
        for k, v in state_dict.items():
            if any(k.startswith(prefix) for prefix in non_layer_prefixes):
                filtered_sd[k] = v

        missing, unexpected = self.load_state_dict(filtered_sd, strict=False)

        for prefix in non_layer_prefixes:
            if hasattr(self, prefix.split('.')[0]):
                module = dict(self.named_modules()).get(prefix)
                if module:
                    module.to(f"cuda:{tp_rank}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.pre_load()
        skip_prefixes = ["lm_head."] if self.hf_config.tie_word_embeddings else []
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
        skip_prefixes.extend(unexpected_keywords)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights)

    def prepare_seed(self, seed=None, batch_size=1):
        #random seed
        if seed is not None:
            return [seed + i for i in range(batch_size)]
        else:
            import random
            return [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
        
    @property
    def pipeline(self):
        if self._pipeline is None:
            # shift hard code
            self.scheduler = FlowMatchDiscreteScheduler(
                shift=self.generation_config.flow_shift, reverse=True, solver="euler",
            )
            self._pipeline = HunyuanImage3Text2ImagePipeline(
                model=self, scheduler=self.scheduler, vae=self.vae
            )
        return self._pipeline

    @staticmethod
    def get_pos_emb(custom_pos_emb, position_ids):
        cos, sin = custom_pos_emb
        cos = real_batched_index_select(cos, dim=1, idx=position_ids)
        sin = real_batched_index_select(sin, dim=1, idx=position_ids)
        return cos, sin

    def instantiate_vae_image_tokens(
            self,
            x: torch.Tensor,
            images: BatchRaggedImages,
            ts: BatchRaggedTensor,
            image_mask: torch.Tensor,
    ):
        """
        Instantiate the VAE image embeddings into the input embedding sequence.

        Args:
            x: input sequence, (batch_size, seq_len, n_embd)
            images: BatchRaggedImages
                images can be a 4-D tensor, or a list of 4-D tensors, or a list of lists of 3-D tensors.
            ts: BatchRaggedTensor
                ts can be a 1-D tensor, or a list of 1-D tensors
            image_mask: (batch_size, seq_len)
        """
        batch_size, seq_len, n_embd = x.shape

        if isinstance(images, list):
            index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            t_emb = []
            for i, (image_i, t_i) in enumerate(zip(images, ts)):
                if isinstance(image_i, torch.Tensor):
                    # time_embed needs a 1-D tensor as input
                    t_i_emb = self.time_embed(t_i)
                    # n_{i} x one_image_seq_len x n_embd
                    image_i_seq, _, _ = self.patch_embed(image_i, t_i_emb)
                    # 1 x (n_{i} * one_image_seq_len)
                    image_i_scatter_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                    x[i:i + 1].scatter_(
                        dim=1,
                        index=image_i_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                        # 1 x (n_{i} * one_image_seq_len) x n_embd
                        src=image_i_seq.reshape(1, -1, n_embd),  # 1 x (n_{i} * one_image_seq_len) x n_embd
                    )
                    t_emb.append(t_i_emb)
                elif isinstance(image_i, list):
                    # time_embed needs a 1-D tensor as input
                    t_i_emb = self.time_embed(t_i)  # n_{i} x d
                    image_i_seq_list = [], []
                    for j in range(len(image_i)):
                        image_ij = image_i[j]
                        if image_ij.dim() == 4:
                            assert image_i[j].shape[0] == 1, "image_i[j] should have a batch dimension of 1"
                        elif image_ij.dim() == 3:
                            image_ij = image_ij.unsqueeze(0)
                        else:
                            raise ValueError(f"image_i[j] should have 3 or 4 dimensions, got {image_ij.dim()}")
                        # 1 x one_image_seq_len_{j} x n_embd
                        image_i_seq_j, _, _ = self.patch_embed(image_ij, t_i_emb[j:j + 1])
                        image_i_seq_list.append(image_i_seq_j)
                    # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                    image_i_seq = torch.cat(image_i_seq_list, dim=1)
                    # 1 x sum_{j}(one_image_seq_len_{j})
                    image_i_scatter_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                    x[i:i + 1].scatter_(
                        dim=1,
                        index=image_i_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                        # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                        src=image_i_seq.reshape(1, -1, n_embd),  # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                    )
                    t_emb.append(t_i_emb)
                else:
                    raise TypeError(f"image_i should be a torch.Tensor or a list, got {type(image_i)}")
            token_h, token_w = None, None
        else:
            # images is a 4-D tensor
            batch_size, seq_len, n_embd = x.shape
            index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            t_emb = self.time_embed(ts)
            image_seq, token_h, token_w = self.patch_embed(images, t_emb)
            image_scatter_index = index.masked_select(image_mask.bool()).reshape(batch_size, -1)
            x.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq,
            )

        return x, token_h, token_w

    def instantiate_timestep_tokens(
            self,
            x: torch.Tensor,
            t: BatchRaggedTensor,
            timestep_scatter_index: BatchRaggedTensor,
    ):
        batch_size, seq_len, n_embd = x.shape
        # batch_size x n x n_embd
        timestep_scatter_src = self.timestep_emb(t.reshape(-1)).reshape(batch_size, -1, n_embd)
        x.scatter_(
            dim=1,
            index=timestep_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
            src=timestep_scatter_src,
        )

        return x

    def instantiate_vit_image_tokens(
            self,
            x: torch.Tensor,
            cond_vit_images: Union[torch.Tensor, List[torch.Tensor]],
            cond_vit_image_mask: torch.Tensor,
            vit_kwargs: Dict[str, Any],
    ):
        # 1. Forward the vit encoder and vit aligner to get the vit image embeddings and align them to the
        # transformer hidden size
        cond_vit_image_embeds = []
        for batch_idx, image in enumerate(cond_vit_images):
            cur_kwargs = {k: v[batch_idx] for k, v in vit_kwargs.items()}
            image_embed = self.vision_model(image, **cur_kwargs).last_hidden_state
            image_embed = self.vision_aligner(image_embed)
            n, seq_len, dim = image_embed.shape
            image_embed = image_embed.reshape(n * seq_len, dim)
            cond_vit_image_embeds.append(image_embed)

        # 2. Instantiate the vit image embeddings into the input sequence
        batch_size, seq_len, n_embd = x.shape
        index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        for i, (image_embed, mask) in enumerate(zip(cond_vit_image_embeds, cond_vit_image_mask)):
            image_scatter_index = index[i:i+1].masked_select(mask.bool()).reshape(1, -1)
            x[i:i+1].scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_embed.reshape(1, -1, n_embd),
            )

        return x

    def ragged_final_layer(self, x, image_mask, timestep, token_h, token_w, first_step):
        bsz, seq_len, n_embd = x.shape
        if first_step:
            image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(bsz, -1, n_embd)
        else:
            image_output = x[:, 1:, :]
        timestep_emb = self.time_embed_2(timestep)
        pred = self.final_layer(image_output, timestep_emb, token_h, token_w)
        return pred

    @staticmethod
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

    @staticmethod
    def check_inputs(prompt=None, message_list=None):
        if prompt is None and message_list is None:
            raise ValueError("Either `prompt` or `message_list` should be provided.")
        if prompt is not None and message_list is not None:
            raise ValueError("Only one of `prompt` or `message_list` should be provided.")
        if prompt is not None:
            assert isinstance(prompt, str) or isinstance(prompt, list), \
                f"`prompt` should be a string or a list of strings, but got {type(prompt)}."
            if isinstance(prompt, list):
                assert len(prompt) > 0 and all(isinstance(p, str) for p in prompt), \
                    "`prompt` should be a non-empty list of strings."
        if message_list is not None:
            if not isinstance(message_list, list):
                raise ValueError(f"`message_list` should be a list of messages, but got {type(message_list)}.")
            assert len(message_list) > 0, "`message_list` should be a non-empty list."
            for message in message_list:
                assert isinstance(message, list) or isinstance(message, dict), \
                    f"Each message should be a list of dicts or a dict, but got {type(message)}."

    def prepare_model_inputs(
            self,
            prompt=None,
            mode="gen_image",
            system_prompt=None,
            cot_text=None,
            image_size="auto",
            message_list=None,
            device=None,
            max_new_tokens=None,
            **kwargs,
    ):
        # 1. Sanity check
        self.check_inputs(prompt, message_list)
        device = default(device, self.device)

        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt
        batch_cot_text = cot_text
        batch_system_prompt = system_prompt
        batch_gen_image_info = None
        # TODO: construct with user input images
        batch_cond_image_info = None

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
                batch_gen_image_info = [self.image_processor.build_image_info(image_size) for _ in range(batch_size)]

        #   -- 2.3 seed
        generator = kwargs.get("generator", None)
        if generator is None:
            seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
            generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # 3. apply chat template
        cfg_factor = {"gen_text": 1, "gen_image": 2}
        bot_task = kwargs.pop("bot_task", "auto")
        # If `drop_think` enabled, always drop <think> parts in the context.
        drop_think = kwargs.get('drop_think', self.generation_config.drop_think)
        # Apply batched prompt or batched message_list to build input sequence with associated info.
        out = self._tkwrapper.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_image_info=batch_cond_image_info,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=kwargs.get('max_length'),
            bot_task=bot_task,
            image_base_size=self.config.image_base_size,
            sequence_template="pretrain",
            cfg_factor=cfg_factor[mode],
            drop_think=drop_think,
        )
        output, sections = out['output'], out['sections']

        # 4. Encode conditional images
        if batch_cond_image_info is not None and len(batch_cond_image_info[0]) > 0:
            cond_vae_images, cond_timestep, cond_vit_images = self._encode_cond_image(
                batch_cond_image_info, cfg_factor[mode]
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
        rope_image_info = self.build_batch_rope_image_info(output, sections)
        if mode == "gen_text":
            seq_len = self.generation_config.max_length
        else:
            seq_len = output.tokens.shape[1]
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len,
            n_elem=self.config.attention_head_dim,
            device=device,
            base=self.config.rope_theta,
        )

        # 6. Build kv cache
        if bot_task == "img_ratio":
            max_new_tokens = 1
        if mode == "gen_image":
            # Image generation will not extend sequence length, using token length as max_cache_len is enough.
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + default(max_new_tokens, self.generation_config.max_length)

        cache = HunyuanStaticCache(
            config=self.config,
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
        tkw = self._tkwrapper
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
            past_key_values=None,
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
                k: to_device(v, self.device) for k, v in vit_kwargs.items()
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

        return model_input_kwargs

    def _prepare_attention_mask_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            generation_config: GenerationConfig,
            model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        # create `4d` bool attention mask (b, 1, seqlen, seqlen) using this implementation to bypass the 2d requirement
        # in the `transformers.generation_utils.GenerationMixin.generate`.
        # This implementation can handle sequences with text and image modalities, where text tokens use causal
        # attention and image tokens use full attention.
        bsz, seq_len = inputs_tensor.shape
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
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
                "query_lens": kwargs.get("query_lens"),
                "seq_lens": kwargs.get("seq_lens"),
                "num_image_tokens": kwargs.get("num_image_tokens"),
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
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
                real_pos = to_device(model_kwargs["tokenizer_output"].real_pos, self.device)
                updated_model_kwargs["position_ids"] = real_pos
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
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"] + 1
            else:
                updated_model_kwargs["position_ids"] = model_kwargs["position_ids"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]
                updated_model_kwargs["gen_timestep_scatter_index"] = model_kwargs["gen_timestep_scatter_index"]

        return updated_model_kwargs

    def _generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            use_model_defaults: Optional[bool] = None,
            generator: Optional[List[torch.Generator]] = None,
            verbose: int = 0,
            **kwargs,
    ):
        mode = kwargs.get("mode", "gen_text")

        # verbose > 1 not support

        if mode == "gen_text":
            raise NotImplementedError("Not support gen text for hunyuan image")

        elif mode == "gen_image":
            batch_gen_image_info: List[ImageInfo] = kwargs.get("batch_gen_image_info")
            if batch_gen_image_info is None:
                raise ValueError("`batch_gen_image_info` should be provided when `mode` is `gen_image`.")

            image_info:ImageInfo = batch_gen_image_info[0]
            num_image_tokens = (
                image_info.image_token_length
                + (1 if image_info.add_timestep_token else 0)
                + (1 if image_info.add_guidance_token else 0)
            )
            kwargs["num_image_tokens"] = num_image_tokens
            # 50 and 5.0 hard code
            from vllm.forward_context import set_forward_context
            with set_forward_context(None, self.vllm_config):
                results = self.pipeline(
                    batch_size=len(batch_gen_image_info),
                    image_size=[batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width],
                    num_inference_steps=kwargs.get("diff_infer_steps", 50),
                    guidance_scale=kwargs.get("diff_guidance_scale", 5.0),
                    generator=generator,
                    model_kwargs=kwargs,
                )
            samples = results[0]
            return samples

        else:
            raise ValueError(f"Unknown mode {mode}, only `gen_text` and `gen_image` are supported.")

    @staticmethod
    def _check_inputs(cond, target, check_list):
        if cond:
            for name, item in check_list:
                assert item is not None, f"`{name}` should be provided when `{target}`."

    def forward_call(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            mode: str = "gen_text",
            first_step: Optional[bool] = None,
            # for gen image
            images: Optional[BatchRaggedImages] = None,
            image_mask: Optional[torch.Tensor] = None,
            timestep: Optional[BatchRaggedTensor] = None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
            # for cond image
            cond_vae_images: Optional[BatchRaggedImages] = None,
            cond_timestep: Optional[BatchRaggedTensor] = None,
            cond_vae_image_mask: Optional[torch.Tensor] = None,
            cond_vit_images: Optional[BatchRaggedImages] = None,
            cond_vit_image_mask: Optional[torch.Tensor] = None,
            vit_kwargs: Optional[Dict[str, Any]] = None,
            cond_timestep_scatter_index: Optional[torch.Tensor] = None,
            query_lens: Optional[list[int]] = None,
            seq_lens: Optional[list[int]] = None,
            num_image_tokens: Optional[int] = None,
    ) -> Union[Tuple, CausalMMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Sanity Check of Inputs
        self._check_inputs(mode == "gen_image", "in `gen_image` mode", [
            ("images", images), ("timestep", timestep), ("gen_timestep_scatter_index", gen_timestep_scatter_index),
        ])
        self._check_inputs(mode == "gen_image" and first_step, "in `gen_image` mode at the first step", [
            ("image_mask", image_mask),
        ])
        self._check_inputs(cond_vae_images is not None, "`cond_vae_images` is provided", [
            ("cond_timestep", cond_timestep), ("cond_vae_image_mask", cond_vae_image_mask),
            ("cond_timestep_scatter_index", cond_timestep_scatter_index),
        ])
        self._check_inputs(cond_vit_images is not None, "`cond_vit_images` is provided", [
            ("cond_vit_image_mask", cond_vit_image_mask), ("vit_kwargs", vit_kwargs),
        ])


        custom_pos_emb = self.get_pos_emb(custom_pos_emb, position_ids)

        inputs_embeds = self.model.wte(input_ids)

        bsz, seq_len, n_embd = inputs_embeds.shape

        # Instantiate placeholder tokens: <timestep>, <img> for the gen image
        if mode == "gen_text":
            # For gen_text, make sure gen_timestep_scatter_index is None
            gen_timestep_scatter_index = None
            token_h, token_w = None, None
        else:
            if first_step:
                inputs_embeds, token_h, token_w = self.instantiate_vae_image_tokens(
                    inputs_embeds, images, timestep, image_mask)
                inputs_embeds = self.instantiate_timestep_tokens(
                    inputs_embeds, timestep, gen_timestep_scatter_index)
            else:
                t_emb = self.time_embed(timestep)
                image_emb, token_h, token_w = self.patch_embed(images, t_emb)
                timestep_emb = self.timestep_emb(timestep).reshape(bsz, -1, n_embd)
                inputs_embeds = torch.cat([timestep_emb, image_emb], dim=1)
        

        # Instantiate placeholder tokens: <timestep>, <img> for cond images
        # Should only run once with kv-cache enabled.
        if cond_vae_images is not None:
            inputs_embeds, _, _ = self.instantiate_vae_image_tokens(
                inputs_embeds, cond_vae_images, cond_timestep, cond_vae_image_mask)
            inputs_embeds = self.instantiate_timestep_tokens(
                inputs_embeds, cond_timestep, cond_timestep_scatter_index)
        if cond_vit_images is not None:
            inputs_embeds = self.instantiate_vit_image_tokens(
                inputs_embeds, cond_vit_images, cond_vit_image_mask, vit_kwargs)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            custom_pos_emb=custom_pos_emb,
            mode=mode,
            first_step=first_step,
            query_lens=query_lens,
            seq_lens=seq_lens,
            num_image_tokens=num_image_tokens,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
        )
        hidden_states = outputs[0]

        if mode == "gen_text":
            hidden_states = self.model.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            diffusion_prediction = None
        else:
            logits = None
            hidden_states = hidden_states.to(input_ids.device)
            assert hidden_states.numel() == bsz * seq_len * n_embd, \
                f"Shape mismatch: {hidden_states.shape} cannot reshape to ({bsz}, {seq_len}, {n_embd})"
            hidden_states = hidden_states.reshape(bsz, seq_len, n_embd)
            diffusion_prediction = self.ragged_final_layer(
                hidden_states, image_mask, timestep, token_h, token_w, first_step)

        if not return_dict:
            output = (logits,) + outputs[1:] + (diffusion_prediction,)
            return output

        output = CausalMMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            diffusion_prediction=diffusion_prediction,
        )

        return output

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] = "",
        negative_prompt: str | list[str] = "",
        image_size = "auto",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        seed: int | None = None,
        system_prompt: str | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]
        generator = req.sampling_params.generator or generator
        height = req.sampling_params.height or height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or width or self.default_sample_size * self.vae_scale_factor
        image_size = (height, width)
        model_inputs = self.prepare_model_inputs(
            prompt=prompt, cot_text=None, system_prompt=system_prompt,
            mode="gen_image", generator=generator, image_size=image_size
        )
        outputs = self._generate(**model_inputs, **kwargs)
        return DiffusionOutput(output=outputs[0])
