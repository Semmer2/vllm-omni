import logging
from collections.abc import Iterable
import torch
import torch.nn as nn

from transformers.generation.utils import GenerationConfig
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.utils import AutoWeightsLoader

from .hunyuan_image_3_models import (
    HunyuanImage3Model,
    HunyuanImage3PreTrainedModel,
    FlowMatchDiscreteScheduler,
    HunyuanImage3Text2ImagePipeline,
    TimestepEmbedder,
    UNetDown,
    UNetUp,
)
from .autoencoder_kl_3d import AutoencoderKLConv3D
from .image_processor import HunyuanImage3ImageProcessor
from .siglip2 import Siglip2VisionTransformer, LightProjector
from .tokenizer_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)

class HunyuanImage3Pipeline(HunyuanImage3PreTrainedModel, nn.Module):
    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        self.hf_config = get_config(od_config.model, trust_remote_code=True)
        super().__init__(self.hf_config)
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
        self.time_embed2 = TimestepEmbedder(hidden_size=self.hf_config.hidden_size)
        self.lm_head = nn.Linear(self.hf_config.hidden_size, self.hf_config.vocab_size, bias=False)
        self.post_init()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["lm_head."] if self.hf_config.tie_word_embeddings else []
        # List of unexpected keywords in weight names
        # unexpected_keywords = [
        #     "vae",
        #     "vision_aligner",
        #     "vision_model",
        #     "final_layer",
        #     "patch_embed",
        #     "timestep_emb",
        #     "time_embed",
        #     "time_embed_2",
        #     "guidance_emb",
        #     "timestep_r_emb",
        # ]
        # skip_prefixes.extend(unexpected_keywords)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights)

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.scheduler = FlowMatchDiscreteScheduler(
                shift=self.generation_config.flow_shift, reverse=True, solver="euler",
            )
            self._pipeline = HunyuanImage3Text2ImagePipeline(
                model=self, scheduler=self.scheduler, vae=self.vae
            )
        return self._pipeline

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
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> DiffusionOutput:

        prompt = req.prompt if req.prompt is not None else prompt
        negative_prompt = req.negative_prompt if req.negative_prompt is not None else negative_prompt

        # get image size
        if image_size == "auto":
            model_inputs = self.prepare_model_inputs(
                prompt=prompt, cot_text=None, bost_task="img_ratio",
                system_prompt=system_prompt, generator=generator
            )
            outputs = self._generate(**model_inputs, **kwargs)
            ratio_index = output[0, -1].item() - self._tkwrappper.ratio_token_offset


            if ratio_index < 0 or ratio_index >= len(self.image_processor.reso_group):
                ratio_index = 16
            reso = self.image_processor.reso_group[ratio_index]
            image_size = reso.height, reso.width

        model_inputs = self.prepare_model_inputs(
            prompt=prompt, cot_text=None, system_prompt=system_prompt,
            mode="gen_image", generator=generator, image_size=image_size
        )
        outputs = self._generate(**model_inputs, **kwargs)
        # 1. check inputs
        # 2. encode prompts
        # 3. prepare latents and timesteps
        # 4. diffusion process
        # 5. decode latents
        # 6. post-process outputs

        return DiffusionOutput(output=outputs[0])
