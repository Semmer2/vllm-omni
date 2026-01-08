import logging
from collections.abc import Iterable
import torch
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.utils import AutoWeightsLoader

from .hunyuan_image_3_models import HunyuanImage3Model
from .autoencoder_kl_3d import AutoencoderKLConv3D

logger = logging.getLogger(__name__)

class HunyuanImage3Pipeline(nn.Module):
    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config
        self.hf_config = get_config(od_config.model, trust_remote_code=True)
        self.model = HunyuanImage3Model(self.hf_config)
        self.vae = AutoencoderKLConv3D.from_config(self.hf_config.vae)
        self._tkwrappper = None
        self.image_processor = None
        self._pipeline = None
        self.vision_model = None
        self.vision_aligner = None
        self.timestep_emb = None
        self.patch_embed = None
        self.time_embed = None
        self.final_layer = None
        self.time_embed2 = None
        self.lm_head = None

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

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] = "",
        negative_prompt: str | list[str] = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        seed: int | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> DiffusionOutput:

        prompt = req.prompt if req.prompt is not None else prompt
        negative_prompt = req.negative_prompt if req.negative_prompt is not None else negative_prompt


        # 1. check inputs
        # 2. encode prompts
        # 3. prepare latents and timesteps
        # 4. diffusion process
        # 5. decode latents
        # 6. post-process outputs
        
        return DiffusionOutput(output=None)