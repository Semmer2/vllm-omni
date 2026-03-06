# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HunyuanFusedMoE (Support HunyuanImage3 Diffusion Model, 5a779b4)."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestHunyuanFusedMoEPlatformDispatch:
    """Test platform dispatch and NotImplementedError for unknown platform."""

    def test_unknown_platform_raises_not_implemented_error(self, mocker):
        """HunyuanFusedMoE should raise NotImplementedError when platform is not NPU or CUDA."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        # Clear cached impl so _get_impl_class() is re-run with mocked platform
        hunyuan_moe._impl_class = None

        mock_platform = mocker.MagicMock()
        mock_platform.is_npu.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.__repr__ = lambda self: "UnknownPlatform"

        mocker.patch.object(
            hunyuan_moe,
            "current_omni_platform",
            mock_platform,
        )

        from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe import (
            HunyuanFusedMoE,
        )

        with pytest.raises(NotImplementedError) as exc_info:
            HunyuanFusedMoE(prefix="")

        assert "HunyuanFusedMoE is not implemented" in str(exc_info.value)
        assert "current_omni_platform" in str(exc_info.value)


class TestHunyuanFusedMoEFactory:
    """Test HunyuanFusedMoE factory __new__ and make_expert_params_mapping delegation."""

    def test_new_delegates_to_impl_class(self, mocker):
        """HunyuanFusedMoE(prefix=..., **kwargs) should instantiate and return impl instance."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        hunyuan_moe._impl_class = None

        # Use a simple mock class as impl so we don't need CUDA/NPU
        class MockImpl:
            def __init__(self, *, prefix: str = "", **kwargs):
                self.prefix = prefix
                self.kwargs = kwargs

        mock_impl_class = mocker.MagicMock(return_value=MockImpl(prefix="test", a=1))
        mocker.patch.object(
            hunyuan_moe,
            "_get_impl_class",
            return_value=mock_impl_class,
        )

        from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe import (
            HunyuanFusedMoE,
        )

        result = HunyuanFusedMoE(prefix="test", a=1)

        assert isinstance(result, MockImpl)
        assert result.prefix == "test"
        assert result.kwargs == {"a": 1}
        mock_impl_class.assert_called_once_with(prefix="test", a=1)

    def test_make_expert_params_mapping_delegates_to_impl(self, mocker):
        """make_expert_params_mapping should delegate to impl class method."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        expected_mapping = [("a", "b", 0, "c")]
        mock_impl_class = mocker.MagicMock()
        mock_impl_class.make_expert_params_mapping = mocker.MagicMock(return_value=expected_mapping)
        mocker.patch.object(
            hunyuan_moe,
            "_get_impl_class",
            return_value=mock_impl_class,
        )

        from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe import (
            HunyuanFusedMoE,
        )

        result = HunyuanFusedMoE.make_expert_params_mapping(
            model=None,
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=4,
            num_redundant_experts=0,
        )

        assert result == expected_mapping
        mock_impl_class.make_expert_params_mapping.assert_called_once_with(
            None,
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=4,
            num_redundant_experts=0,
        )
