from __future__ import annotations

import unittest

import numpy as np
import torch

from vrc_ardy_agent.ardy_runtime import ArdyHeadlessRuntime


class _FakeMotionRep:
    fps = 20.0

    def unnormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion

    def inverse(self, motion: torch.Tensor, *, is_normalized: bool) -> dict[str, torch.Tensor]:
        del is_normalized
        batch, frames, _ = motion.shape
        positions = torch.zeros(batch, frames, 27, 3, dtype=motion.dtype, device=motion.device)
        rotations = torch.eye(3, dtype=motion.dtype, device=motion.device).view(1, 1, 1, 3, 3)
        rotations = rotations.repeat(batch, frames, 27, 1, 1)

        root_z = motion[..., 0]
        positions[..., 0, 1] = 1.1
        positions[..., 6, 1] = 2.0
        positions[..., 10, :] = torch.stack(
            [torch.full_like(root_z, -0.5), torch.full_like(root_z, 1.4), root_z],
            dim=-1,
        )
        positions[..., 16, :] = torch.stack(
            [torch.full_like(root_z, 0.5), torch.full_like(root_z, 1.4), root_z],
            dim=-1,
        )
        positions[..., 21, :] = torch.stack(
            [torch.full_like(root_z, -0.1), torch.full_like(root_z, 0.1), root_z],
            dim=-1,
        )
        positions[..., 25, :] = torch.stack(
            [torch.full_like(root_z, 0.1), torch.full_like(root_z, 0.1), root_z],
            dim=-1,
        )
        positions[..., 22, :] = torch.stack(
            [torch.full_like(root_z, -0.1), torch.zeros_like(root_z), root_z],
            dim=-1,
        )
        positions[..., 26, :] = torch.stack(
            [torch.full_like(root_z, 0.1), torch.zeros_like(root_z), root_z],
            dim=-1,
        )
        positions[..., 6, 2] = root_z
        positions[..., 0, 2] = root_z

        root_positions = torch.stack(
            [torch.zeros_like(root_z), torch.full_like(root_z, 1.1), root_z],
            dim=-1,
        )
        headings = torch.stack([torch.ones_like(root_z), torch.zeros_like(root_z)], dim=-1)
        contacts = torch.zeros(batch, frames, 4, dtype=motion.dtype, device=motion.device)
        return {
            "local_rot_mats": rotations,
            "global_rot_mats": rotations,
            "posed_joints": positions,
            "root_positions": root_positions,
            "smooth_root_pos": root_positions,
            "global_root_heading": headings,
            "foot_contacts": contacts,
        }


class _FakeDiffusion:
    num_base_steps = 10


class _FakeModel:
    gen_horizon_len = 4
    num_frames_per_token = 2
    motion_rep = _FakeMotionRep()
    diffusion = _FakeDiffusion()
    skeleton = object()

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.history_lengths: list[int] = []
        self.generated_until = 0

    def _encode_text(self, texts: list[str]):
        self.encode_calls.append(texts[0])
        feat = torch.tensor([[[float(len(texts[0]))]]], dtype=torch.float32)
        mask = torch.ones(1, 1, dtype=torch.bool)
        return feat, mask

    def autoregressive_step(
        self,
        *,
        num_frames: int,
        num_denoising_steps: int,
        motion_mask,
        observed_motion,
        cfg_weight,
        texts,
        text_feat,
        text_pad_mask,
        init_history_sequence: torch.Tensor | None,
        init_global_translation=None,
        init_first_heading_angle=None,
    ) -> torch.Tensor:
        del (
            num_denoising_steps,
            motion_mask,
            observed_motion,
            cfg_weight,
            texts,
            text_feat,
            text_pad_mask,
            init_global_translation,
            init_first_heading_angle,
        )
        history_len = 0 if init_history_sequence is None else init_history_sequence.shape[1]
        self.history_lengths.append(history_len)
        if num_frames != history_len + self.gen_horizon_len:
            raise AssertionError((num_frames, history_len, self.gen_horizon_len))

        new = torch.arange(
            self.generated_until + 1,
            self.generated_until + 1 + self.gen_horizon_len,
            dtype=torch.float32,
        ).reshape(1, self.gen_horizon_len, 1)
        self.generated_until += self.gen_horizon_len
        if init_history_sequence is None:
            return new
        return torch.cat([init_history_sequence, new], dim=1)


class ArdyHeadlessRuntimeTests(unittest.TestCase):
    def test_prompt_embeddings_are_cached_across_prompt_switches(self):
        model = _FakeModel()
        runtime = ArdyHeadlessRuntime.from_model(model, history_limit_frames=8, postprocess=False)

        runtime.set_prompt("walk")
        runtime.set_prompt("run")
        runtime.set_prompt("walk")

        self.assertEqual(model.encode_calls, ["walk", "run"])
        self.assertEqual(runtime.prompt, "walk")

    def test_rollouts_keep_a_bounded_native_history_and_return_only_new_horizon(self):
        model = _FakeModel()
        runtime = ArdyHeadlessRuntime.from_model(model, history_limit_frames=8, postprocess=False)
        runtime.set_prompt("walk")

        first = runtime.generate_next()
        second = runtime.generate_next()
        third = runtime.generate_next()

        self.assertEqual(model.history_lengths, [0, 4, 8])
        self.assertEqual(first.frame_count, 4)
        self.assertEqual(second.frame_count, 4)
        self.assertEqual(third.frame_count, 4)
        np.testing.assert_allclose(first.root_positions[:, 2], [1, 2, 3, 4])
        np.testing.assert_allclose(second.root_positions[:, 2], [5, 6, 7, 8])
        np.testing.assert_allclose(third.root_positions[:, 2], [9, 10, 11, 12])
        self.assertEqual(runtime.history_frame_count, 8)

    def test_generate_requires_a_prompt(self):
        runtime = ArdyHeadlessRuntime.from_model(_FakeModel(), history_limit_frames=8, postprocess=False)

        with self.assertRaisesRegex(RuntimeError, "prompt"):
            runtime.generate_next()


if __name__ == "__main__":
    unittest.main()