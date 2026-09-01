from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArdyMotionChunk:
    """Bridge-ready output for one newly generated ARDY horizon."""

    posed_joints: np.ndarray
    global_rot_mats: np.ndarray
    root_positions: np.ndarray
    smooth_root_pos: np.ndarray
    global_root_heading: np.ndarray
    foot_contacts: np.ndarray
    fps: float
    prompt: str
    generation_seconds: float

    @property
    def frame_count(self) -> int:
        return int(self.posed_joints.shape[0])


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return numpy()
    return np.asarray(value)


class ArdyHeadlessRuntime:
    """Resident, GUI-free wrapper around ARDY's official autoregressive API.

    The ARDY interactive demo keeps a normalized native motion tensor as the
    generation history and asks ``autoregressive_step`` for one more horizon.
    This class preserves that exact separation: native motion stays internal
    for future generation, while decoded joint/root arrays are returned to the
    VRChat bridge. Tracker-space coordinates are never fed back into ARDY.
    """

    def __init__(
        self,
        model: Any,
        *,
        history_limit_frames: int,
        postprocess: bool = True,
        postprocess_fn: Any | None = None,
        num_denoising_steps: int | None = None,
        cfg_weight: float | tuple[float, float] = (2.0, 2.0),
    ) -> None:
        self.model = model
        self.horizon_frames = int(model.gen_horizon_len)
        self.frames_per_token = int(model.num_frames_per_token)
        self.fps = float(model.motion_rep.fps)

        if self.horizon_frames <= 0:
            raise ValueError("ARDY generation horizon must be positive")
        if history_limit_frames < self.frames_per_token:
            raise ValueError("history_limit_frames is smaller than one ARDY token")
        if history_limit_frames % self.frames_per_token != 0:
            raise ValueError("history_limit_frames must be a multiple of ARDY's token size")

        self.history_limit_frames = int(history_limit_frames)
        default_steps = int(model.diffusion.num_base_steps)
        self.num_denoising_steps = default_steps if num_denoising_steps is None else int(num_denoising_steps)
        if not 1 <= self.num_denoising_steps <= default_steps:
            raise ValueError(
                f"num_denoising_steps must be in 1..{default_steps}, got {self.num_denoising_steps}"
            )

        self.cfg_weight = cfg_weight
        self._history_motion: Any | None = None
        self._prompt: str | None = None
        self._prompt_features: dict[str, tuple[Any, Any]] = {}
        self._postprocess = bool(postprocess)
        self._postprocess_fn = postprocess_fn

        if self._postprocess and self._postprocess_fn is None:
            from ardy.postprocess import post_process_motion

            self._postprocess_fn = post_process_motion

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        history_limit_frames: int | None = None,
        postprocess: bool = True,
        postprocess_fn: Any | None = None,
        num_denoising_steps: int | None = None,
        cfg_weight: float | tuple[float, float] = (2.0, 2.0),
    ) -> "ArdyHeadlessRuntime":
        if history_limit_frames is None:
            fps = float(model.motion_rep.fps)
            patch = int(model.num_frames_per_token)
            horizon = int(model.gen_horizon_len)
            max_window_frames = (int(10 * fps) // patch) * patch
            history_limit_frames = ((max_window_frames - horizon) // patch) * patch
        return cls(
            model,
            history_limit_frames=history_limit_frames,
            postprocess=postprocess,
            postprocess_fn=postprocess_fn,
            num_denoising_steps=num_denoising_steps,
            cfg_weight=cfg_weight,
        )

    @classmethod
    def load(
        cls,
        *,
        model_name: str = "core",
        device: str = "auto",
        checkpoints_dir: str | None = None,
        text_encoder_mode: str | None = None,
        history_limit_frames: int | None = None,
        postprocess: bool = True,
        num_denoising_steps: int | None = None,
        cfg_weight: float | tuple[float, float] = (2.0, 2.0),
    ) -> "ArdyHeadlessRuntime":
        from ardy.model import load_model

        model = load_model(
            model_name,
            device=device,
            checkpoints_dir=checkpoints_dir,
            text_encoder_mode=text_encoder_mode,
        )
        return cls.from_model(
            model,
            history_limit_frames=history_limit_frames,
            postprocess=postprocess,
            num_denoising_steps=num_denoising_steps,
            cfg_weight=cfg_weight,
        )

    @property
    def prompt(self) -> str | None:
        return self._prompt

    @property
    def history_frame_count(self) -> int:
        if self._history_motion is None:
            return 0
        return int(self._history_motion.shape[1])

    def clear_history(self) -> None:
        self._history_motion = None

    def set_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        if prompt not in self._prompt_features:
            # This is the same text-conditioning path used internally by
            # Ardy.autoregressive_step when text_feat is not supplied. We cache
            # it so prompt reuse does not re-run the LLM text encoder.
            text_feat, text_pad_mask = self.model._encode_text([prompt])
            self._prompt_features[prompt] = (text_feat, text_pad_mask)
        self._prompt = prompt

    def generate_next(self) -> ArdyMotionChunk:
        if self._prompt is None:
            raise RuntimeError("set a prompt before generating ARDY motion")

        text_feat, text_pad_mask = self._prompt_features[self._prompt]
        history = self._history_motion
        history_len = 0 if history is None else int(history.shape[1])
        num_frames = history_len + self.horizon_frames
        start = time.perf_counter()

        try:
            import torch

            no_grad = torch.no_grad()
        except ImportError:  # pragma: no cover - real ARDY always provides torch
            no_grad = _NullContext()

        with no_grad:
            samples = self.model.autoregressive_step(
                num_frames=num_frames,
                num_denoising_steps=self.num_denoising_steps,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=self.cfg_weight,
                texts=None,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                init_history_sequence=history,
            )

            if int(samples.shape[1]) != num_frames:
                raise RuntimeError(
                    f"ARDY returned {samples.shape[1]} frames; expected history {history_len} + horizon {self.horizon_frames}"
                )

            # Preserve ARDY-native normalized motion for the next rollout. The
            # returned sample already contains the supplied history followed by
            # the newly generated horizon, so cropping the tail is sufficient.
            self._history_motion = samples[:, -self.history_limit_frames :].detach()

            unnormalized = self.model.motion_rep.unnormalize(samples)
            decoded = self.model.motion_rep.inverse(unnormalized, is_normalized=False)

            if self._postprocess:
                corrected = self._postprocess_fn(
                    decoded["local_rot_mats"],
                    decoded["root_positions"],
                    decoded["foot_contacts"],
                    self.model.skeleton,
                )
                decoded.update(corrected)

        arrays = {key: _tensor_to_numpy(value) for key, value in decoded.items()}
        required = ("posed_joints", "global_rot_mats", "root_positions", "global_root_heading")
        missing = [key for key in required if key not in arrays]
        if missing:
            raise RuntimeError(f"ARDY decoded output is missing required fields: {', '.join(missing)}")

        horizon_slice = slice(history_len, history_len + self.horizon_frames)
        root_positions = np.asarray(arrays["root_positions"])[0, horizon_slice]
        smooth_root = np.asarray(arrays.get("smooth_root_pos", arrays["root_positions"]))[0, horizon_slice]
        foot_contacts_source = arrays.get("foot_contacts")
        if foot_contacts_source is None:
            foot_contacts = np.zeros((self.horizon_frames, 4), dtype=np.float32)
        else:
            foot_contacts = np.asarray(foot_contacts_source)[0, horizon_slice]

        return ArdyMotionChunk(
            posed_joints=np.asarray(arrays["posed_joints"])[0, horizon_slice],
            global_rot_mats=np.asarray(arrays["global_rot_mats"])[0, horizon_slice],
            root_positions=root_positions,
            smooth_root_pos=smooth_root,
            global_root_heading=np.asarray(arrays["global_root_heading"])[0, horizon_slice],
            foot_contacts=foot_contacts,
            fps=self.fps,
            prompt=self._prompt,
            generation_seconds=time.perf_counter() - start,
        )


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False