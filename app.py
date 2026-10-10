"""
SignalGuard
Crane hand-signal decision support: pose tracking, sequence classification,
open-set rejection, telemetry cross-check, alerts, audit, dashboard, training,
evaluation, and unit tests.

All logic is intentionally embedded in this single file.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("signalguard")


# ---------------------------------------------------------------------------
# Embedded configuration: replaces config.yaml / signals.yaml in this 2-file build
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "app": {
        "title": "SignalGuard",
        "disclaimer": (
            "Decision support only. Not a certified safety function. "
            "A qualified human signaller/operator must confirm all actions."
        ),
    },
    "camera": {
        "width": 960,
        "height": 540,
        "fps": 15,
    },
    "inference": {
        "window_size": 64,
        "stride": 8,
        "conf_threshold": 0.62,
        "unclear_threshold": 0.45,
        "visibility_threshold": 0.45,
        "max_history": 600,
        "max_audit": 5000,
        "max_alerts": 200,
    },
    "models": {
        # Model names/weights change often. Candidates are tried lazily.
        "yolo_pose": ["yolo26n-pose.pt", "yolo11n-pose.pt", "yolov8n-pose.pt"],
        "hand_landmarker": ["hand_landmarker.task"],
        "classifier_checkpoint": ["signalguard_classifier.pt"],
    },
    "relay": {
        "enabled": False,
        "broker": "localhost",
        "port": 1883,
        "topic": "signalguard/emergency/request",
    },
    "telemetry": {
        "mode": "simulator",
        "rest_url": "",
        "mqtt_broker": "",
        "mqtt_topic": "crane/telemetry",
    },
    "signals": {
        "idle": {
            "label": "IDLE / NONE",
            "color": [170, 170, 170],
            "directional": False,
            "mirror_to": "idle",
            "description": "No command",
            "expected": "no_motion",
        },
        "unknown": {
            "label": "UNKNOWN / UNCLEAR",
            "color": [255, 0, 255],
            "directional": False,
            "mirror_to": "unknown",
            "description": "Open-set rejection / unclear signal",
            "expected": "stop",
        },
        "stop": {
            "label": "STOP",
            "color": [255, 90, 0],
            "directional": False,
            "mirror_to": "stop",
            "description": "Stop all motion",
            "expected": "stop",
        },
        "emergency_stop": {
            "label": "EMERGENCY STOP",
            "color": [255, 0, 0],
            "directional": False,
            "mirror_to": "emergency_stop",
            "description": "Emergency stop request",
            "expected": "emergency_stop",
        },
        "hoist": {
            "label": "HOIST",
            "color": [0, 200, 255],
            "directional": False,
            "mirror_to": "hoist",
            "description": "Raise load",
            "expected": "up",
        },
        "lower": {
            "label": "LOWER",
            "color": [0, 160, 255],
            "directional": False,
            "mirror_to": "lower",
            "description": "Lower load",
            "expected": "down",
        },
        "travel_left": {
            "label": "TRAVEL LEFT",
            "color": [0, 255, 120],
            "directional": True,
            "mirror_to": "travel_right",
            "description": "Travel left",
            "expected": "left",
        },
        "travel_right": {
            "label": "TRAVEL RIGHT",
            "color": [0, 255, 120],
            "directional": True,
            "mirror_to": "travel_left",
            "description": "Travel right",
            "expected": "right",
        },
        "boom_up": {
            "label": "BOOM UP",
            "color": [255, 220, 0],
            "directional": False,
            "mirror_to": "boom_up",
            "description": "Boom up",
            "expected": "boom_up",
        },
        "boom_down": {
            "label": "BOOM DOWN",
            "color": [255, 180, 0],
            "directional": False,
            "mirror_to": "boom_down",
            "description": "Boom down",
            "expected": "boom_down",
        },
        "dog_down": {
            "label": "DOG DOWN",
            "color": [180, 120, 255],
            "directional": False,
            "mirror_to": "dog_down",
            "description": "Dog down / boom lowering variant",
            "expected": "boom_down",
        },
    },
}

MODELS = CONFIG["models"]
SIGNALS = CONFIG["signals"]

COCO_KEYPOINTS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_eye",
    "right_eye",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

ARM_KEYPOINT_IDS = [5, 6, 7, 8, 9, 10]

LEFT_RIGHT_SWAP = {
    1: 2, 2: 1,
    3: 4, 4: 3,
    5: 6, 6: 5,
    7: 8, 8: 7,
    9: 10, 10: 9,
    11: 12, 12: 11,
    13: 14, 14: 13,
    15: 16, 16: 15,
}

ACTIVE_EXCLUDE = {"idle", "unknown"}

VEST_COLORS_RGB = {
    "orange": (255, 140, 0),
    "yellow": (230, 220, 0),
    "green": (0, 200, 80),
    "blue": (0, 120, 255),
    "neutral": (180, 180, 180),
    "other": (160, 160, 160),
    "unknown": (120, 120, 120),
}

HEURISTIC_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "idle": {
        "left_wrist": (-0.18, 1.10),
        "right_wrist": (0.18, 1.10),
        "left_elbow": (-0.22, 0.55),
        "right_elbow": (0.22, 0.55),
        "motion": 0.08,
        "lateral_motion": 0.03,
    },
    "stop": {
        "left_wrist": (-1.05, 0.02),
        "right_wrist": (1.05, 0.02),
        "left_elbow": (-0.55, 0.02),
        "right_elbow": (0.55, 0.02),
        "motion": 0.10,
        "lateral_motion": 0.05,
    },
    "emergency_stop": {
        "left_wrist": (0.55, -0.70),
        "right_wrist": (-0.55, -0.70),
        "left_elbow": (-0.45, -0.25),
        "right_elbow": (0.45, -0.25),
        "motion": 0.25,
        "lateral_motion": 0.18,
    },
    "hoist": {
        "left_wrist": (-0.18, 1.05),
        "right_wrist": (0.12, -1.25),
        "left_elbow": (-0.22, 0.55),
        "right_elbow": (0.08, -0.55),
        "motion": 0.18,
        "lateral_motion": 0.05,
    },
    "lower": {
        "left_wrist": (-0.18, 1.05),
        "right_wrist": (0.12, 1.55),
        "left_elbow": (-0.22, 0.55),
        "right_elbow": (0.08, 1.05),
        "motion": 0.18,
        "lateral_motion": 0.05,
    },
    "travel_left": {
        "left_wrist": (-1.15, 0.12),
        "right_wrist": (-0.75, 0.18),
        "left_elbow": (-0.60, 0.08),
        "right_elbow": (-0.15, 0.10),
        "motion": 0.14,
        "lateral_motion": 0.12,
    },
    "travel_right": {
        "left_wrist": (0.75, 0.18),
        "right_wrist": (1.15, 0.12),
        "left_elbow": (0.15, 0.10),
        "right_elbow": (0.60, 0.08),
        "motion": 0.14,
        "lateral_motion": 0.12,
    },
    "boom_up": {
        "left_wrist": (-0.18, 1.05),
        "right_wrist": (0.85, -0.95),
        "left_elbow": (-0.22, 0.55),
        "right_elbow": (0.45, -0.35),
        "motion": 0.16,
        "lateral_motion": 0.08,
    },
    "boom_down": {
        "left_wrist": (-0.18, 1.05),
        "right_wrist": (0.85, 1.25),
        "left_elbow": (-0.22, 0.55),
        "right_elbow": (0.45, 0.75),
        "motion": 0.16,
        "lateral_motion": 0.08,
    },
    "dog_down": {
        "left_wrist": (-0.25, 1.35),
        "right_wrist": (0.25, 1.35),
        "left_elbow": (-0.28, 0.80),
        "right_elbow": (0.28, 0.80),
        "motion": 0.10,
        "lateral_motion": 0.04,
    },
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def rgb_to_bgr(color: Tuple[int, int, int] | List[int]) -> Tuple[int, int, int]:
    return int(color[2]), int(color[1]), int(color[0])


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    if temperature <= 0:
        temperature = 1e-3
    x = x / float(temperature)
    x = x - np.max(x)
    e = np.exp(x)
    s = float(np.sum(e))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(x, 1.0 / max(x.size, 1), dtype=np.float32)
    return (e / s).astype(np.float32)


def signal_labels() -> List[str]:
    return list(SIGNALS.keys())


def label_for(code: str) -> str:
    return str(SIGNALS.get(code, {}).get("label", str(code).upper()))


def is_active_signal(code: str) -> bool:
    return code not in ACTIVE_EXCLUDE


def mirror_label(code: str) -> str:
    return str(SIGNALS.get(code, {}).get("mirror_to", code))


def ss_setdefault(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Streamlit compatibility helpers
# ---------------------------------------------------------------------------

def st_image_compat(image: Any, caption: Optional[str] = None, use_container_width: bool = True) -> None:
    """
    Display an image across Streamlit versions.

    Some versions support channels=..., use_container_width=..., while others do not.
    """
    renderable = image

    # Some Streamlit versions handle bytes poorly; convert obvious image bytes to PIL.
    if isinstance(renderable, (bytes, bytearray)):
        try:
            renderable = Image.open(io.BytesIO(renderable))
        except Exception:
            pass

    kwargs: Dict[str, Any] = {}
    if caption is not None:
        kwargs["caption"] = caption
    if use_container_width:
        kwargs["use_container_width"] = True

    try:
        st.image(renderable, channels="rgb", **kwargs)
        return
    except TypeError:
        pass
    except Exception as exc:
        logger.warning("st.image(..., channels='rgb') failed: %s", exc)

    try:
        st.image(renderable, **kwargs)
        return
    except TypeError:
        kwargs.pop("use_container_width", None)
    except Exception as exc:
        logger.warning("st.image(...) failed: %s", exc)

    try:
        st.image(renderable, **kwargs)
    except Exception as exc:
        st.error(f"Image render failed: {exc}")


def st_video_compat(video_bytes: Optional[bytes], mime: str) -> None:
    """
    Play generated demo video across Streamlit versions.

    Avoids incompatible calls such as:
        st.video(data, mime=mime)
    """
    if not video_bytes:
        st.info("Video renderer unavailable; live overlay below is still running.")
        return

    # GIF fallback from render_demo_video().
    if not str(mime).startswith("video"):
        st_image_compat(video_bytes)
        return

    # Preferred: direct bytes.
    try:
        st.video(video_bytes)
        return
    except TypeError:
        pass
    except Exception as exc:
        logger.warning("st.video(bytes) failed: %s", exc)

    # Fallback: write to a temporary local file and pass the path.
    try:
        suffix = ".mp4" if "mp4" in str(mime) else ".video"
        path = Path(tempfile.gettempdir()) / f"signalguard_demo_video{suffix}"
        if not path.exists() or path.stat().st_size != len(video_bytes):
            path.write_bytes(video_bytes)
        st.video(str(path))
        return
    except Exception as exc:
        st.error(
            "Could not play inline demo video: "
            f"{exc}. The live overlay below remains available."
        )


def st_dataframe_compat(
    df: pd.DataFrame,
    hide_index: bool = True,
    use_container_width: bool = True,
) -> None:
    """
    Display a dataframe across Streamlit versions.
    """
    kwargs: Dict[str, Any] = {}
    if hide_index:
        kwargs["hide_index"] = True
    if use_container_width:
        kwargs["use_container_width"] = True

    try:
        st.dataframe(df, **kwargs)
        return
    except TypeError:
        kwargs.pop("hide_index", None)
        kwargs.pop("use_container_width", None)
    except Exception as exc:
        st.error(f"Dataframe render failed: {exc}")
        return

    try:
        st.dataframe(df, **kwargs)
    except Exception as exc:
        st.error(f"Dataframe render failed: {exc}")


# ---------------------------------------------------------------------------
# Geometry, normalization, augmentation, windows, visibility
# ---------------------------------------------------------------------------

def neck_point(kps: np.ndarray) -> np.ndarray:
    kps = np.asarray(kps, dtype=np.float32)
    left = kps[5]
    right = kps[6]
    conf = float(left[2] + right[2])
    if conf < 0.08:
        return np.array([np.mean(kps[:, 0]), np.mean(kps[:, 1])], dtype=np.float32)
    return ((left[:2] * left[2] + right[:2] * right[2]) / max(conf, 1e-6)).astype(np.float32)


def torso_length(kps: np.ndarray) -> float:
    kps = np.asarray(kps, dtype=np.float32)
    neck = neck_point(kps)
    left_hip = kps[11]
    right_hip = kps[12]
    conf = float(left_hip[2] + right_hip[2])
    if conf < 0.08:
        hip = np.array([np.mean(kps[[11, 12], 0]), np.mean(kps[[11, 12], 1])], dtype=np.float32)
    else:
        hip = ((left_hip[:2] * left_hip[2] + right_hip[:2] * right_hip[2]) / max(conf, 1e-6)).astype(np.float32)
    length = float(np.linalg.norm(hip - neck))
    return length if np.isfinite(length) else 0.0


def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[0] < 17:
        return np.zeros((17, 2), dtype=np.float32)
    neck = neck_point(kps)
    scale = torso_length(kps)
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    norm = (kps[:17, :2] - neck) / float(scale)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    return norm.astype(np.float32)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.size == 0:
        return np.zeros((0, 17, 2), dtype=np.float32)
    return np.stack([normalize_keypoints(frame) for frame in seq], axis=0).astype(np.float32)


def mirror_keypoints(kps: np.ndarray, width: Optional[int] = None) -> np.ndarray:
    arr = np.array(kps, copy=True, dtype=np.float32)
    keys = list(LEFT_RIGHT_SWAP.keys())
    vals = list(LEFT_RIGHT_SWAP.values())
    if arr.ndim == 2:
        if width is not None:
            arr[:, 0] = float(width) - arr[:, 0]
        else:
            arr[:, 0] = -arr[:, 0]
        arr[keys, :] = arr[vals, :]
    elif arr.ndim == 3:
        if width is not None:
            arr[:, :, 0] = float(width) - arr[:, :, 0]
        else:
            arr[:, :, 0] = -arr[:, :, 0]
        arr[:, keys, :] = arr[:, vals, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_length(seq: np.ndarray, length: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[0] >= length:
        return seq[:length].copy()
    if seq.shape[0] == 0:
        return np.zeros((length,) + seq.shape[1:], dtype=np.float32)
    pad = np.repeat(seq[-1:], length - seq.shape[0], axis=0)
    return np.concatenate([seq, pad], axis=0).astype(np.float32)


def resample_sequence(seq: np.ndarray, factor: float, target_len: Optional[int] = None) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[0] == 0:
        return seq
    factor = max(float(factor), 1e-3)
    t_in = seq.shape[0]
    t_out = max(2, int(round(t_in / factor)))
    old_idx = np.arange(t_in, dtype=np.float32)
    new_idx = np.linspace(0, t_in - 1, t_out, dtype=np.float32)
    flat = seq.reshape(t_in, -1)
    out_flat = np.empty((t_out, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out_flat[:, j] = np.interp(new_idx, old_idx, flat[:, j])
    out = out_flat.reshape((t_out,) + seq.shape[1:])
    if target_len is not None:
        out = fit_length(out, target_len)
    return out.astype(np.float32)


def iter_windows(seq: np.ndarray, window_size: int = 64, stride: int = 8):
    seq = np.asarray(seq, dtype=np.float32)
    n = seq.shape[0]
    if n == 0:
        return
    if n < window_size:
        yield 0, fit_length(seq, window_size)
        return
    for start in range(0, n - window_size + 1, stride):
        yield start, seq[start:start + window_size]


def visibility_score(kps: np.ndarray) -> float:
    arr = np.asarray(kps, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= max(ARM_KEYPOINT_IDS):
        return 0.0
    conf = arr[ARM_KEYPOINT_IDS, 2]
    conf = conf[np.isfinite(conf)]
    if conf.size == 0:
        return 0.0
    mean_conf = float(np.mean(conf))
    low_ratio = float(np.mean(conf < 0.20))
    return float(np.clip(mean_conf - 0.25 * low_ratio, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Signaller tracker
# ---------------------------------------------------------------------------

class SignallerTracker:
    """
    Locks onto a designated vest/armband color, keeps a stable track id,
    and re-acquires after short losses.
    """

    def __init__(self, vest_color: str = "orange", max_lost_frames: int = 18, min_score: float = 0.20):
        self.vest_color = vest_color
        self.max_lost_frames = max_lost_frames
        self.min_score = min_score
        self.locked_id: Optional[int] = None
        self.lost_frames = 0
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

    def update(self, detections: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
        if not detections:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.locked_id = None
                self.last_bbox = None
            return None, "searching" if self.locked_id is None else "lost"

        locked = next((d for d in detections if int(d.get("track_id", -1)) == self.locked_id), None)
        if locked is not None and float(locked.get("score", 0.0)) >= self.min_score:
            locked_color = str(locked.get("vest_color", "unknown"))
            if locked_color == self.vest_color or self.lost_frames == 0:
                self.lost_frames = 0
                self.last_bbox = tuple(locked["bbox"])
                return locked, "locked"

        vest_candidates = [
            d for d in detections
            if str(d.get("vest_color", "unknown")) == self.vest_color
            and float(d.get("score", 0.0)) >= self.min_score
        ]

        if vest_candidates:
            if self.last_bbox is not None:
                best = max(
                    vest_candidates,
                    key=lambda d: (bbox_iou(tuple(d["bbox"]), self.last_bbox), float(d.get("score", 0.0))),
                )
            else:
                best = max(vest_candidates, key=lambda d: float(d.get("score", 0.0)))
            status = "reacquired" if self.lost_frames > 0 or self.locked_id != int(best["track_id"]) else "locked"
            self.locked_id = int(best["track_id"])
            self.lost_frames = 0
            self.last_bbox = tuple(best["bbox"])
            return best, status

        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.locked_id = None
            self.last_bbox = None
        return None, "searching" if self.locked_id is None else "lost"


# ---------------------------------------------------------------------------
# Classifiers: heuristic fallback + optional PyTorch temporal transformer
# ---------------------------------------------------------------------------

class HeuristicSignalClassifier:
    name = "heuristic"

    def __init__(self) -> None:
        self.classes = signal_labels()
        self.last_latency_ms = 0.0

    def predict_window(
        self,
        norm_seq: np.ndarray,
        visibility: float = 1.0,
        has_signaller: bool = True,
        conf_threshold: float = 0.62,
        unclear_threshold: float = 0.45,
        visibility_threshold: float = 0.45,
    ) -> Tuple[str, float, Dict[str, float]]:
        t0 = time.perf_counter()
        if not has_signaller or norm_seq is None or len(norm_seq) == 0:
            probs = {c: 1.0 / len(self.classes) for c in self.classes}
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return "unknown", 0.0, probs

        seq = np.asarray(norm_seq, dtype=np.float32)
        if seq.ndim == 2:
            seq = seq[None, ...]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        left_wrist = np.mean(seq[:, 9, :], axis=0)
        right_wrist = np.mean(seq[:, 10, :], axis=0)
        left_elbow = np.mean(seq[:, 7, :], axis=0)
        right_elbow = np.mean(seq[:, 8, :], axis=0)

        if seq.shape[0] > 1:
            wrist_diff = np.diff(seq[:, 9:11, 0], axis=0)
            motion_x = float(np.mean(np.abs(wrist_diff)))
            motion_y = float(np.mean(np.std(seq[:, 9:11, 1], axis=0)))
        else:
            motion_x = motion_y = 0.0

        energy = float(np.mean(np.std(seq[:, [5, 6, 7, 8, 9, 10], :], axis=0)))
        raw_scores: Dict[str, float] = {}

        for code in self.classes:
            if code == "unknown":
                continue
            tmpl = HEURISTIC_TEMPLATES.get(code)
            if tmpl is None:
                raw_scores[code] = -10.0
                continue

            dist = (
                float(np.linalg.norm(left_wrist - np.array(tmpl["left_wrist"], dtype=np.float32)))
                + float(np.linalg.norm(right_wrist - np.array(tmpl["right_wrist"], dtype=np.float32)))
                + 0.55 * float(np.linalg.norm(left_elbow - np.array(tmpl["left_elbow"], dtype=np.float32)))
                + 0.55 * float(np.linalg.norm(right_elbow - np.array(tmpl["right_elbow"], dtype=np.float32)))
            )
            mot = (
                -1.2 * abs(motion_y - float(tmpl["motion"]))
                - 0.8 * abs(motion_x - float(tmpl.get("lateral_motion", tmpl["motion"] * 0.5)))
            )
            bonus = 0.0
            if code == "travel_left" and left_wrist[0] < -0.65 and right_wrist[0] < -0.20:
                bonus += 1.2
            if code == "travel_right" and left_wrist[0] > 0.20 and right_wrist[0] > 0.65:
                bonus += 1.2
            if code == "hoist" and right_wrist[1] < -0.75:
                bonus += 1.0
            if code == "lower" and right_wrist[1] > 1.25:
                bonus += 1.0
            if code == "emergency_stop" and left_wrist[1] < -0.35 and right_wrist[1] < -0.35:
                bonus += 1.3
            if code == "stop" and abs(left_wrist[1]) < 0.35 and abs(right_wrist[1]) < 0.35 and left_wrist[0] < -0.55 and right_wrist[0] > 0.55:
                bonus += 1.2

            raw_scores[code] = -1.75 * dist + mot + bonus - 0.35 * energy

        max_known = max(raw_scores.values()) if raw_scores else -10.0
        if not np.isfinite(max_known):
            max_known = -10.0
        raw_scores["unknown"] = 0.35 - 1.5 * float(visibility) + 0.25 * (1.0 - math.tanh(max_known + 2.0))

        logits = np.array([raw_scores[c] for c in self.classes], dtype=np.float32)
        probs = softmax(logits, temperature=0.22)

        if visibility < visibility_threshold and "unknown" in self.classes:
            unknown_idx = self.classes.index("unknown")
            boost = (visibility_threshold - max(float(visibility), 0.0)) * 2.5
            probs[unknown_idx] += boost
            probs = probs / max(float(np.sum(probs)), 1e-9)

        idx = int(np.argmax(probs))
        code = self.classes[idx]
        conf = float(probs[idx])
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return code, conf, {c: float(p) for c, p in zip(self.classes, probs)}


def _torch_model_class() -> Any:
    import torch
    from torch import nn

    class TinyKeypointTransformer(nn.Module):
        def __init__(
            self,
            input_dim: int = 34,
            hidden_dim: int = 96,
            num_heads: int = 4,
            num_layers: int = 2,
            num_classes: int = 10,
            max_len: int = 64,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )
            self.pos = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: Any, padding_mask: Any = None) -> Any:
            x = self.proj(x)
            t = x.size(1)
            x = x + self.pos[:, :t, :]
            x = self.encoder(x, src_key_padding_mask=padding_mask)
            x = torch.mean(x, dim=1)
            return self.head(x)

    return TinyKeypointTransformer


class TorchSignalClassifier:
    name = "transformer"

    def __init__(self, model: Any, classes: List[str], device: str = "cpu") -> None:
        self.model = model
        self.classes = list(classes)
        self.device = device
        self.last_latency_ms = 0.0

    def predict_window(
        self,
        norm_seq: np.ndarray,
        visibility: float = 1.0,
        has_signaller: bool = True,
        conf_threshold: float = 0.62,
        unclear_threshold: float = 0.45,
        visibility_threshold: float = 0.45,
    ) -> Tuple[str, float, Dict[str, float]]:
        import torch

        t0 = time.perf_counter()
        if not has_signaller or norm_seq is None or len(norm_seq) == 0:
            probs = {c: 1.0 / max(len(self.classes), 1) for c in self.classes}
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return "unknown", 0.0, probs

        seq = np.asarray(norm_seq, dtype=np.float32)
        if seq.ndim == 2:
            seq = seq[None, ...]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq = fit_length(seq, 64)
        x = torch.from_numpy(seq.reshape(1, seq.shape[0], -1)).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        probs = softmax(logits[0].detach().cpu().numpy(), temperature=0.18)

        if visibility < visibility_threshold and "unknown" in self.classes:
            unknown_idx = self.classes.index("unknown")
            boost = (visibility_threshold - max(float(visibility), 0.0)) * 2.5
            probs[unknown_idx] += boost
            probs = probs / max(float(np.sum(probs)), 1e-9)

        idx = int(np.argmax(probs))
        code = self.classes[idx] if idx < len(self.classes) else "unknown"
        conf = float(probs[idx])
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return code, conf, {c: float(p) for c, p in zip(self.classes, probs)}


def load_torch_classifier_from_paths(extra_paths: Optional[List[str]] = None) -> Optional[TorchSignalClassifier]:
    candidates: List[str] = []
    if extra_paths:
        candidates.extend([p for p in extra_paths if p])
    candidates.extend(list(MODELS.get("classifier_checkpoint", [])))

    for cand in candidates:
        path = Path(cand)
        if not path.exists():
            continue
        try:
            import torch

            model_cls = _torch_model_class()
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")

            classes = list(ckpt.get("classes", signal_labels()))
            model = model_cls(
                input_dim=int(ckpt.get("input_dim", 34)),
                hidden_dim=int(ckpt.get("hidden_dim", 96)),
                num_classes=len(classes),
                max_len=int(ckpt.get("max_len", 64)),
            )
            model.load_state_dict(ckpt["model"])
            model.eval()
            logger.info("Loaded trained SignalGuard classifier from %s", path)
            return TorchSignalClassifier(model, classes, device="cpu")
        except Exception as exc:
            logger.warning("Classifier %s unavailable: %s. Falling back to heuristic.", cand, exc)

    return None


def train_transformer_from_samples(
    samples: List[Tuple[str, np.ndarray, float]],
    epochs: int = 40,
    batch_size: int = 8,
) -> Tuple[Optional[str], str]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        return None, f"PyTorch unavailable: {exc}"

    if len(samples) < 6:
        return None, "Need at least 6 captured windows before training."

    window_size = int(CONFIG["inference"]["window_size"])
    rng = np.random.default_rng(17)
    augmented: List[Tuple[str, np.ndarray, float]] = []

    for label, window, vis in samples:
        if label not in SIGNALS:
            label = "unknown"
        w = fit_length(np.nan_to_num(np.asarray(window, dtype=np.float32)), window_size)
        augmented.append((label, w, vis))

        mirrored = mirror_keypoints(w)
        mlabel = mirror_label(label) if SIGNALS.get(label, {}).get("directional", False) else label
        augmented.append((mlabel, mirrored, vis))

        for factor in (0.8, 1.25):
            sped = fit_length(resample_sequence(w, factor, target_len=window_size), window_size)
            augmented.append((label, sped, vis))

        for _ in range(2):
            jitter = w + rng.normal(0.0, 0.035, size=w.shape).astype(np.float32)
            augmented.append((label, jitter, vis))

    idle_windows = [w for lbl, w, _ in augmented if lbl == "idle"]
    unknown_count = max(3, len(augmented) // 8)
    for i in range(unknown_count):
        if idle_windows:
            base = idle_windows[i % len(idle_windows)]
        else:
            base = rng.normal(0.0, 0.25, size=(window_size, 17, 2)).astype(np.float32)
        noisy = base + rng.normal(0.0, 0.12, size=base.shape).astype(np.float32)
        augmented.append(("unknown", noisy.astype(np.float32), 0.25))

    labels = sorted({lbl for lbl, _, _ in augmented if lbl in SIGNALS} | {"idle", "unknown"})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    x_np = np.stack([w.reshape(w.shape[0], -1) for _, w, _ in augmented], axis=0).astype(np.float32)
    y_np = np.array([label_to_idx[lbl] for lbl, _, _ in augmented], dtype=np.int64)

    class_counts = np.bincount(y_np, minlength=len(labels)).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1.0)
    class_weights = class_weights / float(np.mean(class_weights))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = _torch_model_class()
    model = model_cls(
        input_dim=int(x_np.shape[2]),
        hidden_dim=96,
        num_classes=len(labels),
        max_len=int(x_np.shape[1]),
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.to("cpu")
    out_path = Path(tempfile.gettempdir()) / "signalguard_classifier.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "classes": labels,
            "input_dim": int(x_np.shape[2]),
            "hidden_dim": 96,
            "max_len": int(x_np.shape[1]),
            "trained_at": utc_now(),
        },
        out_path,
    )
    return str(out_path), f"Trained on {len(augmented)} augmented windows. Checkpoint: {out_path}"


def apply_open_set(
    code: str,
    conf: float,
    visibility: float,
    conf_threshold: float,
    unclear_threshold: float,
    visibility_threshold: float,
) -> Tuple[str, float]:
    if code not in SIGNALS:
        return "unknown", float(conf)
    if visibility < visibility_threshold:
        return "unknown", float(conf)
    if code == "unknown" or conf < unclear_threshold:
        return "unknown", float(conf)
    if code != "idle" and conf < conf_threshold:
        return "unknown", float(conf)
    return code, float(conf)


# ---------------------------------------------------------------------------
# Optional real-camera model loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pose_model() -> Optional[Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        logger.warning("Ultralytics not available: %s. Using simulator/demo pose source.", exc)
        return None

    for name in MODELS.get("yolo_pose", []):
        try:
            model = YOLO(name)
            logger.info("Loaded YOLO pose model: %s", name)
            return model
        except Exception as exc:
            logger.warning("Pose model %s unavailable: %s", name, exc)

    logger.warning("No YOLO pose model could be loaded. Falling back to simulator/demo.")
    return None


@st.cache_resource(show_spinner=False)
def load_hand_model() -> Optional[Any]:
    try:
        import mediapipe as mp
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
        )
    except Exception as exc:
        logger.warning("MediaPipe Hands unavailable: %s. Body pose only.", exc)
        return None


class SimpleIoUTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.active: Dict[int, Tuple[float, float, float, float]] = {}

    def update(self, boxes: np.ndarray) -> List[int]:
        ids: List[int] = []
        used = set()
        new_active: Dict[int, Tuple[float, float, float, float]] = {}
        for box in boxes:
            bbox = tuple(float(v) for v in box)
            best_id = None
            best_iou = 0.30
            for tid, old_bbox in self.active.items():
                if tid in used:
                    continue
                iou = bbox_iou(bbox, old_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            ids.append(int(best_id))
            new_active[best_id] = bbox
        self.active = new_active
        return ids


def estimate_vest_color(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[:, :, 0]))
    sat = float(np.median(hsv[:, :, 1]))
    val = float(np.median(hsv[:, :, 2]))
    if sat < 40 or val < 40:
        return "neutral"
    if 5 <= hue <= 22:
        return "orange"
    if 22 < hue <= 35:
        return "yellow"
    if 35 < hue <= 85:
        return "green"
    if 85 < hue <= 130:
        return "blue"
    return "other"


def apply_mediapipe_hands(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hand_model = load_hand_model()
    if hand_model is None or not detections:
        return detections
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hand_model.process(rgb)
        if res.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for det in detections:
                kps = det["keypoints"]
                for hand in res.multi_hand_landmarks:
                    wrist = hand.landmark[0]
                    px, py = int(wrist.x * w), int(wrist.y * h)
                    for wrist_idx in (9, 10):
                        if np.linalg.norm(kps[wrist_idx, :2] - np.array([px, py], dtype=np.float32)) < 45:
                            kps[wrist_idx, 2] = float(min(0.99, max(float(kps[wrist_idx, 2]), 0.75)))
    except Exception as exc:
        logger.debug("MediaPipe hand enhancement skipped: %s", exc)
    return detections


def yolo_detections(
    frame: np.ndarray,
    model: Any,
    tracker: Optional[SimpleIoUTracker] = None,
) -> List[Dict[str, Any]]:
    if tracker is None:
        if "simple_tracker" not in st.session_state:
            st.session_state.simple_tracker = SimpleIoUTracker()
        tracker = st.session_state.simple_tracker

    try:
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
    except Exception:
        results = model.predict(frame, classes=[0], verbose=False)

    detections: List[Dict[str, Any]] = []
    for r in results:
        if getattr(r, "keypoints", None) is None:
            continue
        kps = r.keypoints.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        if getattr(r.boxes, "id", None) is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.array(tracker.update(boxes), dtype=int)

        for kp, box, cf, tid in zip(kps, boxes, confs, ids):
            vest = estimate_vest_color(frame, tuple(int(v) for v in box))
            detections.append(
                {
                    "track_id": int(tid),
                    "bbox": tuple(int(v) for v in box),
                    "keypoints": np.asarray(kp, dtype=np.float32),
                    "vest_color": vest,
                    "score": float(cf),
                }
            )

    return apply_mediapipe_hands(frame, detections)


def get_or_create_capture() -> Optional[cv2.VideoCapture]:
    source = st.session_state.get("source", "Synthetic site demo")

    if source == "Webcam":
        if "webcam_cap" not in st.session_state:
            st.session_state.webcam_cap = cv2.VideoCapture(0)
        return st.session_state.webcam_cap

    if source == "Uploaded video":
        uploaded = st.session_state.get("uploaded_file")
        if uploaded is None:
            old_cap = st.session_state.pop("upload_cap", None)
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass

            old_path = st.session_state.pop("upload_path", None)
            if old_path and Path(old_path).exists():
                try:
                    os.unlink(old_path)
                except Exception:
                    pass
            return None

        name = getattr(uploaded, "name", "upload.mp4")
        if "upload_cap" not in st.session_state or st.session_state.get("upload_name") != name:
            old_cap = st.session_state.get("upload_cap")
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass

            old_path = st.session_state.get("upload_path")
            if old_path and Path(old_path).exists():
                try:
                    os.unlink(old_path)
                except Exception:
                    pass

            suffix = Path(name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.getbuffer())
            tmp.close()

            st.session_state.upload_path = tmp.name
            st.session_state.upload_cap = cv2.VideoCapture(tmp.name)
            st.session_state.upload_name = name

        return st.session_state.upload_cap

    return None


# ---------------------------------------------------------------------------
# Telemetry simulator and alert/relay logic
# ---------------------------------------------------------------------------

class TelemetrySimulator:
    def __init__(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.boom_angle = 45.0
        self.moving_time = 0.0

    def step(self, dt: float, commanded: str) -> Dict[str, Any]:
        target_vx = target_vy = 0.0
        boom_delta = 0.0
        decay = 0.18

        if commanded == "hoist":
            target_vy = 0.75
        elif commanded == "lower":
            target_vy = -0.75
        elif commanded == "travel_right":
            target_vx = 0.70
        elif commanded == "travel_left":
            target_vx = -0.70
        elif commanded == "boom_up":
            boom_delta = 8.0
        elif commanded in ("boom_down", "dog_down"):
            boom_delta = -8.0
        elif commanded == "stop":
            decay = 0.45
        elif commanded == "emergency_stop":
            decay = 0.75
        elif commanded == "unknown":
            decay = 0.10

        self.vx += (target_vx - self.vx) * min(1.0, decay)
        self.vy += (target_vy - self.vy) * min(1.0, decay)
        self.boom_angle = (self.boom_angle + boom_delta * dt) % 360.0
        moving = math.hypot(self.vx, self.vy) > 0.05 or abs(boom_delta) > 0.05
        self.moving_time = self.moving_time + dt if moving else 0.0

        return {
            "t": time.time(),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "boom_angle": float(self.boom_angle),
            "boom_velocity": float(boom_delta),
            "load_velocity": float(math.hypot(self.vx, self.vy)),
            "moving": bool(moving),
            "moving_duration_s": float(self.moving_time),
            "source_ok": True,
        }


def publish_emergency_relay(payload: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(CONFIG["relay"]["topic"])
    entry = {
        "timestamp": utc_now(),
        "topic": topic,
        "payload": payload,
        "transport": "simulated_local_log",
    }
    if CONFIG["relay"].get("enabled", False):
        try:
            import paho.mqtt.publish as publish_mqtt

            publish_mqtt.single(
                topic,
                payload=json.dumps(payload),
                hostname=str(CONFIG["relay"]["broker"]),
                port=int(CONFIG["relay"]["port"]),
            )
            entry["transport"] = "mqtt"
        except Exception as exc:
            logger.warning("MQTT relay unavailable: %s. Logged locally.", exc)
            entry["transport"] = "local_fallback_after_mqtt_error"
    else:
        logger.info("Simulated emergency relay: %s", payload)
    return entry


@dataclass
class Alert:
    timestamp: str
    code: str
    severity: str
    message: str
    acknowledged: bool = False


def evaluate_alerts(
    prediction: Dict[str, Any],
    telemetry: Dict[str, Any],
    tracker_status: str,
    visibility: float,
    cooldowns: Dict[str, float],
    now_ts: float,
    relay_log: List[Dict[str, Any]],
    visibility_threshold: float,
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    def add(code: str, severity: str, message: str, cooldown: float = 2.0) -> None:
        last = float(cooldowns.get(code, -999.0))
        if now_ts - last >= cooldown:
            alerts.append(Alert(utc_now(), code, severity, message, False).__dict__)
            cooldowns[code] = now_ts

    moving = bool(telemetry.get("moving", False))
    moving_duration = float(telemetry.get("moving_duration_s", 0.0))
    vx = float(telemetry.get("vx", 0.0))
    vy = float(telemetry.get("vy", 0.0))
    pred_code = str(prediction.get("code", "idle"))

    if tracker_status in ("searching", "lost") and moving:
        add(
            "NO_SIGNALLER",
            "critical",
            "No designated signaller visible while crane telemetry reports motion. Stop/re-establish signalling.",
            cooldown=1.5,
        )

    if visibility < visibility_threshold:
        add(
            "LOW_VISIBILITY",
            "warning",
            "Arm/keypoint visibility is low. Verify camera placement, lighting, and signaller position.",
            cooldown=3.0,
        )

    if pred_code == "unknown":
        add(
            "UNCLEAR_SIGNAL",
            "critical",
            "Signal unclear or rejected by open-set gate. Recommend STOP until a human confirms.",
            cooldown=1.5,
        )

    if pred_code in ("stop", "emergency_stop") and moving_duration > 1.0:
        add(
            "STOP_CONFLICT",
            "critical",
            f"STOP shown but crane/load still moving for {moving_duration:.1f}s.",
            cooldown=1.5,
        )

    if pred_code == "travel_left" and vx > 0.12:
        add("DIRECTION_CONFLICT", "critical", "TRAVEL LEFT shown but telemetry indicates rightward motion.", cooldown=2.0)
    if pred_code == "travel_right" and vx < -0.12:
        add("DIRECTION_CONFLICT", "critical", "TRAVEL RIGHT shown but telemetry indicates leftward motion.", cooldown=2.0)
    if pred_code == "hoist" and vy < -0.12:
        add("MOTION_CONFLICT", "critical", "HOIST shown but telemetry indicates lowering.", cooldown=2.0)
    if pred_code == "lower" and vy > 0.12:
        add("MOTION_CONFLICT", "critical", "LOWER shown but telemetry indicates hoisting.", cooldown=2.0)

    if pred_code == "emergency_stop":
        payload = {
            "event": "EMERGENCY_STOP_REQUEST",
            "signal": "emergency_stop",
            "confidence": float(prediction.get("confidence", 0.0)),
            "timestamp": utc_now(),
            "authority": "human_in_the_loop",
            "action": "REQUEST_STOP_NOT_MOTION_COMMAND",
        }
        relay_entry = publish_emergency_relay(payload)
        relay_log.insert(0, relay_entry)
        relay_log[:] = relay_log[:200]
        add(
            "EMERGENCY_RELAY",
            "critical",
            "Emergency-stop request published/logged to relay topic. No motion command issued by SignalGuard.",
            cooldown=1.0,
        )

    if not bool(telemetry.get("source_ok", True)):
        add("TELEMETRY_STALE", "warning", "Crane telemetry source is stale or unavailable.", cooldown=3.0)

    return alerts


# ---------------------------------------------------------------------------
# Synthetic realistic demo generator
# ---------------------------------------------------------------------------

def _base_skeleton(width: int = 960, height: int = 540) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    cx = width * 0.45
    head_y = height * 0.30
    shoulder_y = height * 0.42
    hip_y = height * 0.58
    knee_y = height * 0.74
    ankle_y = height * 0.90
    shoulder_dx = 38
    hip_dx = 24

    pts = {
        0: (cx, head_y),
        1: (cx - 12, head_y - 4),
        2: (cx + 12, head_y - 4),
        3: (cx - 7, head_y - 10),
        4: (cx + 7, head_y - 10),
        5: (cx - shoulder_dx, shoulder_y),
        6: (cx + shoulder_dx, shoulder_y),
        7: (cx - shoulder_dx - 8, shoulder_y + 52),
        8: (cx + shoulder_dx + 8, shoulder_y + 52),
        9: (cx - shoulder_dx - 10, shoulder_y + 105),
        10: (cx + shoulder_dx + 10, shoulder_y + 105),
        11: (cx - hip_dx, hip_y),
        12: (cx + hip_dx, hip_y),
        13: (cx - hip_dx - 4, knee_y),
        14: (cx + hip_dx + 4, knee_y),
        15: (cx - hip_dx - 6, ankle_y),
        16: (cx + hip_dx + 6, ankle_y),
    }
    for idx, (x, y) in pts.items():
        kps[idx] = [x, y, 0.92]
    return kps


def _set_arm(kps: np.ndarray, side: str, wrist: Tuple[float, float], elbow: Optional[Tuple[float, float]] = None) -> None:
    sh_idx = 5 if side == "left" else 6
    el_idx = 7 if side == "left" else 8
    wr_idx = 9 if side == "left" else 10
    sh = kps[sh_idx, :2]
    wrist_arr = np.asarray(wrist, dtype=np.float32)
    if elbow is None:
        mid = (sh + wrist_arr) / 2.0
        vec = wrist_arr - sh
        perp = np.array([-vec[1], vec[0]], dtype=np.float32)
        n = float(np.linalg.norm(perp))
        if n > 1e-3:
            perp = perp / n * 12.0
        else:
            perp = np.zeros(2, dtype=np.float32)
        elbow_arr = mid + perp
    else:
        elbow_arr = np.asarray(elbow, dtype=np.float32)
    kps[el_idx, :2] = elbow_arr
    kps[wr_idx, :2] = wrist_arr


def _apply_signal_pose(kps: np.ndarray, code: str, t: float, rng: np.random.Generator) -> None:
    sh_l = kps[5, :2]
    sh_r = kps[6, :2]
    cx = float(np.mean(kps[[5, 6], 0]))
    head_y = float(kps[0, 1])
    sway = math.sin(t * 2.0) * 3.0

    if code == "stop":
        _set_arm(kps, "left", (sh_l[0] - 95, sh_l[1] + 2), (sh_l[0] - 50, sh_l[1]))
        _set_arm(kps, "right", (sh_r[0] + 95, sh_r[1] + 2), (sh_r[0] + 50, sh_r[1]))
    elif code == "emergency_stop":
        y = head_y - 45 + math.sin(t * 12.0) * 4.0
        _set_arm(kps, "left", (cx + 48, y), (sh_l[0] - 25, sh_l[1] - 35))
        _set_arm(kps, "right", (cx - 48, y), (sh_r[0] + 25, sh_r[1] - 35))
    elif code == "hoist":
        y = sh_r[1] - 115 + math.sin(t * 4.0) * 8.0
        _set_arm(kps, "left", (sh_l[0] - 10 + sway, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (sh_r[0] + 8, y), (sh_r[0] + 4, sh_r[1] - 55))
    elif code == "lower":
        y = sh_r[1] + 125 + math.sin(t * 4.0) * 8.0
        _set_arm(kps, "left", (sh_l[0] - 10 + sway, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (sh_r[0] + 8, y), (sh_r[0] + 4, sh_r[1] + 60))
    elif code == "travel_left":
        xoff = math.sin(t * 5.0) * 5.0
        _set_arm(kps, "left", (sh_l[0] - 100 - xoff, sh_l[1] + 10), (sh_l[0] - 55, sh_l[1] + 5))
        _set_arm(kps, "right", (sh_r[0] - 125 - xoff, sh_r[1] + 12), (sh_r[0] - 55, sh_r[1] + 5))
    elif code == "travel_right":
        xoff = math.sin(t * 5.0) * 5.0
        _set_arm(kps, "left", (sh_l[0] + 125 + xoff, sh_l[1] + 12), (sh_l[0] + 55, sh_l[1] + 5))
        _set_arm(kps, "right", (sh_r[0] + 100 + xoff, sh_r[1] + 10), (sh_r[0] + 55, sh_r[1] + 5))
    elif code == "boom_up":
        _set_arm(kps, "left", (sh_l[0] - 10 + sway, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (sh_r[0] + 80, sh_r[1] - 95 + math.sin(t * 4.0) * 6), (sh_r[0] + 45, sh_r[1] - 45))
    elif code == "boom_down":
        _set_arm(kps, "left", (sh_l[0] - 10 + sway, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (sh_r[0] + 80, sh_r[1] + 95 + math.sin(t * 4.0) * 6), (sh_r[0] + 45, sh_r[1] + 45))
    elif code == "dog_down":
        hip_l = kps[11, :2]
        hip_r = kps[12, :2]
        _set_arm(kps, "left", (hip_l[0] - 5, hip_l[1] + 15), (sh_l[0] - 18, sh_l[1] + 65))
        _set_arm(kps, "right", (hip_r[0] + 5, hip_r[1] + 15), (sh_r[0] + 18, sh_r[1] + 65))
    else:
        _set_arm(kps, "left", (sh_l[0] - 10 + sway, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (sh_r[0] + 10 - sway, sh_r[1] + 100), (sh_r[0] + 12, sh_r[1] + 50))


def _apply_walk_pose(kps: np.ndarray, t: float, rng: np.random.Generator) -> None:
    stride = math.sin(t * 6.0)
    kps[13, :2] = [kps[11, 0] - 4 + stride * 18, kps[11, 1] + 55 - abs(stride) * 8]
    kps[14, :2] = [kps[12, 0] + 4 - stride * 18, kps[12, 1] + 55 - abs(stride) * 8]
    kps[15, :2] = [kps[13, 0] - 2 + stride * 25, kps[13, 1] + 55]
    kps[16, :2] = [kps[14, 0] + 2 - stride * 25, kps[14, 1] + 55]
    _set_arm(kps, "left", (kps[5, 0] - 15 - stride * 25, kps[5, 1] + 90), (kps[5, 0] - 12, kps[5, 1] + 45))
    _set_arm(kps, "right", (kps[6, 0] + 15 + stride * 25, kps[6, 1] + 90), (kps[6, 0] + 12, kps[6, 1] + 45))


def _apply_gesture_pose(kps: np.ndarray, t: float, rng: np.random.Generator) -> None:
    sh_l = kps[5, :2]
    sh_r = kps[6, :2]
    cx = float(np.mean(kps[[5, 6], 0]))
    head_y = float(kps[0, 1])
    g = int((t * 0.7) % 3)
    if g == 0:
        _set_arm(kps, "left", (sh_l[0] - 10, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (sh_r[0] + 45, sh_r[1] - 70 + math.sin(t * 10.0) * 15), (sh_r[0] + 30, sh_r[1] - 25))
    elif g == 1:
        _set_arm(kps, "left", (sh_l[0] - 10, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (sh_r[0] + 110, sh_r[1] - 10), (sh_r[0] + 55, sh_r[1] - 5))
    else:
        _set_arm(kps, "left", (sh_l[0] - 10, sh_l[1] + 100), (sh_l[0] - 12, sh_l[1] + 50))
        _set_arm(kps, "right", (cx + 20, head_y - 10), (sh_r[0] + 45, sh_r[1] - 35))


def _synthesize_signaller_pose(
    code: str,
    segment_type: str,
    t: float,
    rng: np.random.Generator,
    width: int,
    height: int,
    visibility_factor: float = 1.0,
) -> np.ndarray:
    kps = _base_skeleton(width, height)
    if segment_type == "walking":
        _apply_walk_pose(kps, t, rng)
    elif segment_type == "gesture":
        _apply_gesture_pose(kps, t, rng)
    else:
        _apply_signal_pose(kps, code if code in SIGNALS else "idle", t, rng)

    kps[:, 0] += math.sin(t * 0.8) * 2.0
    kps[:, 1] += math.cos(t * 0.6) * 1.0
    noise_scale = 1.0 + 2.0 * (1.0 - clamp(visibility_factor, 0.0, 1.0))
    kps[:, :2] += rng.normal(0.0, 1.1 * noise_scale, size=(17, 2)).astype(np.float32)
    conf = 0.90 * clamp(visibility_factor, 0.0, 1.0) + rng.normal(0.0, 0.035, size=17)
    kps[:, 2] = np.clip(conf, 0.03, 0.99)
    return kps.astype(np.float32)


def _synthesize_distractor_pose(
    tid: int,
    t: float,
    width: int,
    height: int,
    rng: np.random.Generator,
    xshift: float,
) -> np.ndarray:
    kps = _base_skeleton(width, height)
    kps[:, 0] += xshift
    _apply_walk_pose(kps, t + tid * 0.31, rng)
    kps[:, 2] = np.clip(0.72 + rng.normal(0.0, 0.05, size=17), 0.15, 0.90)
    return kps.astype(np.float32)


def _bbox_from_keypoints(kps: np.ndarray, width: int, height: int, margin: int = 20) -> Tuple[int, int, int, int]:
    valid = kps[kps[:, 2] > 0.10]
    if valid.size == 0:
        return 0, 0, 1, 1
    x1 = int(max(0, np.min(valid[:, 0]) - margin))
    y1 = int(max(0, np.min(valid[:, 1]) - margin))
    x2 = int(min(width - 1, np.max(valid[:, 0]) + margin))
    y2 = int(min(height - 1, np.max(valid[:, 1]) + margin))
    return x1, y1, x2, y2


def _segment_at(t: float, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    for seg in segments:
        if float(seg["start"]) <= t < float(seg["end"]):
            return seg
    return segments[-1]


@st.cache_resource(show_spinner="Loading realistic synthetic site demo...")
def get_demo_dataset(duration_s: float = 45.0, seed: int = 7) -> Dict[str, Any]:
    fps = int(CONFIG["camera"]["fps"])
    width = int(CONFIG["camera"]["width"])
    height = int(CONFIG["camera"]["height"])
    window_size = int(CONFIG["inference"]["window_size"])
    stride = int(CONFIG["inference"]["stride"])
    rng = np.random.default_rng(seed)

    segments = [
        {"start": 0.0, "end": 3.5, "signal": "idle", "type": "signal"},
        {"start": 3.5, "end": 6.5, "signal": "hoist", "type": "signal"},
        {"start": 6.5, "end": 8.0, "signal": "idle", "type": "signal"},
        {"start": 8.0, "end": 11.0, "signal": "travel_right", "type": "signal"},
        {"start": 11.0, "end": 13.0, "signal": "stop", "type": "signal"},
        {"start": 13.0, "end": 16.0, "signal": "lower", "type": "signal"},
        {"start": 16.0, "end": 18.5, "signal": "idle", "type": "walking"},
        {"start": 18.5, "end": 20.5, "signal": "idle", "type": "gesture"},
        {"start": 20.5, "end": 23.5, "signal": "boom_up", "type": "signal"},
        {"start": 23.5, "end": 25.5, "signal": "emergency_stop", "type": "signal"},
        {"start": 25.5, "end": 28.5, "signal": "travel_left", "type": "signal"},
        {"start": 28.5, "end": 30.5, "signal": "idle", "type": "lost"},
        {"start": 30.5, "end": 33.5, "signal": "stop", "type": "fault_stop_moving"},
        {"start": 33.5, "end": 36.5, "signal": "boom_down", "type": "signal"},
        {"start": 36.5, "end": 38.5, "signal": "idle", "type": "gesture"},
        {"start": 38.5, "end": 40.5, "signal": "idle", "type": "low_visibility"},
        {"start": 40.5, "end": float(duration_s), "signal": "idle", "type": "signal"},
    ]

    frames: List[Dict[str, Any]] = []
    gt_seq: List[np.ndarray] = []
    vx = vy = 0.0
    boom_angle = 45.0
    moving_time = 0.0
    total_frames = int(duration_s * fps)

    for i in range(total_frames):
        t = i / float(fps)
        seg = _segment_at(t, segments)
        code = str(seg["signal"])
        seg_type = str(seg["type"])
        visibility_factor = 1.0
        detections: List[Dict[str, Any]] = []

        if seg_type == "lost":
            signaller_present = False
            kps = np.zeros((17, 3), dtype=np.float32)
        else:
            signaller_present = True
            if seg_type == "low_visibility":
                visibility_factor = 0.22
            kps = _synthesize_signaller_pose(
                code=code,
                segment_type=seg_type if seg_type in ("walking", "gesture") else "signal",
                t=t,
                rng=rng,
                width=width,
                height=height,
                visibility_factor=visibility_factor,
            )
            bbox = _bbox_from_keypoints(kps, width, height)
            detections.append(
                {
                    "track_id": 101,
                    "bbox": bbox,
                    "keypoints": kps,
                    "vest_color": "orange",
                    "score": float(0.93 * visibility_factor + 0.05),
                }
            )
        gt_seq.append(kps.astype(np.float32))

        for tid, color, xshift in [(201, "yellow", -220.0), (302, "blue", 260.0)]:
            dkps = _synthesize_distractor_pose(tid, t, width, height, rng, xshift)
            dbbox = _bbox_from_keypoints(dkps, width, height)
            detections.append(
                {
                    "track_id": tid,
                    "bbox": dbbox,
                    "keypoints": dkps,
                    "vest_color": color,
                    "score": 0.72,
                }
            )

        motion_cmd = code if seg_type in ("signal", "fault_stop_moving") else "idle"
        target_vx = target_vy = 0.0
        boom_delta = 0.0
        decay = 0.18

        if seg_type == "lost":
            target_vx = 0.35
            decay = 0.10
        elif motion_cmd == "hoist":
            target_vy = 0.75
        elif motion_cmd == "lower":
            target_vy = -0.75
        elif motion_cmd == "travel_right":
            target_vx = 0.70
        elif motion_cmd == "travel_left":
            target_vx = -0.70
        elif motion_cmd == "boom_up":
            boom_delta = 8.0
        elif motion_cmd in ("boom_down", "dog_down"):
            boom_delta = -8.0
        elif motion_cmd == "stop":
            decay = 0.45
        elif motion_cmd == "emergency_stop":
            decay = 0.75

        if seg_type == "fault_stop_moving":
            target_vx = 0.55
            target_vy = 0.12
            boom_delta = 0.0
            decay = 0.04

        vx += (target_vx - vx) * decay
        vy += (target_vy - vy) * decay
        boom_angle = (boom_angle + boom_delta / float(fps)) % 360.0
        moving = math.hypot(vx, vy) > 0.05 or abs(boom_delta) > 0.05
        moving_time = moving_time + (1.0 / float(fps)) if moving else 0.0

        telemetry = {
            "t": t,
            "vx": float(vx),
            "vy": float(vy),
            "boom_angle": float(boom_angle),
            "boom_velocity": float(boom_delta),
            "load_velocity": float(math.hypot(vx, vy)),
            "moving": bool(moving),
            "moving_duration_s": float(moving_time),
            "source_ok": True,
        }

        true_signal = code if seg_type in ("signal", "fault_stop_moving") else "idle"
        frames.append(
            {
                "frame_idx": i,
                "t": t,
                "detections": detections,
                "telemetry": telemetry,
                "true_signal": true_signal,
                "segment_type": seg_type,
                "signal_code": code,
            }
        )

    seq = np.stack(gt_seq, axis=0).astype(np.float32)
    norm_seq = normalize_sequence(seq)
    windows: List[Dict[str, Any]] = []
    for start, win_norm in iter_windows(norm_seq, window_size=window_size, stride=stride):
        mid_t = (start + win_norm.shape[0] / 2.0) / float(fps)
        seg = _segment_at(mid_t, segments)
        true_label = str(seg["signal"]) if seg["type"] in ("signal", "fault_stop_moving") else "idle"
        raw_win = seq[start:start + win_norm.shape[0]]
        vis = float(np.mean([visibility_score(f) for f in raw_win])) if raw_win.size else 0.0
        windows.append(
            {
                "start": int(start),
                "t": float(mid_t),
                "norm_window": win_norm.astype(np.float32),
                "true_signal": true_label,
                "segment_type": str(seg["type"]),
                "visibility": vis,
            }
        )

    return {
        "frames": frames,
        "windows": windows,
        "segments": segments,
        "fps": fps,
        "duration_s": float(duration_s),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _draw_skeleton_bgr(
    img: np.ndarray,
    kps: np.ndarray,
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    radius: int = 3,
    min_conf: float = 0.15,
) -> None:
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[0] < 17:
        return
    for i, j in CONNECTIONS:
        if kps[i, 2] > min_conf and kps[j, 2] > min_conf:
            pt1 = (int(kps[i, 0]), int(kps[i, 1]))
            pt2 = (int(kps[j, 0]), int(kps[j, 1]))
            cv2.line(img, pt1, pt2, color_bgr, thickness, cv2.LINE_AA)
    for x, y, c in kps:
        if c > min_conf:
            cv2.circle(img, (int(x), int(y)), radius, color_bgr, -1, cv2.LINE_AA)


def _vest_color_bgr(name: str) -> Tuple[int, int, int]:
    return rgb_to_bgr(VEST_COLORS_RGB.get(str(name), VEST_COLORS_RGB["unknown"]))


def render_frame_bgr(
    frame_data: Dict[str, Any],
    signaller_det: Optional[Dict[str, Any]],
    prediction: Dict[str, Any],
    visibility: float,
    alerts: List[Dict[str, Any]],
    base_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    width = int(CONFIG["camera"]["width"])
    height = int(CONFIG["camera"]["height"])
    if base_frame is not None:
        img = base_frame.copy()
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (64, 48, 38)
        cv2.rectangle(img, (0, int(height * 0.78)), (width, height), (42, 42, 48), -1)

    telemetry = frame_data.get("telemetry", {})
    mast_x = int(width * 0.82)
    jib_y = int(height * 0.18)
    hook_x = int(mast_x - 180 + float(telemetry.get("vx", 0.0)) * 120)
    hook_y = int(jib_y + 120 - float(telemetry.get("vy", 0.0)) * 120)
    cv2.line(img, (mast_x, height), (mast_x, jib_y), (80, 80, 90), 6)
    cv2.line(img, (mast_x, jib_y), (mast_x - 220, jib_y), (80, 80, 90), 5)
    cv2.line(img, (hook_x, jib_y), (hook_x, hook_y), (100, 100, 110), 2)
    cv2.rectangle(img, (hook_x - 18, hook_y), (hook_x + 18, hook_y + 28), (90, 90, 100), -1)

    for det in frame_data.get("detections", []):
        is_signaller = signaller_det is not None and int(det.get("track_id", -1)) == int(signaller_det.get("track_id", -2))
        color = _vest_color_bgr(det.get("vest_color", "unknown"))
        _draw_skeleton_bgr(img, det.get("keypoints"), color, 3 if is_signaller else 1, 4 if is_signaller else 2)
        if is_signaller:
            x1, y1, x2, y2 = det.get("bbox", (0, 0, 1, 1))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(img, "SIGNALLER", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)

    if signaller_det is None:
        cv2.putText(img, "NO SIGNALLER LOCKED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    pred_code = str(prediction.get("code", "idle"))
    sig_color = rgb_to_bgr(SIGNALS.get(pred_code, {}).get("color", [200, 200, 200]))
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (390, 188), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.rectangle(img, (10, 10), (390, 188), sig_color, 2)

    vis = float(np.nan_to_num(visibility, nan=0.0))
    lines = [
        f"SIGNAL: {label_for(pred_code)}",
        f"CONF: {float(prediction.get('confidence', 0.0)):.2f}",
        f"VIS: {vis:.2f}",
        f"CRANE: {'MOVING' if telemetry.get('moving') else 'STOPPED'}",
        f"vx={float(telemetry.get('vx', 0.0)):.2f} vy={float(telemetry.get('vy', 0.0)):.2f}",
    ]
    if pred_code == "unknown":
        lines.append("ACTION: RECOMMEND STOP")
    if pred_code == "emergency_stop":
        lines.append("RELAY: REQUEST STOP LOGGED")
    if alerts:
        lines.append(f"ALERTS: {len(alerts)}")

    y = 40
    for line in lines[:8]:
        cv2.putText(img, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 22

    if vis < float(CONFIG["inference"]["visibility_threshold"]):
        cv2.rectangle(img, (2, 2), (width - 3, height - 3), (0, 255, 255), 3)
        cv2.putText(img, "CHECK CAMERA PLACEMENT", (22, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return img


def render_frame_rgb(
    frame_data: Dict[str, Any],
    signaller_det: Optional[Dict[str, Any]],
    prediction: Dict[str, Any],
    visibility: float,
    alerts: List[Dict[str, Any]],
    base_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    bgr = render_frame_bgr(frame_data, signaller_det, prediction, visibility, alerts, base_frame)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_demo_video(dataset: Dict[str, Any], seconds: float = 8.0) -> Tuple[Optional[bytes], str]:
    fps = int(dataset["fps"])
    n = min(len(dataset["frames"]), int(seconds * fps))
    if n <= 0:
        return None, ""

    tmp_path: Optional[str] = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (int(CONFIG["camera"]["width"]), int(CONFIG["camera"]["height"])))
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter could not open MP4 codec.")

        for i in range(n):
            fd = dataset["frames"][i]
            signaller = next((d for d in fd["detections"] if int(d["track_id"]) == 101), None)
            vis = visibility_score(signaller["keypoints"]) if signaller else 0.0
            pred = {"code": fd["true_signal"], "confidence": 0.86}
            img_bgr = render_frame_bgr(fd, signaller, pred, vis, [])
            writer.write(img_bgr)
        writer.release()

        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        return data, "video/mp4"
    except Exception as exc:
        logger.warning("MP4 demo render failed: %s. Trying GIF fallback.", exc)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    try:
        pil_frames = []
        for i in range(min(n, 60)):
            fd = dataset["frames"][i]
            signaller = next((d for d in fd["detections"] if int(d["track_id"]) == 101), None)
            vis = visibility_score(signaller["keypoints"]) if signaller else 0.0
            pred = {"code": fd["true_signal"], "confidence": 0.86}
            img_rgb = render_frame_rgb(fd, signaller, pred, vis, [])
            pil_frames.append(Image.fromarray(img_rgb).resize((480, 270)))
        buf = io.BytesIO()
        pil_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / max(fps, 1)),
            loop=0,
        )
        return buf.getvalue(), "image/gif"
    except Exception as exc:
        logger.warning("GIF demo render failed: %s", exc)
        return None, ""


def camera_guide_image() -> np.ndarray:
    w, h = 720, 420
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (35, 35, 40)
    cv2.rectangle(img, (0, int(h * 0.72)), (w, h), (55, 55, 60), -1)
    cv2.circle(img, (90, 150), 22, (255, 200, 0), -1)
    cv2.putText(img, "CAMERA", (55, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (110, 150), (430, 90), (0, 255, 255), 1)
    cv2.line(img, (110, 150), (430, 330), (0, 255, 255), 1)
    kps = _base_skeleton(720, 420)
    kps[:, 0] += 180
    _draw_skeleton_bgr(img, kps, (0, 140, 255), 2, 3)
    cv2.rectangle(img, (390, 80), (690, 340), (0, 255, 0), 2)
    cv2.putText(img, "SIGNALLER FULL BODY", (405, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Recommended: 4-8 m, 1.5-2.2 m height, arms unobstructed", (35, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    preds: List[str],
    truths: List[str],
    classes: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    class_set = set(classes)
    preds = [p if p in class_set else "unknown" for p in preds]
    truths = [t if t in class_set else "unknown" for t in truths]
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for p, t in zip(preds, truths):
        cm[idx[t], idx[p]] += 1

    rows = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        support = int(np.sum(cm[i, :]))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "signal": c,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    df = pd.DataFrame(rows)
    if not df.empty:
        df.loc[len(df)] = {
            "signal": "OVERALL_ACCURACY",
            "precision": acc,
            "recall": acc,
            "f1": acc,
            "support": total,
            "tp": int(np.trace(cm)),
            "fp": int(total - np.trace(cm)),
            "fn": int(total - np.trace(cm)),
        }
    confusion_df = pd.DataFrame(cm, index=classes, columns=classes)
    return df, confusion_df, cm


def compute_false_trigger_rates(preds: List[str], segment_types: List[str]) -> pd.DataFrame:
    rows = []
    for stype in ("walking", "gesture"):
        total = sum(1 for s in segment_types if s == stype)
        false = sum(1 for p, s in zip(preds, segment_types) if s == stype and is_active_signal(p))
        rate = false / total if total > 0 else 0.0
        rows.append(
            {
                "segment_type": stype,
                "windows": total,
                "false_active_signals": false,
                "false_trigger_rate": rate,
            }
        )
    return pd.DataFrame(rows)


def compute_latency_report(latencies_ms: List[float]) -> pd.DataFrame:
    if not latencies_ms:
        return pd.DataFrame([{"metric": "latency_ms_mean", "value": 0.0}])
    arr = np.asarray(latencies_ms, dtype=np.float32)
    mean = float(np.mean(arr))
    return pd.DataFrame(
        [
            {"metric": "latency_ms_mean", "value": mean},
            {"metric": "latency_ms_p50", "value": float(np.percentile(arr, 50))},
            {"metric": "latency_ms_p95", "value": float(np.percentile(arr, 95))},
            {"metric": "latency_ms_p99", "value": float(np.percentile(arr, 99))},
            {"metric": "latency_ms_max", "value": float(np.max(arr))},
            {"metric": "classifier_fps_estimate", "value": float(1000.0 / mean) if mean > 0 else 0.0},
        ]
    )


def evaluate_dataset(
    classifier: Any,
    dataset: Dict[str, Any],
    conf_threshold: float,
    unclear_threshold: float,
    visibility_threshold: float,
) -> Dict[str, Any]:
    classes = signal_labels()
    preds: List[str] = []
    truths: List[str] = []
    segtypes: List[str] = []
    latencies: List[float] = []
    confs: List[float] = []
    times: List[float] = []

    for win in dataset.get("windows", []):
        raw_code, raw_conf, _ = classifier.predict_window(
            win["norm_window"],
            visibility=float(win["visibility"]),
            has_signaller=float(win["visibility"]) > 0.05,
            conf_threshold=conf_threshold,
            unclear_threshold=unclear_threshold,
            visibility_threshold=visibility_threshold,
        )
        final_code, final_conf = apply_open_set(
            raw_code,
            raw_conf,
            float(win["visibility"]),
            conf_threshold,
            unclear_threshold,
            visibility_threshold,
        )
        preds.append(final_code)
        truths.append(str(win["true_signal"]))
        segtypes.append(str(win["segment_type"]))
        latencies.append(float(getattr(classifier, "last_latency_ms", 0.0)))
        confs.append(float(final_conf))
        times.append(float(win["t"]))

    per_signal, confusion_df, cm = compute_classification_metrics(preds, truths, classes)
    false_df = compute_false_trigger_rates(preds, segtypes)
    latency_df = compute_latency_report(latencies)
    predictions_df = pd.DataFrame(
        {
            "t": times,
            "predicted": preds,
            "true": truths,
            "confidence": confs,
            "segment_type": segtypes,
            "latency_ms": latencies,
        }
    )
    return {
        "per_signal": per_signal,
        "confusion_df": confusion_df,
        "confusion_matrix": cm,
        "false_trigger": false_df,
        "latency": latency_df,
        "predictions": predictions_df,
        "classes": classes,
    }


def plot_confusion_matrix(cm: np.ndarray, classes: List[str]) -> Any:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = int(cm[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="black" if val < cm.max() / 2 else "white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Signal confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _save_upload_to_temp(upload: Any, suffix: Optional[str] = None) -> Path:
    if suffix is None:
        suffix = Path(getattr(upload, "name", "upload.bin")).suffix or ".bin"
    path = Path(tempfile.gettempdir()) / f"signalguard_{int(time.time() * 1000)}_{os.getpid()}{suffix}"
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path


def _label_at_time(t: float, labels_df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    row = labels_df[(labels_df["start_time"] <= t) & (labels_df["end_time"] > t)]
    if row.empty:
        return None
    r = row.iloc[0]
    label = str(r["label"])
    segment_type = str(r.get("segment_type", "signal"))
    return label, segment_type


def evaluate_custom_footage(
    video_path: Path,
    labels_path: Path,
    classifier: Any,
    thresholds: Tuple[float, float, float],
    vest_color: str,
) -> Dict[str, Any]:
    try:
        labels_df = pd.read_csv(labels_path)
    except Exception as exc:
        return {"error": f"Could not read labels CSV: {exc}"}

    required = {"start_time", "end_time", "label"}
    if not required.issubset(labels_df.columns):
        return {"error": "Labels CSV must contain columns: start_time,end_time,label[,segment_type]."}

    pose_model = load_pose_model()
    if pose_model is None:
        return {"error": "Pose model unavailable. Install requirements and/or allow model download, or use synthetic demo evaluation."}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = float(CONFIG["camera"]["fps"])

    window_size = int(CONFIG["inference"]["window_size"])
    stride = int(CONFIG["inference"]["stride"])
    conf_threshold, unclear_threshold, visibility_threshold = thresholds

    tracker = SignallerTracker(vest_color=vest_color)
    local_iou_tracker = SimpleIoUTracker()
    buffer: List[np.ndarray] = []
    preds: List[str] = []
    truths: List[str] = []
    segtypes: List[str] = []
    latencies: List[float] = []
    confs: List[float] = []
    times: List[float] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / float(fps)
        detections = yolo_detections(frame, pose_model, local_iou_tracker)
        signaller, _ = tracker.update(detections)
        vis = visibility_score(signaller["keypoints"]) if signaller else 0.0
        norm = normalize_keypoints(signaller["keypoints"]) if signaller else np.zeros((17, 2), dtype=np.float32)
        buffer.append(norm)
        if len(buffer) > window_size:
            buffer.pop(0)

        if len(buffer) >= window_size and frame_idx % stride == 0:
            window = np.stack(buffer[-window_size:], axis=0)
            raw_code, raw_conf, _ = classifier.predict_window(
                window,
                visibility=vis,
                has_signaller=signaller is not None,
                conf_threshold=conf_threshold,
                unclear_threshold=unclear_threshold,
                visibility_threshold=visibility_threshold,
            )
            final_code, final_conf = apply_open_set(
                raw_code,
                raw_conf,
                vis,
                conf_threshold,
                unclear_threshold,
                visibility_threshold,
            )
            lab = _label_at_time(t, labels_df)
            if lab is not None:
                label, segment_type = lab
                if label not in SIGNALS:
                    label = "unknown"
                preds.append(final_code)
                truths.append(label)
                segtypes.append(segment_type)
                latencies.append(float(getattr(classifier, "last_latency_ms", 0.0)))
                confs.append(float(final_conf))
                times.append(float(t))

        frame_idx += 1

    cap.release()
    if not preds:
        return {"error": "No labeled windows were produced. Check label timestamps and video length."}

    classes = signal_labels()
    per_signal, confusion_df, cm = compute_classification_metrics(preds, truths, classes)
    false_df = compute_false_trigger_rates(preds, segtypes)
    latency_df = compute_latency_report(latencies)
    predictions_df = pd.DataFrame(
        {
            "t": times,
            "predicted": preds,
            "true": truths,
            "confidence": confs,
            "segment_type": segtypes,
            "latency_ms": latencies,
        }
    )
    return {
        "per_signal": per_signal,
        "confusion_df": confusion_df,
        "confusion_matrix": cm,
        "false_trigger": false_df,
        "latency": latency_df,
        "predictions": predictions_df,
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# Audit PDF export
# ---------------------------------------------------------------------------

def _xml_escape(value: Any) -> str:
    s = str(value)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_audit_pdf(audit_df: pd.DataFrame, alerts_df: pd.DataFrame, summary: Dict[str, Any]) -> Optional[bytes]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:
        logger.warning("Reportlab unavailable: %s. PDF export disabled.", exc)
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SignalGuard Audit Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(_xml_escape(CONFIG["app"]["disclaimer"]), styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"Generated: {_xml_escape(utc_now())}", styles["Normal"]),
        Paragraph(f"Records: {_xml_escape(summary.get('records', 0))}", styles["Normal"]),
        Paragraph(f"Alerts: {_xml_escape(summary.get('alerts', 0))}", styles["Normal"]),
        Paragraph(f"Emergency relay events: {_xml_escape(summary.get('relay', 0))}", styles["Normal"]),
        Spacer(1, 14),
    ]

    if not alerts_df.empty:
        story.append(Paragraph("Alerts", styles["Heading2"]))
        cols = [c for c in ["timestamp", "severity", "code", "message"] if c in alerts_df.columns]
        df = alerts_df[cols].head(80).copy()
        if "message" in df.columns:
            df["message"] = df["message"].astype(str).str.slice(0, 120)
        data = [cols] + df.astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 14))

    if not audit_df.empty:
        story.append(Paragraph("Recognized signal audit", styles["Heading2"]))
        cols = [c for c in ["timestamp", "predicted", "confidence", "true_signal", "visibility", "vx", "vy", "moving"] if c in audit_df.columns]
        df = audit_df[cols].tail(120).copy()
        data = [cols] + df.astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(table)

    try:
        doc.build(story)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("PDF build failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Unit tests for core state machines, geometry, and scoring
# ---------------------------------------------------------------------------

def run_unit_tests() -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    def check(name: str, fn: Any) -> None:
        try:
            fn()
            results.append({"test": name, "status": "PASS", "detail": ""})
        except AssertionError as exc:
            results.append({"test": name, "status": "FAIL", "detail": str(exc)})
        except Exception as exc:
            results.append({"test": name, "status": "ERROR", "detail": repr(exc)})

    def test_normalize_center_neck() -> None:
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[5] = [100, 200, 0.9]
        kp[6] = [140, 200, 0.9]
        kp[11] = [110, 300, 0.9]
        kp[12] = [130, 300, 0.9]
        norm = normalize_keypoints(kp)
        neck = neck_point(kp)
        assert abs(float(norm[5, 0] - (100 - neck[0]) / 100.0)) < 1e-3
        assert abs(float(norm[11, 1] - 1.0)) < 1e-3

    def test_mirror_left_right() -> None:
        seq = np.zeros((2, 17, 2), dtype=np.float32)
        seq[:, 9, 0] = 1.0
        mirrored = mirror_keypoints(seq)
        assert abs(float(mirrored[0, 10, 0]) + 1.0) < 1e-6
        assert abs(float(mirrored[0, 9, 0])) < 1e-6
        assert mirror_label("travel_left") == "travel_right"

    def test_visibility_score() -> None:
        good = np.ones((17, 3), dtype=np.float32) * 0.9
        bad = np.ones((17, 3), dtype=np.float32) * 0.05
        assert visibility_score(good) > 0.75
        assert visibility_score(bad) < 0.25

    def test_tracker_reacquire() -> None:
        tracker = SignallerTracker(vest_color="orange", max_lost_frames=2)
        kp = np.zeros((17, 3), dtype=np.float32)
        det1 = {"track_id": 1, "bbox": (0, 0, 50, 100), "keypoints": kp, "vest_color": "orange", "score": 0.9}
        det2 = {"track_id": 2, "bbox": (10, 0, 60, 100), "keypoints": kp, "vest_color": "orange", "score": 0.9}
        out, status = tracker.update([det1])
        assert out is not None and int(out["track_id"]) == 1
        tracker.update([])
        tracker.update([])
        out, status = tracker.update([det2])
        assert out is not None and int(out["track_id"]) == 2
        assert status in ("reacquired", "locked")

    def test_open_set_rejection() -> None:
        code, _ = apply_open_set("hoist", 0.50, 0.90, 0.62, 0.45, 0.45)
        assert code == "unknown"
        code, _ = apply_open_set("hoist", 0.80, 0.30, 0.62, 0.45, 0.45)
        assert code == "unknown"

    def test_windows() -> None:
        seq = np.zeros((136, 17, 2), dtype=np.float32)
        wins = list(iter_windows(seq, window_size=64, stride=8))
        assert len(wins) == 10
        assert wins[0][1].shape == (64, 17, 2)

    def test_stop_conflict_alert() -> None:
        alerts = evaluate_alerts(
            {"code": "stop", "confidence": 0.9},
            {"moving": True, "moving_duration_s": 1.5, "vx": 0.1, "vy": 0.0, "source_ok": True},
            "locked",
            0.9,
            {},
            time.time(),
            [],
            0.45,
        )
        assert any(a["code"] == "STOP_CONFLICT" for a in alerts)

    def test_heuristic_idle_vs_stop() -> None:
        clf = HeuristicSignalClassifier()
        idle = np.zeros((64, 17, 2), dtype=np.float32)
        idle[:, 9, :] = [-0.18, 1.10]
        idle[:, 10, :] = [0.18, 1.10]
        idle[:, 7, :] = [-0.22, 0.55]
        idle[:, 8, :] = [0.22, 0.55]
        code, conf, _ = clf.predict_window(idle, visibility=0.95, has_signaller=True)
        assert code in ("idle", "unknown")

        stop = np.zeros((64, 17, 2), dtype=np.float32)
        stop[:, 9, :] = [-1.05, 0.02]
        stop[:, 10, :] = [1.05, 0.02]
        stop[:, 7, :] = [-0.55, 0.02]
        stop[:, 8, :] = [0.55, 0.02]
        code2, conf2, _ = clf.predict_window(stop, visibility=0.95, has_signaller=True)
        assert code2 == "stop"

    check("normalize_center_neck", test_normalize_center_neck)
    check("mirror_left_right", test_mirror_left_right)
    check("visibility_score", test_visibility_score)
    check("tracker_reacquire", test_tracker_reacquire)
    check("open_set_rejection", test_open_set_rejection)
    check("window_stride", test_windows)
    check("stop_conflict_alert", test_stop_conflict_alert)
    check("heuristic_idle_vs_stop", test_heuristic_idle_vs_stop)
    return results


# ---------------------------------------------------------------------------
# Streamlit session and processing
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    ss_setdefault("playing", True)
    ss_setdefault("tick", 0)
    ss_setdefault("processed_frames", 0)
    ss_setdefault("speed", 1.0)
    ss_setdefault("source", "Synthetic site demo")
    ss_setdefault("vest_color", "orange")
    ss_setdefault("conf_threshold", float(CONFIG["inference"]["conf_threshold"]))
    ss_setdefault("unclear_threshold", float(CONFIG["inference"]["unclear_threshold"]))
    ss_setdefault("visibility_threshold", float(CONFIG["inference"]["visibility_threshold"]))
    ss_setdefault("pose_buffer", [])
    ss_setdefault("history", [])
    ss_setdefault("audit", [])
    ss_setdefault("alerts", [])
    ss_setdefault("alert_cooldowns", {})
    ss_setdefault("relay_log", [])
    ss_setdefault("training_samples", [])
    ss_setdefault("force_step", False)
    ss_setdefault("last_visibility", 1.0)
    ss_setdefault("last_window", None)
    ss_setdefault("last_prediction", {"code": "idle", "confidence": 0.0, "raw_confidence": 0.0, "probs": {}})
    ss_setdefault("context", None)

    if "heuristic_classifier" not in st.session_state:
        st.session_state.heuristic_classifier = HeuristicSignalClassifier()
    if "classifier" not in st.session_state:
        st.session_state.classifier = st.session_state.heuristic_classifier
    if "tracker" not in st.session_state:
        st.session_state.tracker = SignallerTracker(st.session_state.vest_color)
    st.session_state.tracker.vest_color = st.session_state.vest_color


def reset_demo_state() -> None:
    st.session_state.tick = 0
    st.session_state.processed_frames = 0
    st.session_state.pose_buffer = []
    st.session_state.history = []
    st.session_state.audit = []
    st.session_state.alerts = []
    st.session_state.alert_cooldowns = {}
    st.session_state.relay_log = []
    st.session_state.last_prediction = {"code": "idle", "confidence": 0.0, "raw_confidence": 0.0, "probs": {}}
    st.session_state.last_window = None
    st.session_state.context = None
    st.session_state.tracker = SignallerTracker(st.session_state.vest_color)


def process_real_frame_if_available() -> Optional[Dict[str, Any]]:
    pose_model = load_pose_model()
    if pose_model is None:
        return None
    cap = get_or_create_capture()
    if cap is None or not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok:
        return None
    detections = yolo_detections(frame, pose_model)
    if "telemetry_sim" not in st.session_state:
        st.session_state.telemetry_sim = TelemetrySimulator()
    dt = 1.0 / float(CONFIG["camera"]["fps"])
    commanded = str(st.session_state.get("last_prediction", {}).get("code", "idle"))
    telemetry = st.session_state.telemetry_sim.step(dt, commanded)
    return {
        "frame_idx": int(st.session_state.processed_frames),
        "t": time.time(),
        "detections": detections,
        "telemetry": telemetry,
        "true_signal": None,
        "segment_type": "real",
        "signal_code": commanded,
        "frame": frame,
    }


def process_tick(demo: Dict[str, Any]) -> Dict[str, Any]:
    tracker = st.session_state.tracker
    tracker.vest_color = st.session_state.vest_color
    classifier = st.session_state.classifier

    effective_source = st.session_state.source
    base_frame = None
    if effective_source == "Synthetic site demo":
        idx = int(st.session_state.tick) % len(demo["frames"])
        frame_data = demo["frames"][idx]
    else:
        real = process_real_frame_if_available()
        if real is None:
            effective_source = "Synthetic site demo"
            idx = int(st.session_state.tick) % len(demo["frames"])
            frame_data = demo["frames"][idx]
        else:
            frame_data = real
            base_frame = real.get("frame")

    detections = frame_data.get("detections", [])
    signaller_det, tracker_status = tracker.update(detections)
    visibility = visibility_score(signaller_det["keypoints"]) if signaller_det else 0.0
    st.session_state.last_visibility = visibility

    if signaller_det is not None:
        norm = normalize_keypoints(signaller_det["keypoints"])
    else:
        norm = np.zeros((17, 2), dtype=np.float32)

    buf = st.session_state.pose_buffer
    buf.append(norm)
    window_size = int(CONFIG["inference"]["window_size"])
    stride = int(CONFIG["inference"]["stride"])
    if len(buf) > window_size:
        buf.pop(0)

    st.session_state.processed_frames += 1
    processed = int(st.session_state.processed_frames)
    have_full = len(buf) >= window_size
    first_window = st.session_state.last_window is None

    # Produce an immediate padded window on startup, then normal 64-frame windows.
    do_window = (
        (have_full and (processed % stride == 0 or first_window))
        or (first_window and len(buf) > 0)
    )

    if do_window:
        if have_full:
            window = np.stack(buf[-window_size:], axis=0)
        else:
            window = fit_length(np.stack(buf, axis=0), window_size)

        st.session_state.last_window = window.copy()
        raw_code, raw_conf, probs = classifier.predict_window(
            window,
            visibility=visibility,
            has_signaller=signaller_det is not None,
            conf_threshold=float(st.session_state.conf_threshold),
            unclear_threshold=float(st.session_state.unclear_threshold),
            visibility_threshold=float(st.session_state.visibility_threshold),
        )
        final_code, final_conf = apply_open_set(
            raw_code,
            raw_conf,
            visibility,
            float(st.session_state.conf_threshold),
            float(st.session_state.unclear_threshold),
            float(st.session_state.visibility_threshold),
        )
        prediction = {
            "code": final_code,
            "confidence": final_conf,
            "raw_confidence": raw_conf,
            "probs": probs,
        }
        st.session_state.last_prediction = prediction
    else:
        prediction = st.session_state.last_prediction

    telemetry = frame_data.get("telemetry", {})
    now_ts = time.time()
    new_alerts = evaluate_alerts(
        prediction,
        telemetry,
        tracker_status,
        visibility,
        st.session_state.alert_cooldowns,
        now_ts,
        st.session_state.relay_log,
        float(st.session_state.visibility_threshold),
    )
    if new_alerts:
        st.session_state.alerts = new_alerts + st.session_state.alerts
        st.session_state.alerts = st.session_state.alerts[: int(CONFIG["inference"]["max_alerts"])]

    history_record = {
        "clock": float(st.session_state.processed_frames) / float(CONFIG["camera"]["fps"]),
        "t": float(frame_data.get("t", 0.0)),
        "predicted": prediction.get("code", "idle"),
        "confidence": float(prediction.get("confidence", 0.0)),
        "true": frame_data.get("true_signal"),
        "segment_type": frame_data.get("segment_type"),
        "visibility": float(visibility),
        "vx": float(telemetry.get("vx", 0.0)),
        "vy": float(telemetry.get("vy", 0.0)),
        "moving": bool(telemetry.get("moving", False)),
        "moving_duration_s": float(telemetry.get("moving_duration_s", 0.0)),
        "alerts": len(new_alerts),
    }
    st.session_state.history.append(history_record)
    st.session_state.history = st.session_state.history[-int(CONFIG["inference"]["max_history"]):]

    if do_window or new_alerts:
        audit_record = {
            "timestamp": utc_now(),
            "clock": history_record["clock"],
            "predicted": history_record["predicted"],
            "confidence": history_record["confidence"],
            "true_signal": history_record["true"],
            "segment_type": history_record["segment_type"],
            "visibility": history_record["visibility"],
            "tracker_status": tracker_status,
            "vx": history_record["vx"],
            "vy": history_record["vy"],
            "moving": history_record["moving"],
            "new_alerts": ",".join(a["code"] for a in new_alerts),
        }
        st.session_state.audit.append(audit_record)
        st.session_state.audit = st.session_state.audit[-int(CONFIG["inference"]["max_audit"]):]

    image = render_frame_rgb(frame_data, signaller_det, prediction, visibility, st.session_state.alerts[:5], base_frame)
    st.session_state.tick += 1

    return {
        "image": image,
        "frame_data": frame_data,
        "signaller_det": signaller_det,
        "tracker_status": tracker_status,
        "prediction": prediction,
        "visibility": visibility,
        "telemetry": telemetry,
        "new_alerts": new_alerts,
        "alerts": st.session_state.alerts,
        "source": effective_source,
    }


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("SignalGuard")
    st.sidebar.warning(CONFIG["app"]["disclaimer"])

    st.sidebar.selectbox(
        "Input source",
        ["Synthetic site demo", "Uploaded video", "Webcam"],
        key="source",
    )

    if st.session_state.source == "Uploaded video":
        uploaded = st.sidebar.file_uploader(
            "Signaller/crane video",
            type=["mp4", "avi", "mov", "mkv"],
            key="uploaded_file_input",
        )
        st.session_state.uploaded_file = uploaded
    elif st.session_state.source == "Webcam":
        st.sidebar.info("Webcam uses local cv2.VideoCapture(0) and optional YOLO pose if installed.")

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Pause" if st.session_state.playing else "Play"):
        st.session_state.playing = not st.session_state.playing
    if c2.button("Step"):
        st.session_state.force_step = True

    st.sidebar.slider("Playback speed", 0.25, 3.0, float(st.session_state.speed), key="speed")
    st.sidebar.selectbox("Signaller vest/armband color", ["orange", "yellow", "green", "blue"], key="vest_color")
    st.sidebar.slider("Accept confidence", 0.30, 0.95, float(st.session_state.conf_threshold), key="conf_threshold")
    st.sidebar.slider("Unclear threshold", 0.10, 0.80, float(st.session_state.unclear_threshold), key="unclear_threshold")
    st.sidebar.slider("Visibility threshold", 0.10, 0.90, float(st.session_state.visibility_threshold), key="visibility_threshold")

    mode = st.sidebar.radio("Classifier", ["Heuristic fallback", "Trained transformer if available"], key="classifier_mode")
    if mode.startswith("Trained"):
        if "trained_classifier" not in st.session_state:
            extra = [st.session_state.get("trained_checkpoint_path")] if st.session_state.get("trained_checkpoint_path") else None
            st.session_state.trained_classifier = load_torch_classifier_from_paths(extra)
        if st.session_state.trained_classifier is not None:
            st.session_state.classifier = st.session_state.trained_classifier
            st.sidebar.success("Trained transformer loaded.")
        else:
            st.session_state.classifier = st.session_state.heuristic_classifier
            st.sidebar.warning("No usable checkpoint. Heuristic fallback active.")
    else:
        st.session_state.classifier = st.session_state.heuristic_classifier

    if st.sidebar.button("Reset demo state"):
        reset_demo_state()

    with st.sidebar.expander("Embedded config.yaml / signals.yaml"):
        st.json(CONFIG)
        try:
            import yaml

            cfg_text = yaml.safe_dump(CONFIG, sort_keys=False)
            st.download_button("Download config.yaml", cfg_text, file_name="config.yaml", mime="text/yaml")
        except Exception:
            st.download_button(
                "Download config.json",
                json.dumps(CONFIG, indent=2),
                file_name="config.json",
                mime="application/json",
            )


def render_timeline(history: List[Dict[str, Any]]) -> None:
    if not history:
        st.info("Waiting for timeline data...")
        return
    df = pd.DataFrame(history[-300:])
    codes = list(df["predicted"].unique())
    code_to_y = {c: i for i, c in enumerate(codes)}
    y_vals = df["predicted"].map(lambda c: code_to_y.get(c, 0))

    fig, ax = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    sc = ax[0].scatter(df["clock"], y_vals, c=df["confidence"], cmap="viridis", s=14)
    ax[0].set_yticks(range(len(codes)))
    ax[0].set_yticklabels(codes)
    ax[0].set_ylabel("Signal")
    ax[0].set_title("Recognized signal vs crane motion")
    fig.colorbar(sc, ax=ax[0], label="confidence")

    ax[1].plot(df["clock"], df["vx"], label="crane x velocity")
    ax[1].plot(df["clock"], df["vy"], label="load y velocity")
    ax[1].axhline(0, color="black", linewidth=0.6)
    ax[1].set_xlabel("clock (s)")
    ax[1].set_ylabel("velocity")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_live_tab(context: Dict[str, Any], demo: Dict[str, Any]) -> None:
    st.markdown("### Live decision-support view")
    st.caption(
        "The app automatically loads a realistic synthetic site demo and starts the simulation. "
        "Use sidebar controls to pause, step, tune thresholds, or switch source."
    )

    if "demo_video_payload" not in st.session_state:
        st.session_state.demo_video_payload = render_demo_video(demo, seconds=8.0)

    video_bytes, mime = st.session_state.demo_video_payload

    with st.expander("Auto-generated demo video", expanded=True):
        st_video_compat(video_bytes, mime)

    col_vid, col_hud = st.columns([2.1, 1])
    with col_vid:
        st_image_compat(context["image"])
        st.caption(f"Source: {context.get('source', 'unknown')} | Tracker: {context.get('tracker_status', 'unknown')}")

    with col_hud:
        pred = context["prediction"]
        telemetry = context["telemetry"]
        code = pred.get("code", "idle")
        conf = float(pred.get("confidence", 0.0))
        vis = float(context.get("visibility", 0.0))
        st.markdown(f"#### Recognized signal: `{label_for(code)}`")
        st.progress(clamp(conf, 0.0, 1.0))
        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{conf:.2f}")
        m2.metric("Visibility", f"{vis:.2f}")
        st.metric(
            "Crane state",
            "MOVING" if telemetry.get("moving") else "STOPPED",
            f"vx={float(telemetry.get('vx', 0)):.2f}, vy={float(telemetry.get('vy', 0)):.2f}",
        )
        if code == "unknown":
            st.error("UNCLEAR — recommend STOP and re-confirm with the signaller.")
        elif code == "emergency_stop":
            st.error("Emergency-stop request logged to relay. SignalGuard does not issue motion commands.")
        elif is_active_signal(code):
            st.success("Active signal recognized. Continue human-in-the-loop confirmation.")
        else:
            st.info("Idle/no command.")

    st.subheader("Recent alerts")
    alerts_df = pd.DataFrame(context.get("alerts", []))
    if alerts_df.empty:
        st.info("No active alerts.")
    else:
        st_dataframe_compat(alerts_df)
        if st.button("Acknowledge all alerts"):
            for a in st.session_state.alerts:
                a["acknowledged"] = True
            st.success("Alerts acknowledged.")

    st.subheader("Signal vs crane motion timeline")
    render_timeline(st.session_state.history)


def render_evaluation_tab(demo: Dict[str, Any]) -> None:
    st.markdown("### Evaluation")
    st.caption("Per-signal precision/recall, confusion matrix, latency, and false-trigger rate during walking and ordinary gestures.")

    if st.button("Run evaluation on synthetic demo"):
        with st.spinner("Evaluating demo windows..."):
            st.session_state.demo_eval = evaluate_dataset(
                st.session_state.classifier,
                demo,
                float(st.session_state.conf_threshold),
                float(st.session_state.unclear_threshold),
                float(st.session_state.visibility_threshold),
            )

    ev = st.session_state.get("demo_eval")
    if ev:
        st.subheader("Per-signal precision / recall")
        st_dataframe_compat(ev["per_signal"])

        st.subheader("Confusion matrix")
        st_dataframe_compat(ev["confusion_df"], hide_index=False)
        fig = plot_confusion_matrix(ev["confusion_matrix"], ev["classes"])
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Latency report")
        st_dataframe_compat(ev["latency"])

        st.subheader("False-trigger rate")
        st_dataframe_compat(ev["false_trigger"])

        st.download_button(
            "Download demo predictions CSV",
            ev["predictions"].to_csv(index=False).encode("utf-8"),
            file_name="signalguard_demo_predictions.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Evaluate your own footage")
    st.caption("CSV columns: start_time,end_time,label[,segment_type]. Labels must match embedded signals.yaml keys.")
    cv = st.file_uploader("Footage video", type=["mp4", "avi", "mov", "mkv"], key="custom_video_file")
    cl = st.file_uploader("Ground-truth labels CSV", type=["csv"], key="custom_labels_file")
    if st.button("Run custom-footage evaluation"):
        if cv is None or cl is None:
            st.warning("Upload both footage video and labels CSV.")
        else:
            vp = _save_upload_to_temp(cv)
            lp = _save_upload_to_temp(cl, ".csv")
            with st.spinner("Processing custom footage..."):
                res = evaluate_custom_footage(
                    vp,
                    lp,
                    st.session_state.classifier,
                    (
                        float(st.session_state.conf_threshold),
                        float(st.session_state.unclear_threshold),
                        float(st.session_state.visibility_threshold),
                    ),
                    st.session_state.vest_color,
                )
            if "error" in res:
                st.error(res["error"])
            else:
                st.session_state.custom_eval = res

    cev = st.session_state.get("custom_eval")
    if cev:
        st_dataframe_compat(cev["per_signal"])
        st_dataframe_compat(cev["confusion_df"], hide_index=False)
        fig = plot_confusion_matrix(cev["confusion_matrix"], cev["classes"])
        st.pyplot(fig)
        plt.close(fig)
        st_dataframe_compat(cev["latency"])
        st_dataframe_compat(cev["false_trigger"])


def render_training_tab() -> None:
    st.markdown("### Recording wizard and training")
    st.caption("Capture 64-frame windows from the live pipeline, then train a tiny temporal transformer. Augmentations: mirror with left/right relabeling, speed change, jitter.")

    options = [c for c in SIGNALS.keys() if c != "unknown"]
    selected = st.selectbox("Signal being demonstrated", options, key="record_signal")
    c1, c2, c3 = st.columns(3)
    if c1.button("Capture current window"):
        if st.session_state.last_window is None:
            st.warning("No full 64-frame window yet. Let the demo run for a moment.")
        else:
            st.session_state.training_samples.append(
                (
                    selected,
                    np.asarray(st.session_state.last_window, dtype=np.float32).copy(),
                    float(st.session_state.last_visibility),
                )
            )
            st.success(f"Captured {selected}. Total samples: {len(st.session_state.training_samples)}")

    if c2.button("Clear captured samples"):
        st.session_state.training_samples = []
        st.info("Training samples cleared.")

    if c3.button("Train tiny transformer"):
        with st.spinner("Training..."):
            path, msg = train_transformer_from_samples(st.session_state.training_samples)
        if path:
            st.session_state.trained_checkpoint_path = path
            st.session_state.trained_classifier = None
            st.success(msg)
        else:
            st.warning(msg)

    counts: Dict[str, int] = {}
    for lbl, _, _ in st.session_state.training_samples:
        counts[lbl] = counts.get(lbl, 0) + 1
    if counts:
        st_dataframe_compat(pd.DataFrame(list(counts.items()), columns=["signal", "samples"]))
    else:
        st.info("No captured samples yet.")


def render_audit_tab() -> None:
    st.markdown("### Audit log and relay")
    audit_df = pd.DataFrame(st.session_state.audit)
    alerts_df = pd.DataFrame(st.session_state.alerts)
    relay_df = pd.DataFrame(st.session_state.relay_log)

    st.subheader("Alerts")
    if alerts_df.empty:
        st.info("No alerts recorded.")
    else:
        st_dataframe_compat(alerts_df)

    st.subheader("Audit records")
    if audit_df.empty:
        st.info("No audit records yet.")
    else:
        st_dataframe_compat(audit_df.tail(200))

    st.subheader("Emergency relay log")
    if relay_df.empty:
        st.info("No relay events.")
    else:
        st_dataframe_compat(relay_df)

    c1, c2 = st.columns(2)
    if not audit_df.empty:
        c1.download_button(
            "Download audit CSV",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="signalguard_audit.csv",
            mime="text/csv",
        )
    summary = {
        "records": len(audit_df),
        "alerts": len(alerts_df),
        "relay": len(relay_df),
    }
    pdf = build_audit_pdf(audit_df, alerts_df, summary)
    if pdf:
        c2.download_button(
            "Download PDF audit",
            pdf,
            file_name="signalguard_audit.pdf",
            mime="application/pdf",
        )
    else:
        c2.info("PDF export requires reportlab. CSV export remains available.")


def render_camera_tab(context: Dict[str, Any]) -> None:
    st.markdown("### Camera placement guide")
    vis = float(context.get("visibility", 1.0))
    if vis < float(st.session_state.visibility_threshold):
        st.error("Low visibility detected. Follow the guide below and re-check the signaller's arms.")
    else:
        st.success("Visibility currently acceptable.")

    st_image_compat(camera_guide_image())
    st.markdown(
        """
**Checklist**
- Fixed camera on signaller: full body visible, both arms unobstructed.
- Recommended distance: 4-8 m; height: 1.5-2.2 m; slight downward angle.
- Avoid backlight, rain on lens, and occlusion by crane members.
- Optional second camera on load for telemetry visual cross-check.
- If visibility drops, SignalGuard outputs UNCLEAR and recommends STOP.
        """
    )


def render_tests_tab() -> None:
    st.markdown("### Core unit tests")
    st.caption("Tests cover geometry, normalization, mirroring, visibility, tracker state machine, open-set rejection, windows, and alert logic.")
    if st.button("Run unit tests"):
        st.session_state.test_results = run_unit_tests()
    results = st.session_state.get("test_results")
    if results:
        df = pd.DataFrame(results)
        st_dataframe_compat(df)
        passed = sum(1 for r in results if r["status"] == "PASS")
        st.info(f"Passed {passed}/{len(results)}")
    else:
        st.info("Press Run unit tests.")


def main() -> None:
    st.set_page_config(page_title="SignalGuard", page_icon="🦺", layout="wide")
    init_session_state()
    demo = get_demo_dataset()
    render_sidebar()

    st.title(CONFIG["app"]["title"])
    st.warning(CONFIG["app"]["disclaimer"])

    if st.session_state.playing or st.session_state.force_step:
        st.session_state.force_step = False
        context = process_tick(demo)
        st.session_state.context = context
    else:
        context = st.session_state.get("context")
        if context is None:
            context = process_tick(demo)
            st.session_state.context = context

    tabs = st.tabs(["Live demo", "Evaluation", "Training wizard", "Audit", "Camera guide", "Tests"])
    with tabs[0]:
        render_live_tab(context, demo)
    with tabs[1]:
        render_evaluation_tab(demo)
    with tabs[2]:
        render_training_tab()
    with tabs[3]:
        render_audit_tab()
    with tabs[4]:
        render_camera_tab(context)
    with tabs[5]:
        render_tests_tab()

    if st.session_state.playing:
        delay = 1.0 / max(float(CONFIG["camera"]["fps"]) * float(st.session_state.speed), 1.0)
        time.sleep(min(delay, 0.20))
        st.rerun()


if __name__ == "__main__":
    main()