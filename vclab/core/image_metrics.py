"""Small, deterministic image metrics with optional OpenCV acceleration.

None of the functions in this module require a neural model or network.  The
OpenCV paths provide better feature matching/optical flow when available, but
the NumPy/Pillow fallbacks keep the application useful on a minimal Python
installation and make tests reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:  # Pillow is the only image dependency we prefer, but keep import optional.
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - exercised on intentionally minimal installs
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def _array(image: Any):
    """Return an RGB float/uint8 array, accepting paths, PIL images and arrays."""

    if Image is not None and isinstance(image, (str, bytes, Path)):
        image = Image.open(image)
    if Image is not None and isinstance(image, Image.Image):
        image = image.convert("RGB")
        if np is None:
            return image
        return np.asarray(image)
    if np is None:
        return image
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3:
        raise ValueError("image must be HxW, HxWx3, or HxWx4")
    # OpenCV callers often supply BGR; this module's public contract is RGB.
    if arr.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def _gray(image: Any):
    arr = _array(image)
    if np is None:
        if Image is not None and isinstance(arr, Image.Image):
            return arr.convert("L")
        return arr
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def _resize_gray(image: Any, size: tuple[int, int] = (32, 32)):
    width, height = size
    if Image is not None:
        arr = _array(image)
        if np is not None:
            pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        else:
            pil = arr
        if isinstance(pil, Image.Image):
            return np.asarray(pil.convert("L").resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32) if np is not None else pil.convert("L")
    if np is None:
        return image
    g = _gray(image)
    ys = np.linspace(0, g.shape[0] - 1, height).astype(int)
    xs = np.linspace(0, g.shape[1] - 1, width).astype(int)
    return g[np.ix_(ys, xs)].astype(np.float32)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def image_size(image: Any) -> tuple[int, int]:
    """Return ``(width, height)``."""

    if Image is not None and isinstance(image, Image.Image):
        return int(image.width), int(image.height)
    if isinstance(image, (str, bytes, Path)) and Image is not None:
        with Image.open(image) as opened:
            return int(opened.width), int(opened.height)
    if np is not None:
        arr = np.asarray(image)
        if arr.ndim < 2:
            return 0, 0
        return int(arr.shape[1]), int(arr.shape[0])
    return 0, 0


def brightness(image: Any) -> float:
    """Mean luminance normalized to 0..1."""

    g = _gray(image)
    if np is None:
        return _safe_float(sum(g.getdata()) / (g.width * g.height) / 255.0)  # type: ignore[attr-defined]
    return _safe_float(np.mean(g) / 255.0)


def contrast(image: Any) -> float:
    """Luminance standard deviation normalized to 0..1."""

    g = _gray(image)
    if np is None:
        vals = list(g.getdata())  # type: ignore[attr-defined]
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        return _safe_float(math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) / 255.0)
    return _safe_float(np.std(g) / 255.0)


def histogram(image: Any, bins: int = 32, *, per_channel: bool = True) -> list[float]:
    """Normalized RGB histogram.

    A compact concatenated histogram is easier to compare than a giant raw
    pixel array.  The output sums to approximately one.
    """

    bins = max(2, int(bins))
    arr = _array(image)
    if np is None:
        if Image is not None and isinstance(arr, Image.Image):
            vals = list(arr.convert("RGB").getdata())
            hist = []
            for c in range(3):
                counts = [0] * bins
                for p in vals:
                    counts[min(bins - 1, int(p[c] * bins / 256))] += 1
                hist.extend(v / max(1, len(vals) * 3) for v in counts)
            return hist
        return []
    if per_channel:
        parts = []
        denom = max(1, arr.shape[0] * arr.shape[1] * 3)
        for c in range(3):
            counts, _ = np.histogram(arr[..., c], bins=bins, range=(0, 256))
            parts.extend((counts / denom).tolist())
        return [float(v) for v in parts]
    values, _ = np.histogram(arr, bins=bins, range=(0, 256))
    return [float(v) for v in values / max(1, arr.size)]


def color_distribution(image: Any, bins: int = 8) -> list[float]:
    """Compact HSV-ish colour distribution.

    OpenCV is used for a hue/saturation/value histogram when available.  The
    fallback quantizes RGB channels, retaining useful colour drift signals.
    """

    bins = max(2, int(bins))
    arr = _array(image)
    if np is None:
        return histogram(arr, bins=bins, per_channel=True)
    if cv2 is not None:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h = np.histogram(hsv[..., 0], bins=bins, range=(0, 180))[0].astype(np.float32)
        s = np.histogram(hsv[..., 1], bins=bins, range=(0, 256))[0].astype(np.float32)
        v = np.histogram(hsv[..., 2], bins=bins, range=(0, 256))[0].astype(np.float32)
        out = np.concatenate((h, s, v))
        return [float(v) for v in out / max(1.0, float(out.sum()))]
    return histogram(arr, bins=bins, per_channel=True)


def perceptual_hash(image: Any, hash_size: int = 8) -> str:
    """Return a DCT-like average hash encoded as hexadecimal.

    The implementation uses a low-frequency resize/threshold and therefore
    remains available without OpenCV's DCT.  It is intentionally stable across
    small resizes and modest brightness changes.
    """

    hash_size = max(2, int(hash_size))
    if np is None:
        # A deterministic fallback based on grayscale samples.
        if Image is None:
            return ""
        g = ImageOps.grayscale(image if isinstance(image, Image.Image) else Image.open(image))
        g = g.resize((hash_size, hash_size))
        vals = list(g.getdata())
        mean = sum(vals) / max(1, len(vals))
        bits = [1 if v >= mean else 0 for v in vals]
    else:
        g = _resize_gray(image, (hash_size, hash_size))
        # A low-frequency DCT block is more robust than raw pixels.
        if cv2 is not None:
            block = cv2.dct(g.astype(np.float32))[:hash_size, :hash_size]
        else:
            block = g
        mean = float(np.mean(block[1:, 1:])) if block.size > 1 else float(np.mean(block))
        bits = (block >= mean).reshape(-1).astype(np.uint8).tolist()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    width = max(1, (len(bits) + 3) // 4)
    return f"{value:0{width}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between hexadecimal perceptual hashes."""

    if not hash_a or not hash_b:
        return 0


def perceptual_hash_distance(hash_a: str, hash_b: str) -> float:
    """Normalized perceptual-hash distance in the 0..1 range."""
    width = max(len(hash_a), len(hash_b)) * 4
    return float(hamming_distance(hash_a, hash_b) / max(1, width))
    try:
        width = max(len(hash_a), len(hash_b)) * 4
        return int((int(hash_a, 16) ^ int(hash_b, 16)).bit_count()) + abs(len(hash_a) - len(hash_b)) * 4
    except Exception:
        return 0


def edge_map(image: Any, *, threshold: int = 60):
    """Return a normalized uint8 edge map (or a nested list without NumPy)."""

    arr = _array(image)
    if np is None:
        return []
    g = _gray(arr).astype(np.uint8)
    if cv2 is not None:
        return cv2.Canny(g, int(threshold), int(threshold) * 2)
    gx = np.diff(g.astype(np.float32), axis=1, prepend=g[:, :1])
    gy = np.diff(g.astype(np.float32), axis=0, prepend=g[:1, :])
    mag = np.sqrt(gx * gx + gy * gy)
    return (mag > float(threshold)).astype(np.uint8) * 255


def edge_density(image: Any) -> float:
    edges = edge_map(image)
    if np is None or not hasattr(edges, "size") or edges.size == 0:
        return 0.0
    return _safe_float(np.mean(edges > 0))


def sharpness(image: Any) -> float:
    """Variance of the Laplacian, scaled to a practical 0..1 range."""

    g = _gray(image)
    if np is None:
        return 0.0
    if cv2 is not None:
        lap = cv2.Laplacian(g, cv2.CV_32F)
    else:
        lap = np.diff(g, n=2, axis=0, prepend=g[:1], append=g[-1:])
    return _safe_float(min(1.0, float(np.var(lap)) / 5000.0))


def _normalized_correlation(a, b) -> float:
    if np is None:
        return 0.0
    aa, bb = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(aa), len(bb))
    if n == 0:
        return 0.0
    aa, bb = aa[:n], bb[:n]
    aa -= np.mean(aa)
    bb -= np.mean(bb)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-8:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.clip(np.dot(aa, bb) / den, -1.0, 1.0))


def compare_feature_matches(image_a: Any, image_b: Any) -> dict[str, Any]:
    """Compare local features, returning score and evidence.

    ``score`` is 0..1 where one means a strong match.  ORB+BFMatcher is used
    when OpenCV is present; a normalized edge/pixel correlation is the fallback.
    """

    a, b = _array(image_a), _array(image_b)
    if cv2 is not None and np is not None:
        ga, gb = _gray(a).astype(np.uint8), _gray(b).astype(np.uint8)
        try:
            orb = cv2.ORB_create(nfeatures=500)
            ka, da = orb.detectAndCompute(ga, None)
            kb, db = orb.detectAndCompute(gb, None)
            if da is not None and db is not None and len(da) and len(db):
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                raw = matcher.knnMatch(da, db, k=2)
                good = [m for m, n in raw if m.distance < 0.75 * n.distance]
                score = min(1.0, len(good) / max(1.0, min(len(ka), len(kb)) * 0.35))
                return {"score": float(score), "matches": len(good), "keypoints_a": len(ka), "keypoints_b": len(kb), "method": "orb"}
        except Exception:
            pass
        # Same-size normalized correlation fallback.
        ra = _resize_gray(a, (64, 64))
        rb = _resize_gray(b, (64, 64))
        score = (_normalized_correlation(ra, rb) + 1.0) / 2.0
        return {"score": float(score), "matches": 0, "method": "correlation"}
    return {"score": 0.0, "matches": 0, "method": "unavailable"}


# Readable alias used in the product brief.
feature_matching = compare_feature_matches


def optical_flow(image_a: Any, image_b: Any) -> dict[str, Any]:
    """Estimate movement from ``image_a`` to ``image_b``.

    Returns mean/max magnitude and a compact ``available`` flag.  A simple
    resized frame difference is used when Farneback is unavailable.
    """

    if np is None:
        return {"mean_magnitude": 0.0, "max_magnitude": 0.0, "available": False, "method": "unavailable"}
    ga = _gray(image_a).astype(np.uint8)
    gb = _gray(image_b).astype(np.uint8)
    # Flow is a diagnostic signal, not a render; cap its working resolution so
    # a 4K shot does not make a whole-sequence report unnecessarily slow.
    max_dim = max(1, max(ga.shape[:2]))
    if max_dim > 256:
        scale = 256.0 / max_dim
        target = (max(1, int(round(ga.shape[1] * scale))), max(1, int(round(ga.shape[0] * scale))))
        if Image is not None:
            ga = np.asarray(Image.fromarray(ga).resize(target, Image.Resampling.BILINEAR), dtype=np.uint8)
            gb = np.asarray(Image.fromarray(gb).resize(target, Image.Resampling.BILINEAR), dtype=np.uint8)
    # Match dimensions before calculating flow.
    if ga.shape != gb.shape:
        target = (max(1, min(256, ga.shape[1])), max(1, min(256, ga.shape[0])))
        if Image is not None:
            ga = np.asarray(Image.fromarray(ga).resize(target), dtype=np.uint8)
            gb = np.asarray(Image.fromarray(gb).resize(target), dtype=np.uint8)
        else:
            gb = _resize_gray(gb, (ga.shape[1], ga.shape[0])).astype(np.uint8)
    if cv2 is not None:
        try:
            flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            return {"mean_magnitude": float(np.mean(mag)), "max_magnitude": float(np.max(mag)), "available": True, "method": "farneback"}
        except Exception:
            pass
    diff = np.abs(gb.astype(np.float32) - ga.astype(np.float32)) / 255.0
    return {"mean_magnitude": float(np.mean(diff)), "max_magnitude": float(np.max(diff)), "available": True, "method": "frame_difference"}


def vector_distance(a: Any, b: Any) -> float:
    """Euclidean distance normalized to a useful 0..1 range."""

    if np is None:
        return 0.0
    aa, bb = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(aa), len(bb))
    if n == 0:
        return 0.0
    d = float(np.linalg.norm(aa[:n] - bb[:n]) / math.sqrt(n))
    return float(np.clip(d, 0.0, 1.0))


def histogram_distance(a: Any, b: Any) -> float:
    """L1 distance between normalized histograms, clipped to 0..1."""

    if np is None:
        return 0.0
    aa, bb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    n = min(aa.size, bb.size)
    if n == 0:
        return 0.0
    # Histograms generally sum to one; half-L1 is total variation distance.
    return float(np.clip(0.5 * np.abs(aa.reshape(-1)[:n] - bb.reshape(-1)[:n]).sum(), 0.0, 1.0))


__all__ = [
    "brightness", "color_distribution", "compare_feature_matches", "feature_matching", "contrast",
    "edge_density", "edge_map", "hamming_distance", "histogram",
    "histogram_distance", "image_size", "optical_flow", "perceptual_hash",
    "perceptual_hash_distance", "sharpness", "vector_distance",
]
