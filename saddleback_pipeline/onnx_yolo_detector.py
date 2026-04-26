"""Optional **ONNX YOLO** detector on raster PDF pages → same JSON shape as ``gemini_spatial_detection``.

Designed for **Ultralytics YOLOv8** ONNX exports (single-class steel models or multi-class).

Install::

    pip install onnxruntime

Environment (optional overrides):

* ``YOLO_CONF_THRESHOLD`` — default ``0.25``
* ``YOLO_IOU_THRESHOLD`` — default ``0.45``
* ``YOLO_INPUT_SIZE`` — default ``640``
* ``YOLO_CLASS_NAMES`` — comma-separated labels (used for ``structural_instances[].class``),
  e.g. ``beam,column``. If omitted, classes are ``class_0``, ``class_1``, …

**Training**: You must supply a labeled dataset of drawing crops and export ONNX.
This module does **not** ship weights — it only runs inference when you provide ``model.onnx``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy NMS; boxes (N,4) xyxy, scores (N,)."""
    idxs = scores.argsort()[::-1]
    keep: list[int] = []
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_j - inter + 1e-9
        iou = inter / union
        idxs = rest[iou < iou_thres]
    return keep


def _letterbox(
    img: np.ndarray,
    new_size: int,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize with padding; returns letterboxed BGR image, ratio (rw,rh), pad (dw,dh)."""
    import cv2

    h0, w0 = img.shape[:2]
    r = min(new_size / h0, new_size / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = (new_size - nw) / 2
    dh = (new_size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return out, (r, r), (float(left), float(top))


def _decode_yolov8_like(pred: np.ndarray, conf_thres: float, iou_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pred shape (N, C) where C = 4 + num_classes."""
    xywh = pred[:, :4]
    cls_logits = pred[:, 4:]
    if cls_logits.shape[1] == 1:
        scores = _sigmoid(cls_logits[:, 0])
        cls_ids = np.zeros(len(pred), dtype=np.int32)
    else:
        cls_prob = _sigmoid(cls_logits)
        scores = cls_prob.max(axis=1)
        cls_ids = cls_prob.argmax(axis=1).astype(np.int32)
    mask = scores >= conf_thres
    xywh = xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    if len(scores) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )
    boxes = _xywh_to_xyxy(xywh)
    keep = _nms_xyxy(boxes, scores, iou_thres)
    return boxes[keep], scores[keep], cls_ids[keep]


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    if len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not PNG")
    w = int.from_bytes(png_bytes[16:20], "big")
    h = int.from_bytes(png_bytes[20:24], "big")
    return w, h


def run_yolo_onnx_for_png_pages(
    png_pages: list[bytes],
    onnx_path: Path,
    *,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
    input_size: int | None = None,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        import cv2
    except ImportError as e:
        raise ImportError(
            "onnx_yolo_detector requires onnxruntime and opencv-python. "
            "pip install onnxruntime opencv-python-headless",
        ) from e

    onnx_path = onnx_path.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    conf_thres = float(os.getenv("YOLO_CONF_THRESHOLD", str(conf_thres if conf_thres is not None else "0.25")))
    iou_thres = float(os.getenv("YOLO_IOU_THRESHOLD", str(iou_thres if iou_thres is not None else "0.45")))
    inp_sz = int(os.getenv("YOLO_INPUT_SIZE", str(input_size if input_size is not None else "640")))
    names_env = (os.getenv("YOLO_CLASS_NAMES", "") or "").strip()
    class_names = [x.strip() for x in names_env.split(",") if x.strip()] if names_env else []

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_name = inp.name

    pages_out: list[dict[str, Any]] = []
    total = len(png_pages)

    for i, png in enumerate(png_pages):
        page_1based = i + 1
        w_px, h_px = _png_dimensions(png)
        arr = np.frombuffer(png, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_lb, ratio, pad = _letterbox(img, inp_sz)
        blob = cv2.dnn.blobFromImage(img_lb, scalefactor=1 / 255.0, size=(inp_sz, inp_sz), swapRB=True)
        outs = sess.run(None, {in_name: blob})
        raw = outs[0]
        # Expect (1, C, N) YOLOv8 style
        if raw.ndim != 3:
            raise ValueError(f"Unexpected ONNX output ndim={raw.ndim}; shape={raw.shape}")
        batch = raw[0]
        if batch.shape[0] < batch.shape[1]:
            # (features, anchors) e.g. (84, 8400)
            pred = np.transpose(batch)
        else:
            pred = batch
        boxes, scores, cls_ids = _decode_yolov8_like(pred.astype(np.float32), conf_thres, iou_thres)

        # Map letterboxed coords back to original image pixels
        r = ratio[0]
        px_pad, py_pad = pad
        xyxy_orig = boxes.copy()
        xyxy_orig[:, [0, 2]] -= px_pad
        xyxy_orig[:, [1, 3]] -= py_pad
        xyxy_orig /= r

        instances: list[dict[str, Any]] = []
        for bi in range(len(scores)):
            x0, y0, x1, y1 = xyxy_orig[bi]
            x0 = float(np.clip(x0, 0, w_px))
            x1 = float(np.clip(x1, 0, w_px))
            y0 = float(np.clip(y0, 0, h_px))
            y1 = float(np.clip(y1, 0, h_px))
            cid = int(cls_ids[bi])
            label = (
                class_names[cid]
                if cid < len(class_names)
                else f"class_{cid}"
            )
            instances.append(
                {
                    "class": label.replace(" ", "_").lower(),
                    "view_id": "",
                    "bbox": {
                        "x_min": x0 / float(w_px),
                        "y_min": y0 / float(h_px),
                        "x_max": x1 / float(w_px),
                        "y_max": y1 / float(h_px),
                    },
                    "confidence": float(scores[bi]),
                },
            )

        pages_out.append(
            {
                "page_index": page_1based,
                "width_px": w_px,
                "height_px": h_px,
                "view_regions": [],
                "structural_instances": instances,
                "notes": [f"onnx_yolo:{onnx_path.name}"],
            },
        )

    print(
        f"ONNX YOLO: {total} page(s), total instances={sum(len(p['structural_instances']) for p in pages_out)}",
        file=sys.stderr,
    )
    return {"version": 1, "detector": "onnx_yolo", "pages": pages_out}
