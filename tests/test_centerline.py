import numpy as np
import pytest
import tempfile
import os
from PIL import Image

from pathracer.centerline import ordered_centerline


def _make_stroke(w=200, h=200, thick=6):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = np.array(img)
    # just a diagonal yellow line
    for i in range(20, 180):
        for dy in range(-thick // 2, thick // 2 + 1):
            y = i + dy
            if 0 <= y < h:
                px[y, i] = [255, 200, 0, 255]
    return Image.fromarray(px)


def test_ordered_centerline_returns_array():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _make_stroke().save(f.name)
        try:
            pts = ordered_centerline(f.name)
            assert isinstance(pts, np.ndarray)
            assert pts.ndim == 2
            assert pts.shape[1] == 2
            assert len(pts) > 10
        finally:
            os.unlink(f.name)


def test_ordered_centerline_direction():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _make_stroke().save(f.name)
        try:
            pts = ordered_centerline(f.name)
            x_span = pts[:, 0].max() - pts[:, 0].min()
            assert x_span > 100  # should cover most of the 200px image
        finally:
            os.unlink(f.name)


def test_empty_image_raises():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(f.name)
        try:
            with pytest.raises(ValueError):
                ordered_centerline(f.name)
        finally:
            os.unlink(f.name)
