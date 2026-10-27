"""Paint the whole race track into one textured image and show it under the cars.

Grass (noise + mowing stripes), asphalt (grain + darker racing groove), white edge lines,
red/white kerbs on the corners and a checkered start line are rasterized once at startup
with PIL, then shown as a single pyqtgraph GLImageItem: one texture, one draw call.

Usage, in the render setup of the run script (replaces set_racetrack_renderer):
    from sit_lmpc.utils.track_texture import add_textured_track
    add_textured_track(env.unwrapped.renderer, track, config.half_width)
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------- look
GRASS_A = np.array([58, 122, 52])          # mowing stripe colors
GRASS_B = np.array([70, 138, 60])
ASPHALT = np.array([62, 63, 66])
GROOVE = np.array([46, 47, 50])            # darker rubbered-in strip along the middle
LINE_WHITE = (232, 232, 228)
KERB_RED = (198, 38, 34)
KERB_WHITE = (236, 236, 232)
CHECK_BLACK = (20, 20, 22)


# ---------------------------------------------------------------- geometry
def _offset_xy(track, ss, ey):
    """(s, e_y) -> (x, y) via the track's Frenet conversion. ey: scalar or array."""
    import jax.numpy as jnp
    ss = np.asarray(ss, dtype=np.float32)
    ey = np.broadcast_to(np.asarray(ey, dtype=np.float32), ss.shape)
    fr = jnp.stack([jnp.asarray(ss), jnp.asarray(ey), jnp.zeros_like(jnp.asarray(ss))], axis=1)
    return np.asarray(track.vmap_frenet_to_cartesian_jax(fr))[:, :2]


def _curvature(xy):
    """Unsigned curvature of a closed polyline (1/m), smoothed."""
    d = np.gradient(xy, axis=0)
    dd = np.gradient(d, axis=0)
    k = np.abs(d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]) / (np.linalg.norm(d, axis=1) ** 3 + 1e-9)
    w = 15
    kp = np.concatenate([k[-w:], k, k[:w]])
    return np.convolve(kp, np.ones(2 * w + 1) / (2 * w + 1), mode="same")[w:-w]


# ---------------------------------------------------------------- textures
def _noise(h, w, scale_px, rng):
    """Smooth value noise in [0, 1], feature size ~scale_px."""
    sh, sw = max(2, int(h / scale_px) + 2), max(2, int(w / scale_px) + 2)
    small = Image.fromarray((rng.random((sh, sw)) * 255).astype(np.uint8))
    return np.asarray(small.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0


def _fbm(h, w, ppm, rng, sizes_m):
    out = np.zeros((h, w), np.float32)
    amp_sum = 0.0
    for i, size_m in enumerate(sizes_m):
        a = 0.5 ** i
        out += a * _noise(h, w, max(1.0, size_m * ppm), rng)
        amp_sum += a
    return out / amp_sum


def build_track_texture(track, half_width, ppm=20.0, margin_m=12.0, max_px=8192, ds=0.25,
                        kerb_width=0.6, kerb_len=1.0, kerb_curvature=0.04, line_width=0.12,
                        stripe_m=5.0, seed=0):
    """Return (rgba uint8 image as (rows, cols, 4), x_min, y_min, ppm).

    ppm:            pixels per meter (lowered automatically to keep the image <= max_px)
    margin_m:       grass drawn around the track's bounding box
    kerb_curvature: kerbs only where the track turns tighter than this (1/m); 0 = everywhere
    """
    s_max = float(track.s_frame_max)
    ss = np.arange(0.0, s_max, ds)
    center = _offset_xy(track, ss, 0.0)
    left, right = _offset_xy(track, ss, half_width), _offset_xy(track, ss, -half_width)

    pts = np.vstack([left, right])
    x_min, y_min = pts.min(0) - margin_m
    x_max, y_max = pts.max(0) + margin_m
    ppm = min(ppm, (max_px - 1) / max(x_max - x_min, y_max - y_min))
    W, H = int(np.ceil((x_max - x_min) * ppm)), int(np.ceil((y_max - y_min) * ppm))
    rng = np.random.default_rng(seed)

    def to_px(xy):                      # world -> image pixels (row 0 = top = y_max)
        return np.stack([(xy[:, 0] - x_min) * ppm, (y_max - xy[:, 1]) * ppm], axis=1)

    def ribbon(draw, ey_a, ey_b, fill, mask_idx=None):
        """Fill the band between two Frenet offsets, segment by segment (works for any shape)."""
        A = to_px(_offset_xy(track, ss, ey_a)); B = to_px(_offset_xy(track, ss, ey_b))
        n = len(ss)
        idx = range(n) if mask_idx is None else mask_idx
        for i in idx:
            j = (i + 1) % n
            f = fill(i) if callable(fill) else fill
            if f is None:
                continue
            draw.polygon([tuple(A[i]), tuple(A[j]), tuple(B[j]), tuple(B[i])], fill=f)

    # ---- grass: mowing stripes + clumpy noise
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    stripes = (np.floor((xx + yy) / (stripe_m * ppm * 1.414)) % 2)[..., None]
    grass = GRASS_A * (1 - stripes) + GRASS_B * stripes
    g_noise = _fbm(H, W, ppm, rng, [3.0, 0.8, 0.2])[..., None]
    grass = grass * (0.82 + 0.36 * g_noise)
    grass += (rng.random((H, W, 1)) - 0.5) * 10                      # blade-level speckle

    # ---- asphalt: grain + patchy wear + darker groove near the middle
    a_noise = _fbm(H, W, ppm, rng, [4.0, 1.0])[..., None]
    asphalt = ASPHALT * (0.9 + 0.2 * a_noise) + (rng.random((H, W, 1)) - 0.5) * 18
    groove_mask = Image.new("L", (W, H), 0)
    ribbon(ImageDraw.Draw(groove_mask), -0.35 * half_width, 0.35 * half_width, 255)
    gm = np.asarray(groove_mask.filter(ImageFilter.GaussianBlur(0.6 * half_width * ppm)),
                    np.float32)[..., None] / 255.0 * 0.55
    asphalt = asphalt * (1 - gm) + (GROOVE * (0.9 + 0.2 * a_noise)) * gm

    # ---- composite asphalt over grass with a soft 1 px edge
    track_mask = Image.new("L", (W, H), 0)
    ribbon(ImageDraw.Draw(track_mask), -half_width, half_width, 255)
    tm = np.asarray(track_mask.filter(ImageFilter.GaussianBlur(0.7)), np.float32)[..., None] / 255.0
    img = np.clip(grass * (1 - tm) + asphalt * tm, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(img, "RGB")

    # Markings are drawn on a separate layer at SS x resolution, then downsampled:
    # that antialiases thin lines and kerb edges instead of leaving jagged pixels.
    SS = 3
    base_to_px = to_px
    to_px = lambda xy: base_to_px(xy) * SS
    overlay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ---- kerbs on the corners (outside both edges), alternating red/white blocks
    kappa = _curvature(center)
    corner = kappa > kerb_curvature if kerb_curvature > 0 else np.ones_like(kappa, bool)
    # widen each corner a little so kerbs start before the apex
    grow = int(4.0 / ds)
    corner = np.convolve(np.concatenate([corner[-grow:], corner, corner[:grow]]).astype(float),
                         np.ones(2 * grow + 1), "same")[grow:-grow] > 0
    kerb_idx = np.where(corner)[0]
    block = lambda i: KERB_RED if int(ss[i] // kerb_len) % 2 == 0 else KERB_WHITE
    for side in (1, -1):
        ribbon(draw, side * (half_width - 0.02), side * (half_width + kerb_width), block, kerb_idx)

    # ---- white edge lines just inside the asphalt
    for side in (1, -1):
        ribbon(draw, side * (half_width - 0.15), side * (half_width - 0.15 - line_width), LINE_WHITE)

    # ---- checkered start/finish line at s = 0
    cell = 0.5
    rows = 2
    for r in range(rows):
        s0, s1 = r * cell, (r + 1) * cell
        for c, ey0 in enumerate(np.arange(-half_width, half_width - 1e-6, cell)):
            ey1 = min(ey0 + cell, half_width)
            quad = to_px(np.vstack([_offset_xy(track, [s0, s1, s1, s0], [ey0, ey0, ey1, ey1])]))
            draw.polygon([tuple(p) for p in quad], fill=CHECK_BLACK if (r + c) % 2 == 0 else KERB_WHITE)

    overlay = overlay.resize((W, H), Image.LANCZOS)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    rgba = np.asarray(canvas).copy()
    rgba[..., 3] = 255
    return rgba, float(x_min), float(y_min), float(ppm)


# ---------------------------------------------------------------- display
def _cached_texture(track, half_width, cache_dir, **kwargs):
    """build_track_texture, but reuse a PNG from cache_dir when the same track and settings were built before."""
    import hashlib, json, os
    probe = _offset_xy(track, np.linspace(0, float(track.s_frame_max), 64, endpoint=False), 0.0)
    key = hashlib.sha1(json.dumps({"s": round(float(track.s_frame_max), 3), "hw": float(half_width),
                                   "probe": np.round(probe, 2).tolist(), "kw": sorted(kwargs.items())},
                                  default=str).encode()).hexdigest()[:16]
    path = os.path.join(os.path.expanduser(cache_dir), f"track_{key}.png")
    meta = path[:-4] + ".json"
    if os.path.exists(path) and os.path.exists(meta):
        with open(meta) as f:
            m = json.load(f)
        return np.asarray(Image.open(path).convert("RGBA")), m["x_min"], m["y_min"], m["ppm"]
    rgba, x_min, y_min, ppm = build_track_texture(track, half_width, **kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgba).save(path)
    with open(meta, "w") as f:
        json.dump({"x_min": x_min, "y_min": y_min, "ppm": ppm}, f)
    return rgba, x_min, y_min, ppm


def add_textured_track(env_renderer, track, half_width, depth=-30, cache_dir="~/.cache/sit_lmpc_track",
                       **kwargs):
    """Build (or load from cache) the texture and add it to the gym's pyqtgraph view, under everything else.

    kwargs go to build_track_texture (ppm, margin_m, kerb_curvature, stripe_m, seed, ...).
    cache_dir=None always rebuilds.
    """
    import pyqtgraph.opengl as gl
    from OpenGL import GL

    if cache_dir:
        rgba, x_min, y_min, ppm = _cached_texture(track, half_width, cache_dir, **kwargs)
    else:
        rgba, x_min, y_min, ppm = build_track_texture(track, half_width, **kwargs)
    data = np.ascontiguousarray(rgba[::-1].transpose(1, 0, 2))   # (x, y, 4), y up
    item = gl.GLImageItem(data, smooth=True)
    item.setGLOptions({GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True, GL.GL_CULL_FACE: False,
                       'glBlendFunc': (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)})
    item.setDepthValue(depth)                                     # drawn first
    item.scale(1.0 / ppm, 1.0 / ppm, 1, local=False)              # pixels -> meters
    item.translate(x_min, y_min, 0)
    env_renderer.view.addItem(item)
    print(f"[track_texture] {rgba.shape[1]}x{rgba.shape[0]} px texture at {ppm:.1f} px/m")
    return item