"""Draw a car image instead of f1tenth_gym's plain rectangle, without editing the installed package.

Usage (once, before gym.make):
    from sit_lmpc.utils.car_sprite import install_car_sprite
    install_car_sprite()

It finds f1tenth_gym's pyqtgraph CarRenderer class and wraps its __init__, update and render
methods. Patching the class object itself works no matter how the gym imported it, and the
installed package in .venv stays untouched, so `uv sync` can never undo it.
"""
from __future__ import annotations

import importlib
import os
import pkgutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPRITE = os.path.join(HERE, "f1tenth_car_photo_600.png")


def _find_car_renderer():
    """Return f1tenth_gym's pyqtgraph CarRenderer class (the one built from a base_rect mesh)."""
    import f1tenth_gym.envs.rendering as pkg
    found = []
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue                      # e.g. a backend whose library isn't installed
        cls = getattr(m, "CarRenderer", None)
        if cls is not None and cls.__module__ == m.__name__:   # defined here, not just imported
            found.append(cls)
    for cls in found:                     # the pyqtgraph one builds its mesh from base_rect
        if hasattr(cls, "apply_pose"):
            return cls
    if found:
        return found[0]
    raise ImportError("CarRenderer not found in f1tenth_gym.envs.rendering")


def _load_sprite(path, max_px):
    """RGBA image -> (x, y, 4) uint8 array, the layout pyqtgraph's GLImageItem expects."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    if max(im.size) > max_px:
        k = max_px / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)            # rows run downward, row 0 = top of image
    return np.ascontiguousarray(arr[::-1].transpose(1, 0, 2))   # world y runs upward


def reset_gl_caches():
    """Forget OpenGL shader programs that pyqtgraph cached in earlier runs.

    pyqtgraph compiles each shader once and keeps its id in a class-level cache. When the run
    script retries with a new env, a new window (new OpenGL context) is created, and those cached
    ids are invalid there: glUseProgram fails with GLError 1281 and the points are not drawn.
    Call this at the start of every run, before gym.make. It is a no-op on the first run.
    """
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import shaders
    for prog in getattr(shaders.ShaderProgram, "names", {}).values():
        prog.prog = None
    if isinstance(getattr(shaders, "CompiledShaderPrograms", None), dict):
        shaders.CompiledShaderPrograms.clear()
    for name in dir(gl):
        cls = getattr(gl, name)
        if isinstance(cls, type) and "_shaderProgram" in vars(cls):
            cls._shaderProgram = None


def install_car_sprite(path: str = DEFAULT_SPRITE, max_px: int = 256, keep_aspect: bool = True):
    """Patch CarRenderer so every car is drawn with the image at `path` (front pointing +x).

    max_px:      texture is downscaled once to this size (big textures shimmer when shown small)
    keep_aspect: draw it car_width wide at the image's own length/width ratio, instead of
                 stretching it into the car's box (your run script makes that box square)
    """
    import pyqtgraph.opengl as gl

    cls = _find_car_renderer()
    if getattr(cls, "_sprite_patched", False):
        return cls
    if not os.path.exists(path):
        print(f"[car_sprite] image not found at {path}; keeping the plain rectangle")
        return cls
    data = _load_sprite(path, max_px)

    orig_init, orig_update, orig_render = cls.__init__, cls.update, cls.render

    def __init__(self, env_renderer, *args, **kwargs):
        orig_init(self, env_renderer, *args, **kwargs)
        self.sprite_px = data.shape[:2]
        self.sprite_len = self.car_length
        if keep_aspect:
            self.sprite_len = self.car_width * self.sprite_px[0] / self.sprite_px[1]
        self.sprite = gl.GLImageItem(data, smooth=True)   # 'translucent' keeps the alpha
        env_renderer.view.addItem(self.sprite)
        self.mesh.setVisible(False)                       # rectangle only reappears on collision
        self._sprite_collided = False

    def update(self, obs, id):
        orig_update(self, obs, id)
        if obs[id]["collision"] > 0 and not self._sprite_collided:
            self._sprite_collided = True
            rgba = np.array([[1.0, 0.0, 0.0, 0.5]] * 2)   # half-transparent red block over the image
            self.mesh.setMeshData(vertexes=self.base_rect * self.scale, faces=self.faces,
                                  faceColors=rgba, smooth=False, drawEdges=False)
            self.mesh.setGLOptions("translucent")
            self.mesh.setVisible(True)

    def render(self, scale=1.0):
        orig_render(self, scale)
        # each call is applied after the previous one (local=False):
        # center on origin -> size in meters -> rotate by yaw -> move to the car
        w_px, h_px = self.sprite_px
        x, y, yaw = self.pose
        self.sprite.resetTransform()
        self.sprite.translate(-w_px / 2, -h_px / 2, 0)
        self.sprite.scale(self.sprite_len * scale / w_px, self.car_width * scale / h_px, 1, local=False)
        self.sprite.rotate(np.degrees(yaw), 0, 0, 1)
        self.sprite.translate(x, y, self.z_offset)

    cls.__init__, cls.update, cls.render = __init__, update, render
    cls._sprite_patched = True
    print(f"[car_sprite] {cls.__module__}.CarRenderer now draws {os.path.basename(path)}")
    return cls