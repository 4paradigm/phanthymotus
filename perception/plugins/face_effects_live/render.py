"""Premultiplied alpha, bounded CPU ROIs or optional EGL/OpenGL shaders.

GL path uploads host frames and reads back the framebuffer. It is NOT zero copy.
"""
import numpy as np
import cv2


class CPU:
    name = "cpu-opencv-roi"

    def render(self, image, commands):
        output = image.copy()
        h, w = image.shape[:2]
        for cmd in commands:
            th, tw = cmd.texture.shape[:2]
            corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]]) @ cmd.matrix[:, :2].T + cmd.matrix[:, 2]
            if not np.isfinite(corners).all():
                continue
            x1, y1 = np.maximum(np.floor(corners.min(0))-1, (0, 0)).astype(int)
            x2, y2 = np.minimum(np.ceil(corners.max(0))+1, (w, h)).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            matrix = cmd.matrix.copy()
            matrix[:, 2] -= (x1, y1)
            sprite = cv2.warpAffine(cmd.texture, matrix, (x2-x1, y2-y1),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            opacity = float(np.clip(cmd.opacity, 0, 1))
            alpha = sprite[:, :, 3:4]*opacity
            roi = output[y1:y2, x1:x2]
            mixed = sprite[:, :, 2::-1]*(opacity*255) + roi.astype(np.float32)*(1-alpha)
            roi[:] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
        return output

    def close(self):
        pass


class GL:
    """Context is created, used and released on the same processing thread."""
    name = "egl-opengl"

    def __init__(self, allow_software=False):
        import moderngl
        self.gl = moderngl
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.device = self.ctx.info.get("GL_RENDERER", "unknown")
        self.software = any(x in self.device.lower() for x in ("llvmpipe", "softpipe", "software"))
        if self.software and not allow_software:
            self.ctx.release()
            raise RuntimeError("EGL is software-rendered; selecting CPU")
        self.program = self.ctx.program(vertex_shader="""
            #version 330
            in vec2 in_pos;
            uniform vec2 viewport;
            uniform vec2 dimensions;
            uniform mat3 transform;
            out vec2 uv;
            void main() {
                vec2 p = (transform * vec3(in_pos * dimensions, 1.0)).xy;
                gl_Position = vec4(2.0*p.x/viewport.x-1.0, 1.0-2.0*p.y/viewport.y, 0.0, 1.0);
                uv = in_pos;
            }
        """, fragment_shader="""
            #version 330
            uniform sampler2D image;
            uniform float opacity;
            in vec2 uv;
            out vec4 color;
            void main() { color = texture(image, uv)*opacity; }
        """)
        self.vbo = self.ctx.buffer(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype="f4").tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, "2f", "in_pos")])
        self.cache = {}
        self.fbo = self.background = None
        self.size = None

    def _draw(self, texture, matrix, dimensions, opacity):
        self.program["viewport"].value = self.size
        self.program["dimensions"].value = dimensions
        matrix3 = np.eye(3, dtype=np.float32)
        matrix3[:2] = matrix
        self.program["transform"].write(matrix3.T.copy().tobytes())
        self.program["opacity"].value = opacity
        texture.use(0)
        self.vao.render(self.gl.TRIANGLE_STRIP)

    def render(self, image, commands):
        h, w = image.shape[:2]
        if self.size != (w, h):
            if self.fbo:
                self.fbo.release()
                self.background.release()
            self.size = (w, h)
            self.fbo = self.ctx.simple_framebuffer((w, h), components=4)
            self.background = self.ctx.texture((w, h), 3, dtype="f1")
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.disable(self.gl.BLEND)
        self.background.write(np.ascontiguousarray(image[:, :, ::-1]).tobytes())
        self._draw(self.background, np.array([[1, 0, 0], [0, 1, 0]]), (w, h), 1.)
        self.ctx.enable(self.gl.BLEND)
        self.ctx.blend_func = self.gl.ONE, self.gl.ONE_MINUS_SRC_ALPHA
        active = {id(cmd.texture) for cmd in commands}
        for key in list(self.cache):
            if key not in active:
                self.cache.pop(key).release()
        for cmd in commands:
            th, tw = cmd.texture.shape[:2]
            key = id(cmd.texture)
            if key not in self.cache:
                tex = self.ctx.texture((tw, th), 4, np.ascontiguousarray(cmd.texture).tobytes(), dtype="f4")
                tex.filter = self.gl.LINEAR, self.gl.LINEAR
                tex.repeat_x = tex.repeat_y = False
                self.cache[key] = tex
            self._draw(self.cache[key], cmd.matrix, (tw, th), cmd.opacity)
        raw = self.fbo.read(components=3, alignment=1)
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3)[::-1, :, ::-1].copy()

    def close(self):
        for tex in self.cache.values():
            tex.release()
        self.cache.clear()
        if self.fbo:
            self.fbo.release()
            self.background.release()
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.ctx.release()


def create_renderer(request="auto", allow_software=False):
    if request not in ("cpu", "gl", "auto"):
        raise ValueError("renderer must be cpu/gl/auto")
    reason = ""
    if request != "cpu":
        try:
            return GL(allow_software), reason
        except Exception as exc:
            if request == "gl":
                raise RuntimeError(f"OpenGL initialization failed: {type(exc).__name__}: {exc}") from exc
            reason = f"OpenGL unavailable: {type(exc).__name__}: {exc}"
    return CPU(), reason
