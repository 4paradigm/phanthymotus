"""Check framing against MuJoCo's actual free-camera frame and near plane."""
import itertools
from pathlib import Path
import sys

import pytest
np = pytest.importorskip("numpy")
mujoco = pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_service.camera_fit import RobotCameraFit

SCENE = '''<mujoco><compiler angle="radian"/><worldbody>
<geom type="plane" size="50 50 .1"/>
<body pos="0 0 1"><freejoint/><geom type="box" size=".12 .1 .4" mass="2"/>
<body pos="0 0 .35"><joint axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .7 .1 .1" size=".035" mass=".1"/>
<body pos=".7 .1 .1"><geom type="box" size=".06 .04 .025" mass=".1"/></body></body>
<body pos="0 0 -.4"><geom type="box" size=".18 .07 .035" mass=".1"/></body>
</body></worldbody></mujoco>'''


def screen_coordinates(model, data, view, points, width, height):
    """Projection uses MuJoCo's generated camera, not the fitter's trigonometry."""
    camera = mujoco.MjvCamera()
    for name in ("distance", "azimuth", "elevation"):
        setattr(camera, name, view[name])
    camera.lookat[:] = view["lookat"]
    scene = mujoco.MjvScene(model, maxgeom=100)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera,
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    left, right = scene.camera
    position = (np.array(left.pos) + np.array(right.pos)) / 2
    forward, up = np.array(left.forward), np.array(left.up)
    horizontal = np.cross(forward, up)
    relative = points - position
    depth = relative @ forward
    tan_vertical = left.frustum_top / left.frustum_near
    screen = np.column_stack((relative @ horizontal / (depth * tan_vertical * width / height),
                              relative @ up / (depth * tan_vertical)))
    return screen, depth, left.frustum_near


@pytest.mark.parametrize("viewport", [(192, 216), (240, 160), (640, 480), (160, 640)])
@pytest.mark.parametrize("angles", [(130, -25), (20, -30), (275, 15)])
def test_rotated_limbs_and_near_geometry_stay_inside_the_actual_camera(viewport, angles):
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    # Tilt the complete robot and articulate a long arm toward the camera.
    data.qpos[3:7] = [.9238795325, 0, .3826834324, 0]
    data.qpos[7] = -.7
    mujoco.mj_forward(model, data)
    geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] != 0)
    fitter = RobotCameraFit(model, geoms)
    before_qpos, before_time = data.qpos.copy(), data.time
    view = fitter.view(data, *viewport, {"azimuth": angles[0], "elevation": angles[1]})
    screen, depth, near = screen_coordinates(model, data, view, fitter.points(data), *viewport)
    assert np.max(abs(screen)) <= .84001
    assert np.max(abs(screen)) >= .839  # No redundant whole-robot padding.
    assert np.min(depth) > near
    np.testing.assert_array_equal(data.qpos, before_qpos)
    assert data.time == before_time


def test_long_thin_robot_uses_height_without_spherical_inflation():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="0 0 1">
      <freejoint/><geom type="box" size=".07 .05 .8" mass="1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fitter = RobotCameraFit(model, [0])
    view = fitter.view(data, 240, 180)
    screen, _, _ = screen_coordinates(model, data, view, fitter.points(data), 240, 180)
    height_fraction = np.ptp(screen[:, 1]) / 2
    assert .75 <= height_fraction <= .85
    assert view["distance"] < 2.5


def test_hidden_collision_proxies_do_not_shrink_the_visible_robot():
    source = '''<mujoco><asset><mesh name="visual" vertex="-.1 -.1 -.4 .1 -.1 -.4 0 .1 -.4 0 0 .4"/></asset>
    <worldbody><body pos="0 0 1"><freejoint/>
    <geom type="mesh" mesh="visual" group="2" mass="1"/>
    <geom type="box" size="10 10 10" group="3" mass="1"/>
    <geom type="sphere" size="20" rgba="1 1 1 0" mass="1"/>
    </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(source)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fitter = RobotCameraFit(model, [0, 1, 2])
    assert fitter.geoms.tolist() == [0]
    view = fitter.view(data, 240, 180)
    assert view["distance"] < 2
