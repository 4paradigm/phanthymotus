# Third-party lock

The companion extends the previously validated local image
`phanthy-fast-livo2:g1-1fcd0d0-n3save1`. The expected local image ID is recorded
in `source-lock.env`; deployment must refuse a different image ID.

- FAST-LIVO2 ROS 2/MID360 fork:
  `Rhymer-Lcy/FAST-LIVO2-ROS2-MID360-Fisheye@1fcd0d05cadaeb25ca59fd87cda95aaaee41e3ea`, GPL-2.0-only.
- rpg_vikit ROS 2 fork:
  `Rhymer-Lcy/rpg_vikit_ros2_fisheye@0b5548d9adab58128f3a59a507e96a83acfa8fdf`, GPL-3.0-only.
- The Perception adapter under `g1_fast_livo2/` is Apache-2.0.

The GPL algorithm remains isolated in its own companion image. No FAST-LIVO2
source is copied into the main Perception image.
