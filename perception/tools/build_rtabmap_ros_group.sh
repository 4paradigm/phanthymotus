#!/usr/bin/env bash
# Build JP5.11 ROS package groups in separate, reusable Docker layers.
set -e
. /opt/ros/humble/install/setup.bash
if [ -f /opt/rtabmap_ros/setup.bash ]; then
    . /opt/rtabmap_ros/setup.bash
fi

group="$1"
case "$group" in
    foundations)
        candidates='geometry_msgs tf2 tf2_ros tf2_py tf2_ros_py tf2_eigen tf2_geometry_msgs message_filters'
        ;;
    support)
        candidates='rtabmap_msgs pcl_conversions pcl_ros diagnostic_updater cv_bridge image_geometry image_transport laser_geometry rtabmap_conversions rtabmap_sync rtabmap_util'
        ;;
    slam)
        candidates='rtabmap_slam'
        ;;
    *)
        echo "[rtabmap-build] unknown JP5.11 group: $group" >&2
        exit 2
        ;;
esac

test -s /tmp/rtabmap-ws/packages.list
read -r -a available < /tmp/rtabmap-ws/packages.list
selected=()
for package in "${available[@]}"; do
    case " $candidates " in
        *" $package "*) selected+=("$package") ;;
    esac
done
if [ "${#selected[@]}" -eq 0 ]; then
    echo "[rtabmap-build] no packages selected for $group" >&2
    exit 2
fi

export CCACHE_DIR=/var/cache/rtabmap-ccache
export CCACHE_COMPILERCHECK=content
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
rtabmap_python="$(command -v python3)"
rtabmap_rcutils="$(readlink -f /opt/ros/humble/install/lib/librcutils.so)"
rtabmap_builtin="$(readlink -f /opt/ros/humble/install/lib/libbuiltin_interfaces__rosidl_generator_c.so)"
rtabmap_crypto="$(readlink -f /usr/lib/aarch64-linux-gnu/libcrypto.so)"
test -f "$rtabmap_rcutils"
test -f "$rtabmap_builtin"
test -f "$rtabmap_crypto"

echo "[rtabmap-build] JP5.11 $group packages: ${selected[*]}"
# Review builds are terminated after 600 seconds without output. Large C++
# translation units can remain silent; emit a heartbeat every 60 seconds.
# A failed colcon command still returns its real error code.
heartbeat() {
    while sleep 60; do
        echo "[rtabmap-build] JP5.11 $group still compiling..."
    done
}
heartbeat &
heartbeat_pid=$!
cleanup_heartbeat() {
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT

cd /tmp/rtabmap-ws
MAKEFLAGS=-j2 colcon build --event-handlers console_direct+ \
    --merge-install --install-base /opt/rtabmap_ros \
    --executor sequential --packages-select "${selected[@]}" --cmake-args \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DPython3_EXECUTABLE="$rtabmap_python" -DPYTHON_EXECUTABLE="$rtabmap_python" \
    -Drcutils_LIB:FILEPATH="$rtabmap_rcutils" \
    -Dbuiltin_interfaces__rosidl_generator_c_LIB:FILEPATH="$rtabmap_builtin" \
    -Dcrypto_LIB:FILEPATH="$rtabmap_crypto" \
    '-DCMAKE_INSTALL_RPATH=/opt/rtabmap_ros/lib;/usr/local/lib;/opt/ros/humble/install/lib' \
    -DCMAKE_DISABLE_FIND_PACKAGE_octomap_msgs=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_grid_map_ros=TRUE
