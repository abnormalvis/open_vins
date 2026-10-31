#!/usr/bin/env bash
set -euo pipefail

OV_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
BRIDGE_WS="${BRIDGE_WS:-/home/orangepi/vision2pose_ws}"
FLIGHT_WS="${FLIGHT_WS:-/home/orangepi/Fast-Drone-250}"
ROS_DISTRO="${ROS_DISTRO:-noetic}"

required_setups=(
    "/opt/ros/${ROS_DISTRO}/setup.bash"
    "${OV_WS}/devel/setup.bash"
    "${BRIDGE_WS}/devel/setup.bash"
    "${FLIGHT_WS}/devel/setup.bash"
)

for setup_file in "${required_setups[@]}"; do
    if [[ ! -f "${setup_file}" ]]; then
        printf 'Missing ROS setup file: %s\n' "${setup_file}" >&2
        exit 1
    fi
    source "${setup_file}"
done

export ROS_PACKAGE_PATH="${OV_WS}/src/open_vins:${BRIDGE_WS}/src:${FLIGHT_WS}/src/realflight_modules/realsense-ros:${ROS_PACKAGE_PATH:-}"
export CMAKE_PREFIX_PATH="${BRIDGE_WS}/devel:${OV_WS}/devel:${FLIGHT_WS}/devel:/opt/ros/${ROS_DISTRO}:${CMAKE_PREFIX_PATH:-}"

if ! rospack find rosmon_core >/dev/null 2>&1; then
    printf 'ROS package not found: rosmon_core\n' >&2
    exit 1
fi

for package in mavros realsense2_camera ov_msckf vins_mavros_bridge; do
    if ! rospack find "${package}" >/dev/null 2>&1; then
        printf 'ROS package not found: %s\n' "${package}" >&2
        exit 1
    fi
done

if ! python3 -c "from roslib.packages import find_node; import sys; sys.exit(0 if find_node('vins_mavros_bridge', 'vins_mavros_bridge_node') else 1)"; then
    printf 'Bridge executable not found. Build ${BRIDGE_WS} and source its devel/setup.bash.\n' >&2
    exit 1
fi

exec rosrun rosmon_core rosmon ov_msckf mavros_realsense_openvins.launch \
    "fcu_type:=${FCU_TYPE:-px4}" \
    "fcu_url:=${FCU_URL:-/dev/ttyS6:921600}" \
    "$@"