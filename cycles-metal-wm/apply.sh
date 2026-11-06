#!/bin/bash
# Applies the Metal-device-over-winemetal changes to a Blender 5.2.3 source tree: apply.sh <blender source dir>
set -eu
B="${1:?blender source dir}"; H="$(cd "$(dirname "$0")" && pwd)"
cp -R "$H/files/." "$B/"
( cd "$B" && git apply --whitespace=nowarn "$H/blender-cmake.patch" )
echo "applied: configure with -DWITH_CYCLES_DEVICE_METAL_WM=ON -DCYCLES_WINEMETAL_INCLUDE=<winemetal headers> -DCYCLES_WINEMETAL_LIBRARIES=<libraries>"
