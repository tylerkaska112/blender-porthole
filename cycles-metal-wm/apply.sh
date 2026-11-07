#!/bin/bash
# Applies the Metal-device-over-winemetal changes to a Blender 5.2.3 source tree: apply.sh <blender source dir>
set -eu
B="${1:?blender source dir}"; H="$(cd "$(dirname "$0")" && pwd)"
cp -R "$H/files/." "$B/"
( cd "$B" && git apply --whitespace=nowarn "$H/blender-cmake.patch" && git apply --whitespace=nowarn "$H/packed-state.patch" && git apply --whitespace=nowarn "$H/metal-default.patch" )
echo "applied. Windows: the device is on by default (WITH_CYCLES_DEVICE_METAL_WM). macOS test builds: -DWITH_CYCLES_DEVICE_METAL_WM=ON -DCYCLES_WINEMETAL_LIBRARIES=<native winemetal library>"
