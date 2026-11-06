# Cycles Metal device over winemetal

`files/intern/cycles/device/metal_wm/` is a Cycles GPU device for Apple GPUs written in plain C++ against winemetal.h (DXMT's Metal layer for Windows programs, with the Porthole additions
`patches/0183-0186` of the Porthole engine) instead of Objective-C. `blender-cmake.patch` adds the build option `WITH_CYCLES_DEVICE_METAL_WM`. `apply.sh` puts both into a Blender 5.2.3 tree.

Status (2026-11-06): renders on macOS with winemetal built natively; a scene (`cornell.xml`) matches the CPU and the original Metal backend (`compare.py`: RMS difference 0.17). Software BVH only (no
hardware ray tracing), generic kernels only, no image textures yet.
