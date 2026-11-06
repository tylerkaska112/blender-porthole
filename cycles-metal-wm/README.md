# Cycles Metal device over winemetal

`files/intern/cycles/device/metal_wm/` is a Cycles GPU device for Apple GPUs written in plain C++ against winemetal.h (DXMT's Metal layer for Windows programs, with the Porthole additions
`patches/0183-0186` of the Porthole engine) instead of Objective-C. `blender-cmake.patch` adds the build option `WITH_CYCLES_DEVICE_METAL_WM`. `apply.sh` puts both into a Blender 5.2.3 tree.

Status (2026-11-06): renders on macOS with winemetal built natively; a scene (`cornell.xml`) matches the CPU and the original Metal backend (`compare.py`: RMS difference 0.17). Software BVH only (no
hardware ray tracing), generic kernels only, no image textures yet.

## Image textures (added)

2D images are `MTLTexture`s created through `MTLDevice_newTexture` (the GPU resource id comes back in
`WMTTextureInfo.gpu_resource_id` and goes into the bindings buffer the kernel indexes); 1D images are
linear buffers whose GPU address goes into the image info. The samplers are constants inside the kernel.
`cornell_tex.xml` (with `tex_checker.png`, an RGBA8 image, and `tex_grad16.png`, 16-bit gray) renders the same
on the CPU, the original Metal backend and this device: RMS 0.24 between every pair.

Bug found on the way: the sort passes (`integrator_sort_bucket_pass`/`write_pass`) need 1024 threads per
threadgroup on every GPU; forcing 64 for all kernels on M3 and later made them write outside the
sort-offsets buffer (it clobbered the image table). The device now carries the original's full tuning table.
