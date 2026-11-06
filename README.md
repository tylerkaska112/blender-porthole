# blender-porthole

Build scripts for a Windows-on-Arm build of [Blender](https://www.blender.org) that renders with Cycles on the GPU when it runs inside [Porthole](https://github.com/tylerkaska112/Porthole)
(Windows programs on iPhone and iPad). Blender is GPL; this repository holds only the build workflow and, later, the changes to Blender's Cycles Metal device. Blender's own source
is fetched from <https://github.com/blender/blender> at build time.

Status: experiment. The first workflow (`dry-run`) builds an unmodified Blender 5.2.3 for Windows Arm64 and records how long each step takes and how much disk it uses.
