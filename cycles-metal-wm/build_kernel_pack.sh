#!/bin/bash
# Builds Cycles' Metal kernels ahead of time, on a Mac, for the GPUs of iPhones and iPads (Apple's offline tools: metal, metallib, metal-tt).
# On a phone the biggest kernels kill the system's Metal compiler; with this pack the device loads GPU binaries instead and compiles nothing.
#
#   build_kernel_pack.sh <cycles binary> <features> <out dir> <ios|macos> <os version> <gpu architecture> [more architectures]
#     <cycles binary>  a macOS build of Cycles with the Porthole Metal device (build-cycles-wm/.../cycles); it only writes out the source it would compile
#     <features>       the kernel feature mask (decimal or 0x hex) the pack is for; the scene's own features are added to it, so use the scene's full mask
#     <ios|macos>      the platform the AIR and binaries are for (macos: to test the pack on this Mac, with applegpu_g16g for an M4)
#     <os version>     the system version the device runs, e.g. 27.2: GPU binaries only fit the system version they were built for (an older build misses and the device compiles instead)
#     gpu architecture e.g. applegpu_g16p (A17 Pro), applegpu_g16g (M4), applegpu_g14p (A15): `applegpu-nt -available-archs ios 18.0` lists them
# Writes <out>/<features, 8 hex digits>.metallib and <out>/<features>-<architecture>-<os version>.archive (and masks.txt) : Porthole's Software page installs them for the device that fits.
set -eu
CYC="$1"; WANT="$2"; OUT="$3"; PLAT="$4"; OSV="$5"; shift 5
# "generic" (or ffffffff): the kernels for any scene, without a feature mask (what Cycles calls generic kernels; slower than a specialised pack, but never misses).
if [ "$WANT" = generic ] || [ "$WANT" = ffffffff ]; then FEATURES=ffffffff; GENERIC=1; else FEATURES=$(printf '%08x' "$WANT"); GENERIC=0; fi
mkdir -p "$OUT"; W=$(mktemp -d)
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLS_SRC="${CYCLES_KERNEL_PATH:?set CYCLES_KERNEL_PATH to blender/intern/cycles}"
# 1. The exact source the device would compile for these features (a tiny scene just to get the device to ask).
export CYCLES_KERNEL_PATH="$TOOLS_SRC" CYCLES_WM_DUMP_SOURCE="$W" CYCLES_WM_NO_PREBUILT=1
if [ $GENERIC = 1 ]; then export CYCLES_METAL_ADAPTIVE_COMPILE=0; else export CYCLES_METAL_ADAPTIVE_COMPILE=1 CYCLES_WM_FEATURES_OR="0x$FEATURES"; fi
"$CYC" --device METAL --samples 1 --width 32 --height 32 --background --output "$W/x.png" "${SCENE:-$HERE/../../../scenes/cornell.xml}" >/dev/null 2>&1 || true
if [ $GENERIC = 1 ]; then SRCFILE=$(ls "$W"/*.metal | head -1); else SRCFILE="$W/$FEATURES.metal"; fi
[ -f "$SRCFILE" ] || { echo "the device did not write the source for 0x$FEATURES (the scene's own features must be inside it)"; ls "$W"; exit 1; }
# 2. Source -> AIR -> metallib (device independent).
if [ "$PLAT" = ios ]; then SDK=iphoneos; MIN="-mios-version-min=$OSV"; TT="-ios-version-min $OSV"; else SDK=macosx; MIN="-mmacosx-version-min=$OSV"; TT="-macos-version-min $OSV"; fi
xcrun -sdk $SDK metal -c -std=metal3.2 -ffast-math $MIN "$SRCFILE" -o "$W/k.air"
xcrun -sdk $SDK metallib "$W/k.air" -o "$OUT/$FEATURES.metallib"
# 3. GPU binaries per architecture. The pipeline descriptors match what the device asks for (threadgroup size, buffers 0-2 immutable), or the lookup misses.
python3 - "$W/core.mtlp-json" <<'PY'
import json, sys
names = "reset init_from_camera intersect_closest intersect_shadow shade_background shade_light_nee shade_surface shade_shadow queued_paths_array queued_shadow_paths_array active_paths_array terminated_paths_array terminated_shadow_paths_array sorted_paths_array sort_bucket_pass sort_write_pass compact_paths_array compact_states compact_shadow_paths_array compact_shadow_states".split()
fn = ["cycles_metal_integrator_" + n for n in names] + ["cycles_metal_shader_eval_background", "cycles_metal_prefix_sum"]
def entry(f):
    t = 1024 if ("sort_bucket" in f or "sort_write" in f) else 64
    return {"compute_function": f, "threadgroup_size_is_multiple_of_thread_execution_width": True, "max_total_threads_per_threadgroup": t,
            "buffers": [{"mutability": "Immutable"}] * 3}
json.dump({"pipelines": {"compute_pipelines": [entry(f) for f in fn]}}, open(sys.argv[1], "w"), indent=1)
PY
for ARCH in "$@"; do
  xcrun metal-tt -arch "$ARCH" $TT "$OUT/$FEATURES.metallib" "$W/core.mtlp-json" -o "$OUT/$FEATURES-$ARCH-$OSV.archive"
  echo "built $OUT/$FEATURES-$ARCH-$OSV.archive ($(stat -f %z "$OUT/$FEATURES-$ARCH-$OSV.archive") bytes)"
done
grep -qx "$FEATURES" "$OUT/masks.txt" 2>/dev/null || echo "$FEATURES" >> "$OUT/masks.txt"
echo "built $OUT/$FEATURES.metallib ($(stat -f %z "$OUT/$FEATURES.metallib") bytes)"
rm -rf "$W"
