#!/bin/bash
# usage: run_scene.sh <scene-name> [extra cycles args]   (renders cpu, original metal, port into scenes/<name>_{cpu,metal,wm}.png)
cd /Volumes/MadeiraWork2/update3/cycles-gpu
export CYCLES_KERNEL_PATH=$PWD/blender/intern/cycles
A=Blender.app/Contents/MacOS/cycles
n=$1; shift
for pair in "CPU:build-cycles:cpu" "METAL:build-cycles:metal" "METAL:build-cycles-wm:wm"; do
  IFS=: read dev dir tag <<< "$pair"
  $dir/intern/cycles/app/$A --device $dev --samples 128 --width 256 --height 256 --background "$@" --output scenes/${n}_$tag.png scenes/$n.xml 2>&1 | tr '\r' '\n' | grep -iE "error|invalid|unknown|out of GPU" | head -3
  echo "$tag done"
done
for p in cpu:wm metal:wm cpu:metal; do a=${p%%:*}; b=${p##*:}; echo "$a vs $b"; python3 compare.py scenes/${n}_$a.png scenes/${n}_$b.png; done
