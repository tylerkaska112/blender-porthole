#!/bin/bash
cd /Volumes/MadeiraWork2/update3/cycles-gpu
export CYCLES_KERNEL_PATH=$PWD/blender/intern/cycles
A=Blender.app/Contents/MacOS/cycles
ARGS="--samples 128 --width 256 --height 256 --background"
build-cycles/intern/cycles/app/$A --device CPU $ARGS --output scenes/tex_cpu.png scenes/cornell_tex.xml > tex_cpu.log 2>&1
echo cpu done
build-cycles/intern/cycles/app/$A --device METAL $ARGS --output scenes/tex_metal.png scenes/cornell_tex.xml > tex_metal.log 2>&1
echo metal done
build-cycles-wm/intern/cycles/app/$A --device METAL $ARGS --output scenes/tex_wm.png scenes/cornell_tex.xml > tex_wm.log 2>&1
echo wm done
