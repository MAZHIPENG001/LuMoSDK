#!/bin/bash
set -e

echo ">>> 编译中..."
g++ FZMotion_Receive_Sample.cpp \
    -I./include \
    -L./lib \
    -Wl,-rpath-link,./lib \
    -Wl,--disable-new-dtags \
    -Wl,-rpath,'$ORIGIN/lib' \
    -Wl,--no-as-needed \
    -lLuMoSDK \
    -lpthread \
    -o FZMotion_Receive_Sample

echo ">>> 编译成功，运行中..."
./FZMotion_Receive_Sample
