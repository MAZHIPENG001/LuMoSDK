#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  ./switch_sdk_arch.sh [auto|arm64|x86_64]

参数:
  auto     根据 uname -m 自动选择（默认）
  arm64    使用 ARM64/aarch64 版本，即 lib_arm
  x86_64   使用 x86_64/amd64 版本，即 lib_x86
EOF
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

requested_arch="${1:-auto}"
if [[ "$requested_arch" == "auto" ]]; then
    requested_arch="$(uname -m)"
fi

case "$requested_arch" in
    aarch64|arm64)
        library_dir="lib_arm"
        normalized_arch="arm64"
        ;;
    x86_64|amd64)
        library_dir="lib_x86"
        normalized_arch="x86_64"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "错误: 不支持的平台架构: $requested_arch" >&2
        echo "当前脚本仅支持 ARM64/aarch64 和 x86_64/amd64。" >&2
        exit 1
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sdk_dir="$script_dir/src/mocap_bridge/sdk"
library_roots=("$script_dir" "$sdk_dir")

# 先检查两个架构目录，避免只切换了其中一个位置。
for library_root in "${library_roots[@]}"; do
    source_dir="$library_root/$library_dir"
    if [[ ! -d "$source_dir" ]]; then
        echo "错误: 架构库目录不存在: $source_dir" >&2
        exit 1
    fi
    if [[ ! -f "$source_dir/libLuMoSDK.so" ]]; then
        echo "错误: 架构库不存在: $source_dir/libLuMoSDK.so" >&2
        exit 1
    fi
done

for library_root in "${library_roots[@]}"; do
    source_dir="$library_root/$library_dir"
    lib_path="$library_root/lib"

    if [[ -e "$lib_path" || -L "$lib_path" ]]; then
        echo "删除旧路径: $lib_path"
        rm -rf -- "$lib_path"
    fi

    if [[ -e "$lib_path" || -L "$lib_path" ]]; then
        echo "错误: 无法删除旧路径: $lib_path" >&2
        exit 1
    fi

    cp -a -- "$source_dir" "$lib_path"
    echo "已复制: $source_dir -> $lib_path"
done

echo "LuMoSDK 已切换到 $normalized_arch 架构。"
