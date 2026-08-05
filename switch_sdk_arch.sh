#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -m)" in
    aarch64|arm64)
        ARCH_LIB_DIR="lib_arm"
        PLATFORM_NAME="ARM64"
        ;;
    x86_64|amd64)
        ARCH_LIB_DIR="lib_x86"
        PLATFORM_NAME="x86_64"
        ;;
    *)
        echo "错误：不支持的平台架构：$(uname -m)" >&2
        echo "当前仅支持 ARM64（aarch64、arm64）和 x86_64（x86_64、amd64）。" >&2
        exit 1
        ;;
esac

SDK_DIRS=(
    "${SCRIPT_DIR}"
    "${SCRIPT_DIR}/src/mocap_bridge/sdk"
)

# 删除任何现有 lib 前，先确认两处架构库都存在，避免只切换了一部分。
for sdk_dir in "${SDK_DIRS[@]}"; do
    source_dir="${sdk_dir}/${ARCH_LIB_DIR}"
    if [[ ! -d "${source_dir}" ]]; then
        echo "错误：架构库目录不存在：${source_dir}" >&2
        exit 1
    fi
done

echo "检测到 ${PLATFORM_NAME} 平台，使用 ${ARCH_LIB_DIR}。"

for sdk_dir in "${SDK_DIRS[@]}"; do
    source_dir="${sdk_dir}/${ARCH_LIB_DIR}"
    target_dir="${sdk_dir}/lib"

    if [[ -e "${target_dir}" || -L "${target_dir}" ]]; then
        echo "删除：${target_dir}"
        rm -rf -- "${target_dir}"
    fi

    echo "复制：${source_dir} -> ${target_dir}"
    cp -a -- "${source_dir}" "${target_dir}"
done

echo "LuMoSDK 库架构切换完成。"
