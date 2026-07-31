#!/usr/bin/env bash

set -Eeuo pipefail

# 默认以脚本所在目录作为项目根目录；也可以把项目根目录作为第一个参数传入。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${1:-$SCRIPT_DIR}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "错误：项目根目录不存在：$PROJECT_ROOT" >&2
    exit 1
fi
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"

MACHINE_ARCH="$(uname -m)"
case "${MACHINE_ARCH,,}" in
    x86_64|amd64|i386|i486|i586|i686)
        SOURCE_LIB_NAME="lib_x86"
        ;;
    aarch64|arm64|armhf|armv5*|armv6*|armv7*|armv8*)
        SOURCE_LIB_NAME="lib_arm"
        ;;
    *)
        echo "错误：不支持的系统架构：$MACHINE_ARCH" >&2
        exit 1
        ;;
esac

SDK_DIR="$PROJECT_ROOT/src/mocap_bridge/sdk"
TARGET_BASE_DIRS=(
    "$PROJECT_ROOT"
    "$SDK_DIR"
)

# 开始修改前先检查两处源目录，避免只切换成功一处。
for base_dir in "${TARGET_BASE_DIRS[@]}"; do
    source_dir="$base_dir/$SOURCE_LIB_NAME"
    if [[ ! -d "$source_dir" ]]; then
        echo "错误：找不到源目录：$source_dir" >&2
        exit 1
    fi
done

replace_lib_directory() {
    local base_dir="$1"
    local source_dir="$base_dir/$SOURCE_LIB_NAME"
    local destination_dir="$base_dir/lib"
    local staging_dir
    local backup_dir=""

    # 先完整复制到临时目录，复制失败时不会破坏原来的 lib。
    staging_dir="$(mktemp -d "$base_dir/.lib-stage.XXXXXX")"
    if ! cp -a -- "$source_dir/." "$staging_dir/"; then
        rm -rf -- "$staging_dir"
        echo "错误：复制失败：$source_dir" >&2
        return 1
    fi

    # 临时保留旧 lib；新目录就位成功后再删除旧目录。
    if [[ -e "$destination_dir" || -L "$destination_dir" ]]; then
        backup_dir="$(mktemp -d "$base_dir/.lib-backup.XXXXXX")"
        rmdir -- "$backup_dir"
        mv -- "$destination_dir" "$backup_dir"
    fi

    if mv -- "$staging_dir" "$destination_dir"; then
        if [[ -n "$backup_dir" ]]; then
            rm -rf -- "$backup_dir"
        fi
    else
        rm -rf -- "$staging_dir"
        if [[ -n "$backup_dir" && -e "$backup_dir" ]]; then
            mv -- "$backup_dir" "$destination_dir"
        fi
        echo "错误：无法更新目录：$destination_dir" >&2
        return 1
    fi

    echo "已更新：$destination_dir <- $source_dir"
}

echo "检测到系统架构：$MACHINE_ARCH，选择：$SOURCE_LIB_NAME"
for base_dir in "${TARGET_BASE_DIRS[@]}"; do
    replace_lib_directory "$base_dir"
done
echo "SDK 架构库切换完成。"