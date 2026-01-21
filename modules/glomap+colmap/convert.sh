#!/bin/bash

# 检查是否提供了根目录参数
if [ -z "$1" ]; then
    echo "Usage: $0 <root_directory>"
    echo "Example: $0 /path/to/project"
    exit 1
fi

ROOT_DIR="$1"
INPUT_ROOT="$ROOT_DIR/input"
OUTPUT_ROOT="$ROOT_DIR/images"

# 获取脚本所在目录，用于定位同级目录下的二进制文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GLOMAP_BIN="$SCRIPT_DIR/glomap"
COLMAP_BIN="$SCRIPT_DIR/colmap"

# 检查输入目录是否存在
if [ ! -d "$INPUT_ROOT" ]; then
    echo "Error: Input directory '$INPUT_ROOT' does not exist."
    exit 1
fi

# 遍历 input 下的所有 frame 文件夹 (确保按数字顺序排序)
find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "frame*" | sort | while read FRAME_DIR; do
    FRAME_NAME=$(basename "$FRAME_DIR")
    # 从 'frame000123' 或 'frame_000123' 中提取数字
    # sed 's/^[^0-9]*//' 去掉开头非数字字符 (如 frame, frame_)
    # sed 's/^0*//' 去掉前导零
    FRAME_IDX=$(echo "$FRAME_NAME" | sed 's/^[^0-9]*//' | sed 's/^0*//')
    # 处理纯0的情况 (sed 可能会变为空)
    if [ -z "$FRAME_IDX" ]; then FRAME_IDX=0; fi

    echo "========================================"
    echo "Processing Frame: $FRAME_NAME (Index: $FRAME_IDX)"
    echo "========================================"
    echo "Debug: IMAGE_PATH='$FRAME_DIR'"
    if [ ! -d "$FRAME_DIR" ]; then
        echo "Error: Directory not found!"
        continue
    fi
    # 列出目录内容数量确认正常
    NUM_FILES=$(ls -1 "$FRAME_DIR" | wc -l)
    echo "Debug: Found $NUM_FILES files in directory."

    # 定义路径
    IMAGE_PATH="$FRAME_DIR"
    DATABASE_PATH="$FRAME_DIR/database.db"
    

    
    # 最终输出路径 (Undistorted) - 用户的 output 目录
    FINAL_OUTPUT_DIR="$OUTPUT_ROOT/$FRAME_NAME"

    # 计算 Index
    MOD_VAL=$((FRAME_IDX % 10))
    
    # 确定本帧的 Distorted Model 存储位置 (仅关键帧有)
    KEYFRAME_DISTORTED_DIR="$ROOT_DIR/tem_params/$FRAME_NAME/sparse"
    
    # 变量：用于 Undistorter 的输入模型路径
    INPUT_MODEL_PATH=""

    if [ "$MOD_VAL" -eq 0 ]; then
        # 中间结果路径 (Distorted Sparse) - 仅关键帧需要
        CACHE_DISTORTED_DIR="$ROOT_DIR/tem_params/$FRAME_NAME"
        mkdir -p "$CACHE_DISTORTED_DIR"
        
        # database.db 路径
        DATABASE_PATH="$CACHE_DISTORTED_DIR/database.db"
        # === 关键帧逻辑 ===
        echo ">> [Keyframe $FRAME_IDX] Handling..."
        
        # 1. 检查缓存
        if [ -f "$KEYFRAME_DISTORTED_DIR/0/cameras.bin" ] && [ -f "$KEYFRAME_DISTORTED_DIR/0/images.bin" ]; then
            echo "   [Cache] Found existing distorted model at $KEYFRAME_DISTORTED_DIR"
            INPUT_MODEL_PATH="$KEYFRAME_DISTORTED_DIR/0"
        else
            echo "   [Generate] No cache found. Running reconstruction..."
            
            # 建立目录
            mkdir -p "$KEYFRAME_DISTORTED_DIR"
            
            # Step 1: Feature Extraction
            echo "   -> Feature Extractor..."
            "$COLMAP_BIN" feature_extractor \
                --image_path "$IMAGE_PATH" \
                --database_path "$DATABASE_PATH" \
                --ImageReader.camera_model OPENCV \
                --ImageReader.single_camera 0 > /dev/null

            # Step 2: Matcher
            echo "   -> Exhaustive Matcher..."
            "$COLMAP_BIN" exhaustive_matcher \
                --database_path "$DATABASE_PATH" > /dev/null

            # Step 3: Mapper
            
            echo "   -> Glomap Mapper (using $GLOMAP_BIN)..."
            GLOMAP_OUTPUT_DIR="${KEYFRAME_DISTORTED_DIR}_glomap"
            mkdir -p "$GLOMAP_OUTPUT_DIR"

            "$GLOMAP_BIN" mapper \
                --database_path "$DATABASE_PATH" \
                --image_path    "$IMAGE_PATH" \
                --output_path   "$GLOMAP_OUTPUT_DIR" > /dev/null

            echo "   -> COLMAP Mapper (Refinement)..."
            "$COLMAP_BIN" mapper \
                --database_path "$DATABASE_PATH" \
                --image_path    "$IMAGE_PATH" \
                --input_path    "$GLOMAP_OUTPUT_DIR/0" \
                --output_path   "$KEYFRAME_DISTORTED_DIR" > /dev/null
            
            INPUT_MODEL_PATH="$KEYFRAME_DISTORTED_DIR/0"
        fi
        
        # 将此关键帧的路径保存为参考，供后续非关键帧使用
        echo "$INPUT_MODEL_PATH" > /tmp/last_keyframe_model_path
        
    else
        # === 非关键帧逻辑 ===
        # 不需要 Feature Extractor / Matcher / Mapper / Triangulator
        # 直接复用上一个关键帧的模型路径作为 Undistorter 的输入
        # 修正: 不在中间目录创建任何新文件
        
        if [ -f /tmp/last_keyframe_model_path ]; then
            INPUT_MODEL_PATH=$(cat /tmp/last_keyframe_model_path)
            echo ">> [Non-Keyframe $FRAME_IDX] Direct reuse of Reference Model: $INPUT_MODEL_PATH"
        else
            echo "Error: Non-keyframe $FRAME_IDX has no preceding keyframe reference!"
            echo "Skipping..."
            continue
        fi
    fi

    # 3. Image Undistortion (Always, for ALL frames)
    # 无论是关键帧还是非关键帧，都使用 INPUT_MODEL_PATH 中的参数来去畸变当前的 IMAGE_PATH
    echo ">> [Undistort] Running Image Undistorter..."
    
    # 最终输出目录必须创建
    mkdir -p "$FINAL_OUTPUT_DIR"
    
    "$COLMAP_BIN" image_undistorter \
        --image_path "$IMAGE_PATH" \
        --input_path "$INPUT_MODEL_PATH" \
        --output_path "$FINAL_OUTPUT_DIR" \
        --output_type COLMAP \
        --max_image_size 2000 > /dev/null

    echo "Finished $FRAME_NAME"
    echo ""
done

# 清理
rm -f /tmp/last_keyframe_model_path
echo "Keeping intermediate files (Keyframes Only) at $ROOT_DIR/intermediate_distorted"

echo "All Done!"