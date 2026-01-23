---
name: 4DGS Preprocessing Pipeline Design
description: Design guidelines and architecture for the 4DGS Gaussian Splatting data preprocessing project.
---

# 4DGS Preprocessing Pipeline Design

This skill outlines the architecture, module breakdown, and UI integration standards for the 4D Gaussian Splatting (4DGS) data preprocessing project.

## 1. Project Overview

The goal of this project is to preprocess video data for 4D Gaussian Splatting training. The pipeline transforms raw video and audio into calibrated camera parameters and background-masked image sequences.

### Key Functionalities
1.  **Video Confirmation**: User selection and validation of input video files.
2.  **Audio-Sync Frame Extraction**: Extraction of video frames based on audio synchronization, calculating precise floating-point frame positions.
3.  **Camera Calibration**: Using COLMAP and GLOMAP (via `glomap+colmap` module) to solve for camera intrinsics and extrinsics.
4.  **Background Removal**: User-defined 3D region selection followed by Gaussian training (LiteGS) to separate foreground from background, producing masks and clean training images.

## 2. Architecture & File Structure

The project follows a modular architecture where core logic is decoupled from the UI.

-   **Root Directory**: `/home/crgj/wdd/work/@AtWork/MirrorTime_Converter/`
-   **Modules**: All core logic resides in `modules/`. Each subdirectory is a self-contained module.
    -   `modules/audio_sync/`: Logic for video processing and frame extraction.
    -   `modules/glomap+colmap/`: Wrappers and logic for camera calibration.
    -   `modules/mask_LiteGS/`: Logic for segmentation, background removal, and Gaussian training.
-   **GUI**: `modules/gui/` contains the main application entry point (`main.py`) and UI components.
-   **Scripts**: `scripts/` for standalone utilities.

## 3. Module Design Guidelines

### General Principles
-   **Python Standard**: Follow PEP 8 naming conventions (`snake_case` for functions/variables, `CamelCase` for classes).
-   **Conciseness**: Keep code logic direct and efficient. Avoid over-engineering.
-   **Self-Contained**: Modules should have clear entry points (functions or classes) that can be called by the UI without exposing internal complexity.

### Component Details

#### A. Video & Audio Sync (`modules/audio_sync`)
-   **Input**: Video file path.
-   **Process**: Analyze audio to find sync points.
-   **Output**: Extracted frames folder, frame timestamps (float).

#### B. Calibration (`modules/glomap+colmap`)
-   **Input**: Extracted frames.
-   **Process**: Run COLMAP/GLOMAP algorithms.
-   **Output**: Sparse point cloud, camera parameters (intrinsics/extrinsics).

#### C. Background Removal (`modules/mask_LiteGS`)
-   **Input**: Images + Camera Parameters.
-   **Process**:
    1.  Render initial point cloud.
    2.  User selects 3D region (cylinder/box).
    3.  Train LiteGS to distinguish foreground/background.
    4.  Render masks.
-   **Output**: Masked images, mask files.

## 4. UI Integration Guidelines

The UI is the central command center for the pipeline.

-   **Framework**: Python-based GUI (e.g., Dear PyGui, PyQt, or custom Python UI framework as used in `modules/gui/main.py`).
-   **Language**: **English** for all labels, buttons, and logs.
-   **Integration Pattern**:
    1.  **Trigger**: Each pipeline step (1-4) should have a dedicated button or control group in the UI.
    2.  **Execution**: UI calls the corresponding module's function in a separate thread/process to avoid freezing.
    3.  **Monitoring**:
        -   **Progress**: Use callbacks or shared state to update progress bars in the UI.
        -   **Logs**: Stream stdout/stderr or internal logs to a UI console window.
    4.  **Visualization**: Display intermediate results in the UI (e.g., show extracted frames, render the sparse point cloud, show the generated masks).

## 5. Development Workflow

1.  **Implement Logic**: Write the core functionality in the respective `modules/` subdirectory.
2.  **Test Logic**: distinct from UI, ensure the module works via script or unit test.
3.  **Integrate UI**: Import the module in `modules/gui/main.py`. Connect the button signal to the module's entry point.
4.  **Add Feedback**: specific progress updates and success/failure notifications in the UI.
