#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import re
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert MirrorTime data using Glomap and Colmap")
    parser.add_argument("root_dir", help="Root directory of the project")
    return parser.parse_args()

def get_binaries():
    """Get paths to local glomap and colmap binaries."""
    script_dir = Path(__file__).resolve().parent
    glomap_bin = script_dir / "glomap"
    colmap_bin = script_dir / "colmap"
    
    if not glomap_bin.exists():
        print(f"Warning: Local glomap binary not found at {glomap_bin}")
    if not colmap_bin.exists():
        print(f"Warning: Local colmap binary not found at {colmap_bin}")
        
    return str(glomap_bin), str(colmap_bin)

def run_command(cmd, desc=None):
    """Run a command. Capture output but print on error."""
    try:
        # Capture text=True to get strings instead of bytes
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        # If it matches the specific "not exist" error we saw in logs, strictly warn
        tqdm.write(f"\n[Error] Command failed: {' '.join(cmd)}")
        if desc:
            tqdm.write(f"Step: {desc}")
        tqdm.write(f"Exit code: {e.returncode}")
        # Print the last few lines of stderr for context
        if e.stderr:
            tqdm.write(f"Stderr:\n{e.stderr[-1000:]}")
        if e.stdout:
            # colmap sometimes prints errors to stdout
            tqdm.write(f"Stdout (last 5 lines):\n{e.stdout[-500:]}")
        raise

def ensure_model_structure(model_dir):
    """Ensure model directory has a '0' subdirectory containing the files."""
    if not model_dir.exists():
        return
        
    # Check if files are in root
    files_in_root = [f for f in model_dir.iterdir() if f.is_file() and f.name in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]]
    
    if files_in_root:
        # Move to 0 subdirectory
        subdir = model_dir / "0"
        subdir.mkdir(exist_ok=True)
        for f in files_in_root:
            shutil.move(str(f), str(subdir / f.name))

def copy_model(src, dst):
    """Copy model from src to dst, ensuring '0' subdirectory structure in dst."""
    if not src.exists():
        return
    
    # Determine output subdir
    dst_0 = dst / "0"
    dst_0.mkdir(parents=True, exist_ok=True)
    
    # Check if src has '0' or files in root
    src_files_dir = src
    if (src / "0").exists() and (src / "0" / "cameras.bin").exists(): # Check bin, could be txt
         src_files_dir = src / "0"
    elif (src / "0").exists() and (src / "0" / "cameras.txt").exists():
         src_files_dir = src / "0"
         
    # Copy relevant files
    for fname in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
        f_src = src_files_dir / fname
        if f_src.exists():
            shutil.copy2(f_src, dst_0 / fname)

def main():
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    input_root = root_dir / "input"
    output_root = root_dir / "images"
    
    glomap_bin, colmap_bin = get_binaries()

    if not input_root.exists():
        print(f"Error: Input directory '{input_root}' does not exist.")
        sys.exit(1)

    # Find and sort frame directories
    frame_dirs = []
    for d in input_root.iterdir():
        if d.is_dir() and d.name.startswith("frame"):
            frame_dirs.append(d)
    
    # Sort by extracting number from frame name (e.g., frame00123 -> 123)
    def extract_frame_idx(name):
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0
        
    frame_dirs.sort(key=lambda x: extract_frame_idx(x.name))
    
    print(f"Found {len(frame_dirs)} frames input '{input_root}'")

    # Temp file to store the last keyframe model path (mimicking shell script logic)
    # Using a variable in Python is cleaner than a temp file
    last_keyframe_model_path = None

    # Intermediate directory for cleanup info
    print(f"Keeping intermediate files (Keyframes Only) at {root_dir / 'tem_params'}")
    
    # Main progress bar for Frames
    pbar_frames = tqdm(frame_dirs, desc="Total Progress", unit="frame")
    
    # Global reference model path (calculated from First Frame if needed)
    global_reference_model_path = None
    first_frame_dir = frame_dirs[0] if frame_dirs else None
    
    def compute_model_for_frame(f_dir, f_name):
        """Helper to run the full reconstruction pipeline for a specific frame."""
        # Define paths
        f_keyframe_distorted_dir = root_dir / "tem_params" / f_name / "sparse"
        f_cache_distorted_dir = root_dir / "tem_params" / f_name
        f_database_path = f_cache_distorted_dir / "database.db"
        
        f_cache_distorted_dir.mkdir(parents=True, exist_ok=True)
        f_keyframe_distorted_dir.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        cached_model_0 = f_keyframe_distorted_dir / "0"
        if (cached_model_0 / "cameras.bin").exists() and (cached_model_0 / "images.bin").exists():
            return cached_model_0
        if (f_keyframe_distorted_dir / "cameras.bin").exists():
            return f_keyframe_distorted_dir
            
        # Run Pipeline
        tqdm.write(f">> [Compute] Generating Reference Model from {f_name}...")
        
        # 1. Feature Extraction
        run_command([
            colmap_bin, "feature_extractor",
            "--image_path", str(f_dir),
            "--database_path", str(f_database_path),
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "0"
        ], desc=f"Ref-FeatExt {f_name}")

        # 2. Exhaustive Matcher
        run_command([
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(f_database_path)
        ], desc=f"Ref-Match {f_name}")

        # 3. Glomap Mapper
        glomap_out = Path(str(f_keyframe_distorted_dir) + "_glomap")
        glomap_out.mkdir(parents=True, exist_ok=True)
        run_command([
            glomap_bin, "mapper",
            "--database_path", str(f_database_path),
            "--image_path", str(f_dir),
            "--output_path", str(glomap_out)
        ], desc=f"Ref-Glomap {f_name}")

        # 4. Colmap Mapper
        glomap_model_0 = glomap_out / "0"
        if not glomap_model_0.exists():
             tqdm.write(f"   [Error] Glomap output missing (Ref), trying root...")
             if (glomap_out / "cameras.bin").exists(): glomap_model_0 = glomap_out
        
        run_command([
            colmap_bin, "mapper",
            "--database_path", str(f_database_path),
            "--image_path", str(f_dir),
            "--input_path", str(glomap_model_0),
            "--output_path", str(f_keyframe_distorted_dir)
        ], desc=f"Ref-Colmap {f_name}")
        
        # Return valid path
        if (f_keyframe_distorted_dir / "0" / "cameras.bin").exists():
            return f_keyframe_distorted_dir / "0"
        return f_keyframe_distorted_dir


    # Prepare Global output paths
    global_distorted_dir = output_root / "sparse_distorted"
    global_undistorted_dir = output_root / "sparse"
    
    for frame_dir in pbar_frames:
        frame_name = frame_dir.name
        frame_idx = extract_frame_idx(frame_name)
        image_path = frame_dir
        final_output_dir = output_root / frame_name
        
        pbar_frames.set_postfix_str(f"Current: {frame_name}")
        
        # Priority 1: Check existing Global Distorted Model
        input_model_path = None
        
        # Check standard 0 subdirectory or root
        if (global_distorted_dir / "0" / "cameras.bin").exists():
            input_model_path = global_distorted_dir / "0"
        elif (global_distorted_dir / "cameras.bin").exists():
            input_model_path = global_distorted_dir
            
        # Priority 2: Use Global Reference (Frame 0)
        if not input_model_path:
            if not global_reference_model_path:
                if not first_frame_dir:
                    tqdm.write("Error: No frames found to compute reference!")
                    break
                # Compute (or load) Frame 0 model ONCE
                try:
                    global_reference_model_path = compute_model_for_frame(first_frame_dir, first_frame_dir.name)
                except Exception as e:
                     tqdm.write(f"   [Fatal] Failed to compute global reference from {first_frame_dir.name}: {e}")
                     break
            
            input_model_path = global_reference_model_path

        # 3. Image Undistorter
        if input_model_path and input_model_path.exists():
            final_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check valid model content
            if not list(input_model_path.iterdir()):
                 tqdm.write(f"   [Error] Input model directory is empty: {input_model_path}")
                 continue

            try:
                # 1. Copy to Global Distorted location if not present
                if not (global_distorted_dir / "0" / "cameras.bin").exists():
                    copy_model(input_model_path, global_distorted_dir)

                # 2. Run Undistorter
                # Output to final_output_dir (images/frame_name)
                # This creates images/frame_name/images and images/frame_name/sparse
                run_command([
                    colmap_bin, "image_undistorter",
                    "--image_path", str(image_path),
                    "--input_path", str(input_model_path),
                    "--output_path", str(final_output_dir),
                    "--output_type", "COLMAP"
                ])
                
                # 3. Handle Undistorted Model (Move local sparse to Global sparse)
                local_sparse = final_output_dir / "sparse"
                if local_sparse.exists():
                    # If global sparse doesn't exist, move/copy this one there
                    if not (global_undistorted_dir / "0" / "cameras.bin").exists():
                        ensure_model_structure(local_sparse) # Ensure it has 0/ structure locally first
                        copy_model(local_sparse, global_undistorted_dir)
                    
                    # User requested not to clean "result redundancy" in advance loop, 
                    # so we keep local_sparse for now or just skip deleting it.
                    # shutil.rmtree(local_sparse) 

            except Exception as e:
                tqdm.write(f"   [Fail] Undistortion failed for frame {frame_name}: {e}")
        else:
            tqdm.write(f"   [Skip] Invalid input model path for {frame_name}")
            
    # Final Cleanup
    print("Cleaning up intermediate reconstruction files...")
    tem_params_dir = root_dir / "tem_params"
    if tem_params_dir.exists():
        try:
            shutil.rmtree(tem_params_dir)
            print(f"Removed temporary directory: {tem_params_dir}")
        except Exception as e:
            print(f"Warning: Failed to remove {tem_params_dir}: {e}")

    print("\n[Post-Processing] Verifying sparse directory structure...")
    for frame_dir in output_root.iterdir():
        if frame_dir.is_dir():
            sparse_dir = frame_dir / "sparse"
            if sparse_dir.exists():
                ensure_model_structure(sparse_dir)

    print("\nAll Done!")

if __name__ == "__main__":
    main()
