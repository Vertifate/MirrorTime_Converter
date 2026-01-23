
import torch
import sys
import os
import time
import numpy as np

# Add local path to sys.path to find litegs module
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'modules/mask_LiteGS'))

from litegs.utils import wrapper

def test_wrapper_fix():
    print("Testing wrapper.py fix with 3D tensors...")
    
    # Mock data dimensions
    N = 100
    
    # Create 3D tensors (simulating the problematic input)
    # Case 1: Extra singleton dimensions [1, 3, N] or [3, 1, N]
    scale_3d = torch.rand(1, 3, N).cuda()
    rot_3d = torch.rand(1, 4, N).cuda() # Wait, rot is usually 4 dims
    
    # Real case from previous error: "TensorAccessor expected 2 dims but tensor has 3"
    # This implies input was 3D.
    # Let's try to pass 3D tensors to the fused wrapper
    
    try:
        # Use our wrapper function (which we patched)
        # Note: wrapper.CreateTransformMatrix.call does: 
        # return __create_transform_matrix_fused(scaling_vec, rotator_vec)
        
        # We need to call the method that eventually calls __create_transform_matrix_fused
        # In wrapper.py: CreateTransformMatrix.call = __create_transform_matrix_fused
        
        # Let's construct arguments that match the problematic shape
        # Assume scale is [3, N] but somehow got an extra dim -> [1, 3, N]
        scale_bad = torch.rand(1, 3, N).float().cuda()
        rot_bad = torch.rand(1, 4, N).float().cuda()
        
        print(f"Passing scale shape: {scale_bad.shape}, rot shape: {rot_bad.shape}")
        
        result = wrapper.CreateTransformMatrix.call(scale_bad, rot_bad)
        print("Successfully computed TransformMatrix with 3D input!")
        print(f"Result shape: {result.shape}")
        
    except Exception as e:
        print(f"Failed to handle 3D input: {e}")
        import traceback
        traceback.print_exc()

def test_ply_loading():
    print("\nTesting PLY loading speed...")
    ply_path = "/home/wyk/data/wanou/wdd-jpg/images/tem/frame000008/point_cloud/finish/point_cloud_cleaned.ply"
    
    if not os.path.exists(ply_path):
        print(f"PLY file not found at {ply_path}")
        return

    try:
        from litegs.io_manager.ply import load_ply
        start_time = time.time()
        print(f"Loading {ply_path} (size: {os.path.getsize(ply_path)/1024/1024:.2f} MB)...")
        # Load PLY (sh_degree=3)
        xyz, scale, rot, sh_0, sh_rest, opacity = load_ply(ply_path, 3)
        end_time = time.time()
        print(f"Loaded PLY in {end_time - start_time:.2f} seconds.")
        print(f"XYZ Shape: {xyz.shape}")
        
    except Exception as e:
        print(f"Failed to load PLY: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wrapper_fix()
    test_ply_loading()
