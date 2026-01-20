import torch
import numpy as np
import argparse
import os
import sys
from PIL import Image
from tqdm import tqdm

# Ensure the script can find the litegs package
sys.path.append(os.getcwd())
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from litegs.io_manager.colmap import load_frames
from litegs.io_manager.ply import load_ply
from litegs import render
from litegs import arguments

def main():
    parser = argparse.ArgumentParser(description="Generate masks from Gaussian Splatting model")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source directory (containing sparse/)")
    parser.add_argument("--input_ply", type=str, required=True, help="Path to the input PLY file")
    parser.add_argument("--image_dir", type=str, default="images", help="Name of the images directory (default: images)")
    args = parser.parse_args()

    # Load cameras
    print(f"Loading cameras from {args.source_path}...")
    try:
        cameras, frames = load_frames(args.source_path, args.image_dir)
    except Exception as e:
        print(f"Error loading frames: {e}")
        return

    # Load PLY
    print(f"Loading PLY: {args.input_ply}")
    # load_ply returns numpy arrays
    # sh_degree 必须与 ply 文件中的属性匹配，通常是 3
    xyz, scale, rot, sh_0, sh_rest, opacity = load_ply(args.input_ply, sh_degree=3)
    
    # Check shapes and transpose if necessary to match what litegs.render expects
    # load_ply returns:
    # xyz: (3, N) -> Need (N, 3)
    # scale: (3, N) -> Need (N, 3)
    # rot: (4, N) -> Need (N, 4)
    # sh_0: (C, 1, N) or similar?
    # ply.py says: 
    # xyz=xyz.transpose(1,0)
    # scale=scale.transpose(1,0)
    # rot=rot.transpose(1,0)
    # sh_0=sh_0.transpose(2,1,0) -> (N, 1, 3) if input was (3, 1, N)
    
    # Wait, let's verify load_ply return shapes from previous knowledge or test.
    # In clean_gs_by_cylinder.py:
    # xyz, scale, rot, sh_0, sh_rest, opacity = load_ply(...)
    # xyz_np = xyz.transpose(1, 0) # so load_ply output xyz is (3, N)
    
    # Prepare tensors for rendering (Expects N, C)
    device = torch.device("cuda")
    
    xyz = torch.tensor(xyz, dtype=torch.float32, device=device) # (3, N)
    scale = torch.tensor(scale, dtype=torch.float32, device=device) # (3, N)
    rot = torch.tensor(rot, dtype=torch.float32, device=device) # (4, N)
    opacity = torch.tensor(opacity, dtype=torch.float32, device=device) # (1, N)

    # Define num_points strictly after xyz is created
    num_points = xyz.shape[1]

    # For white mask:
    # We want final color to be white (1, 1, 1).
    # sh_0 is used to calculate color (sh_0 * C0 + 0.5 = RGB).
    # RGB 1.0 -> SH_0 approx 1.7725
    white_sh_val = 0.5 / 0.28209479177387814
    
    # render_preprocess expects sh_0 as (1, 3, N) and sh_rest as (15, 3, N)
    # Based on wrapper.py line 521: sh_base: [1, num_channels, num_points]
    sh_0 = torch.full((1, 3, num_points), white_sh_val, dtype=torch.float32, device=device)
    
    # sh_rest as (15, 3, N)
    sh_rest = torch.zeros((15, 3, num_points), dtype=torch.float32, device=device)
    
    # Define PipelineParams
    class PipelineParams:
        def __init__(self):
            self.tile_size = (16, 16)
            self.enable_transmitance = False
            self.enable_depth = False
            self.sparse_grad = False
            self.cluster_size = None 
            
    pp = PipelineParams()
    op = None 
    
    # Output directories
    output_dir = os.path.join(args.source_path, "masks")
    os.makedirs(output_dir, exist_ok=True)
    
    # Masked images directory
    masked_images_dir = os.path.join(args.source_path, "images_masked")
    os.makedirs(masked_images_dir, exist_ok=True)
    
    print(f"Generating masks in {output_dir} and masked images in {masked_images_dir}...")
    
    # Start worker threads
    import queue
    import threading
    
    write_queue = queue.Queue()
    
    def worker():
        while True:
            item = write_queue.get()
            if item is None:
                break
            
            try:
                mask_np = item['mask_np']
                img_name = item['img_name']
                output_dir = item['output_dir']
                masked_images_dir = item['masked_images_dir']
                source_path = item['source_path']
                image_dir = item['image_dir']
                frame_name = item['frame_name']
                
                # Save Mask
                mask_to_save = (mask_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(mask_to_save)
                
                save_path = os.path.join(output_dir, img_name)
                img_pil.save(save_path)
    
                # --- Apply mask to original image ---
                # Original image path
                orig_img_path = os.path.join(source_path, image_dir, os.path.basename(frame_name))
                if os.path.exists(orig_img_path):
                    orig_img = Image.open(orig_img_path).convert("RGB")
                    orig_np = np.array(orig_img).astype(np.float32) / 255.0 # (H, W, 3)
                    
                    # Resize mask if dimensions mismatch (though they should match if cam params are correct)
                    if orig_np.shape[:2] != mask_np.shape[:2]:
                        # This shouldn't happen if resolution=1 in training, but just in case
                        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
                        mask_img = mask_img.resize((orig_img.width, orig_img.height), Image.BILINEAR)
                        mask_np_resized = np.array(mask_img).astype(np.float32) / 255.0
                        mask_to_use = mask_np_resized
                    else:
                        mask_to_use = mask_np
                    
                    # Create RGBA image
                    # color = orig_np (RGB)
                    # alpha = mask_to_use (L) - we use the mask as the alpha channel
                    
                    # Ensure mask is single channel for alpha concatenation if it's not already
                    if mask_to_use.ndim == 3 and mask_to_use.shape[2] == 3:
                         # If mask was RGB (white on black), take one channel
                         alpha_channel = mask_to_use[:, :, 0]
                    else:
                         alpha_channel = mask_to_use
    
                    # orig_np is (H, W, 3), alpha_channel is (H, W) or (H, W, 1)
                    if alpha_channel.ndim == 2:
                        alpha_channel = alpha_channel[:, :, np.newaxis]
                    
                    rgba_np = np.concatenate([orig_np, alpha_channel], axis=2) # (H, W, 4)
                    
                    masked_img_final = (rgba_np * 255).astype(np.uint8)
                    masked_img_pil = Image.fromarray(masked_img_final, mode='RGBA')
                    
                    masked_save_path = os.path.join(masked_images_dir, img_name)
                    # Ensure png extension for transparency support
                    masked_save_path = os.path.splitext(masked_save_path)[0] + ".png"
                    masked_img_pil.save(masked_save_path)
                else:
                    # Just print warning, don't crash thread
                    pass
                    # print(f"Warning: Original image not found at {orig_img_path}")
            
            except Exception as e:
                print(f"Error saving {item.get('img_name', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                write_queue.task_done()

    num_threads = 8
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Process frames
    for frame in tqdm(frames):
        cam = cameras[frame.camera_id]
        
        view_matrix = torch.tensor(frame.get_viewmatrix(), dtype=torch.float32, device=device).unsqueeze(0) # (1, 4, 4)
        proj_matrix = torch.tensor(cam.get_project_matrix(), dtype=torch.float32, device=device).unsqueeze(0) # (1, 4, 4)
        
        frustumplane = torch.zeros((1, 6, 4), dtype=torch.float32, device=device) # Dummy
        
        output_shape = (cam.height, cam.width)
        
        try:
            # We use render_preprocess to handle SH->Color conversion (though we fixed it to white)
            # and potential culling logic if any.
            _, culled_xyz, culled_scale, culled_rot, color, culled_opacity = render.render_preprocess(
                None, None, frustumplane, view_matrix,
                xyz, scale, rot, sh_0, sh_rest, opacity,
                op, pp, actived_sh_degree=0
            )

            # Explicitly force color to white
            color = torch.ones_like(color)

            # Render
            img, _, _, _, _ = render.render(
                view_matrix, proj_matrix,
                culled_xyz, culled_scale, culled_rot, color, culled_opacity,
                actived_sh_degree=0, output_shape=output_shape, pp=pp
            )
            
            # img shape: (1, 3, H, W)
            # Convert to numpy image
            mask_np = img[0].detach().cpu().numpy().transpose(1, 2, 0) # (H, W, 3)
            mask_np = np.clip(mask_np, 0, 1)
            
            img_name = os.path.basename(frame.name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                 img_name += ".png"
            else:
                 base = os.path.splitext(img_name)[0]
                 img_name = base + ".png"

            # Put to queue
            item = {
                'mask_np': mask_np,
                'img_name': img_name,
                'output_dir': output_dir,
                'masked_images_dir': masked_images_dir,
                'source_path': args.source_path,
                'image_dir': args.image_dir,
                'frame_name': frame.name
            }
            write_queue.put(item)
            
        except Exception as e:
            print(f"Error rendering frame {frame.name}: {e}")
            import traceback
            traceback.print_exc()

    # Wait for queue to drain
    print("Waiting for IO threads to finish...")
    write_queue.join()
    
    # Stop threads
    for _ in range(num_threads):
        write_queue.put(None)
    for t in threads:
        t.join()

    print("Done.")

if __name__ == "__main__":
    main()
