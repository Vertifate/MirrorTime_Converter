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
from litegs.scene import cluster as cluster_utils

def main():
    parser = argparse.ArgumentParser(description="Generate masks from Gaussian Splatting model")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source directory (containing sparse/)")
    parser.add_argument("--input_ply", type=str, required=True, help="Path to the input PLY file")
    parser.add_argument("--image_dir", type=str, default="images", help="Name of the images directory (default: images)")
    parser.add_argument("--save_images_masked", action="store_true", help="Generate masked images")
    parser.add_argument("--save_images_color", action="store_true", help="Generate color-rendered images")
    parser.add_argument("--sh_degree", type=int, default=None, help="SH degree (auto-detect from PLY if not specified)")
    parser.add_argument("--resolution", "-r", type=int, default=1, help="Resolution downsampling factor (must match training, default: 1)")
    args = parser.parse_args()

    # Load cameras
    print(f"Loading cameras from {args.source_path}...")
    try:
        cameras, frames = load_frames(args.source_path, args.image_dir)
    except Exception as e:
        print(f"Error loading frames: {e}")
        return

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Auto-detect sh_degree from PLY if not specified
    if args.sh_degree is None:
        from plyfile import PlyData
        plydata = PlyData.read(args.input_ply)
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # Calculate sh_degree from number of f_rest features: 3*(sh_degree+1)^2 - 3
        num_rest = len(extra_f_names)
        sh_degree = int(np.sqrt((num_rest + 3) / 3)) - 1
        print(f"Auto-detected sh_degree={sh_degree} from PLY file")
    else:
        sh_degree = args.sh_degree
    
    # Load PLY
    print(f"Loading PLY from {args.input_ply}...")
    xyz, scale, rot, sh_0, sh_rest, opacity = load_ply(args.input_ply, sh_degree)
    
    # Process data to device
    xyz = torch.tensor(xyz, dtype=torch.float32, device=device).requires_grad_(False)
    scale = torch.tensor(scale, dtype=torch.float32, device=device).requires_grad_(False)
    rot = torch.tensor(rot, dtype=torch.float32, device=device).requires_grad_(False)
    sh_0 = torch.tensor(sh_0, dtype=torch.float32, device=device).requires_grad_(False)
    sh_rest = torch.tensor(sh_rest, dtype=torch.float32, device=device).requires_grad_(False)
    opacity = torch.tensor(opacity, dtype=torch.float32, device=device).requires_grad_(False)

    # Cluster points (Required for render_preprocess)
    print("Clustering points...")
    chunk_size = 256
    xyz, scale, rot, sh_0, sh_rest, opacity = cluster_utils.cluster_points(chunk_size, xyz, scale, rot, sh_0, sh_rest, opacity)

    # Prepare default params
    op = arguments.OptimizationParams.get_class_default_obj()
    pp = arguments.PipelineParams.get_class_default_obj()

    # Output directories
    output_dir = os.path.join(args.source_path, "masks")
    os.makedirs(output_dir, exist_ok=True)
    
    # Masked images directory
    masked_images_dir = None
    if args.save_images_masked:
        masked_images_dir = os.path.join(args.source_path, "images_masked")
        os.makedirs(masked_images_dir, exist_ok=True)
    
    # Color-rendered images directory
    color_images_dir = None
    if args.save_images_color:
        color_images_dir = os.path.join(args.source_path, "images_color")
        os.makedirs(color_images_dir, exist_ok=True)
    
    # Print output info
    outputs = [f"masks in {output_dir}"]
    if masked_images_dir:
        outputs.append(f"masked images in {masked_images_dir}")
    if color_images_dir:
        outputs.append(f"color images in {color_images_dir}")
    print(f"Generating {', '.join(outputs)}...")
    
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
                color_np = item.get('color_np')
                img_name = item['img_name']
                output_dir = item['output_dir']
                masked_images_dir = item.get('masked_images_dir')
                color_images_dir = item.get('color_images_dir')
                source_path = item['source_path']
                image_dir = item['image_dir']
                frame_name = item['frame_name']
                
                # Save Mask
                mask_to_save = (mask_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(mask_to_save)
                
                save_path = os.path.join(output_dir, img_name)
                img_pil.save(save_path)
                
                # Save color-rendered image if requested
                if color_images_dir and color_np is not None:
                    color_to_save = (color_np * 255).astype(np.uint8)
                    color_img_pil = Image.fromarray(color_to_save)
                    color_save_path = os.path.join(color_images_dir, img_name)
                    # Save as JPG for color images
                    color_save_path = os.path.splitext(color_save_path)[0] + ".jpg"
                    color_img_pil.save(color_save_path, quality=95)
    
                # --- Apply mask to original image (Only if requested) ---
                if masked_images_dir:
                    # Original image path
                    orig_img_path = os.path.join(source_path, image_dir, os.path.basename(frame_name))
                    if os.path.exists(orig_img_path):
                        # WDD Update: Load as RGBA or RGB, don't force lossy conversion yet
                        orig_img = Image.open(orig_img_path).convert("RGBA")
                        orig_np = np.array(orig_img).astype(np.float32) / 255.0 # (H, W, 4)
                        
                        # Resize mask if dimensions mismatch
                        # Note: mask_np is (H, W, 3) or (H, W)
                        if orig_np.shape[:2] != mask_np.shape[:2]:
                            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
                            mask_img = mask_img.resize((orig_img.width, orig_img.height), Image.BILINEAR)
                            mask_np_resized = np.array(mask_img).astype(np.float32) / 255.0
                            mask_to_use = mask_np_resized
                        else:
                            mask_to_use = mask_np
                        
                        # Ensure mask is single channel
                        if mask_to_use.ndim == 3 and mask_to_use.shape[2] >= 3:
                             alpha_channel = mask_to_use[:, :, 0]
                        else:
                             alpha_channel = mask_to_use

                        if alpha_channel.ndim == 2:
                            # Clamp values to ensure full opacity for foreground
                            # Values > 0.95 are forced to 1.0 to prevent darkening original pixels
                            alpha_channel[alpha_channel > 0.95] = 1.0
                            
                        # Prepare RGBA: RGB from original, A from mask
                        # Original might already have alpha, but we want to apply OUR mask
                        rgb_channels = orig_np[:, :, :3]
                        
                        # Combine: RGB + Mask Alpha
                        # Result is (H, W, 4)
                        masked_img_final_np = np.dstack([rgb_channels, alpha_channel])

                        masked_img_final = (masked_img_final_np * 255).astype(np.uint8)
                        masked_img_pil = Image.fromarray(masked_img_final, mode='RGBA')
                        
                        masked_save_path = os.path.join(masked_images_dir, img_name)
                        # Ensure WebP extension for efficient storage with transparency
                        masked_save_path = os.path.splitext(masked_save_path)[0] + ".webp"
                        # User requested highest quality: Lossless WebP
                        # Speed optimization: method=4 (balanced) vs 6 (slowest). Quality=100 + lossless ensures perfect pixels.
                        masked_img_pil.save(masked_save_path, format="WEBP", lossless=True, quality=100, method=4)
                    else:
                        # Just print warning, don't crash thread
                        pass
            
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
        
        # Apply resolution downsampling to match training resolution
        # IMPORTANT: Use round() to match training image loading (data.py:94)
        if args.resolution > 1:
            output_shape = (
                round(cam.height / args.resolution),
                round(cam.width / args.resolution)
            )
        else:
            output_shape = (cam.height, cam.width)
        
        try:
            # Preprocess: SH->Color conversion and culling
            _, culled_xyz, culled_scale, culled_rot, color, culled_opacity = render.render_preprocess(
                None, None, frustumplane, view_matrix,
                xyz, scale, rot, sh_0, sh_rest, opacity,
                op, pp, actived_sh_degree=sh_degree
            )

            # Render color image if requested
            color_np = None
            if color_images_dir:
                img_color, _, _, _, _ = render.render(
                    view_matrix, proj_matrix,
                    culled_xyz, culled_scale, culled_rot, color, culled_opacity,
                    actived_sh_degree=sh_degree, output_shape=output_shape, pp=pp
                )
                color_np = img_color[0].detach().cpu().numpy().transpose(1, 2, 0)
                color_np = np.clip(color_np, 0, 1)

            # Render mask (force color to white)
            color_white = torch.ones_like(color)
            img_mask, _, _, _, _ = render.render(
                view_matrix, proj_matrix,
                culled_xyz, culled_scale, culled_rot, color_white, culled_opacity,
                actived_sh_degree=0, output_shape=output_shape, pp=pp
            )
            
            # img shape: (1, 3, H, W)
            # Convert to numpy image
            mask_np = img_mask[0].detach().cpu().numpy().transpose(1, 2, 0) # (H, W, 3)
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
                'color_np': color_np,
                'img_name': img_name,
                'output_dir': output_dir,
                'masked_images_dir': masked_images_dir,
                'color_images_dir': color_images_dir,
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
