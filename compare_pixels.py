import os
from PIL import Image
import sys

# Try importing numpy, handle failure
try:
    import numpy as np
except ImportError:
    print("Numpy not found, cannot do pixel math easily. Please install numpy.")
    sys.exit(1)

base_path = "/home/wyk/data/wanou/wdd-jpg/images/frame000001"
img_name = "001.jpg"
mask_name = "001.png"

img_path = os.path.join(base_path, "images", img_name)
mask_path = os.path.join(base_path, "masks", mask_name)
masked_path = os.path.join(base_path, "images_masked", img_name)

print(f"Loading {img_path}...")
orig_img = Image.open(img_path).convert("RGB")
orig_np = np.array(orig_img).astype(np.float32)

print(f"Loading {masked_path}...")
masked_img = Image.open(masked_path).convert("RGB")
masked_np = np.array(masked_img).astype(np.float32)

print(f"Loading {mask_path}...")
mask_img = Image.open(mask_path).convert("L") # Read as grayscale
# Use exact same resize logic as generate_masks.py
mask_img = mask_img.resize((orig_img.width, orig_img.height), Image.BILINEAR)
mask_np = np.array(mask_img).astype(np.float32) / 255.0

# Define "foreground" as where mask is almost 1.0 (to avoid edge artifacts in stats)
foreground_mask = mask_np > 0.99

if np.sum(foreground_mask) == 0:
    print("Warning: No fully foreground pixels found (mask > 0.99). Reducing threshold to 0.5")
    foreground_mask = mask_np > 0.5

# Calculate stats in foreground
diff = np.abs(orig_np - masked_np)
mean_diff = np.mean(diff[foreground_mask])
max_diff = np.max(diff[foreground_mask])
p99_diff = np.percentile(diff[foreground_mask], 99)

print(f"--- Foreground Stats (Mask > {0.99 if np.sum(mask_np > 0.99) > 0 else 0.5}) ---")
print(f"Pixel Count: {np.sum(foreground_mask)}")
print(f"Mean Difference: {mean_diff:.4f}")
print(f"Max Difference: {max_diff:.4f}")
print(f"99th Percentile Diff: {p99_diff:.4f}")

# Check for color shift (Mean difference per channel)
diff_signed = masked_np - orig_np
mean_diff_ch = np.mean(diff_signed[foreground_mask], axis=0)
print(f"Mean Shift per Channel (R, G, B): {mean_diff_ch}")
