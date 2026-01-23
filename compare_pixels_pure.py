import os
from PIL import Image
import sys
import math

base_path = "/home/wyk/data/wanou/wdd-jpg/images/frame000001"
img_name = "001.jpg"
mask_name = "001.png"

img_path = os.path.join(base_path, "images", img_name)
mask_path = os.path.join(base_path, "masks", mask_name)
masked_path = os.path.join(base_path, "images_masked", img_name)

print(f"Loading {img_path}...")
orig_img = Image.open(img_path).convert("RGB") # Ensure RGB
print(f"Loading {masked_path}...")
masked_img = Image.open(masked_path).convert("RGB") # Ensure RGB

print(f"Loading {mask_path}...")
mask_img = Image.open(mask_path).convert("L") # Grayscale
# Use exact same resize logic as generate_masks.py
mask_img = mask_img.resize((orig_img.width, orig_img.height), Image.BILINEAR)

width, height = orig_img.size
print(f"Dimensions: {width}x{height}")

# Stats
count = 0
sum_diff = 0
max_diff = 0
diffs = []

step = 20 # Check every 20th pixel to be faster but dense enough

print("Comparing pixels in foreground (mask > 250)...")

for y in range(0, height, step):
    for x in range(0, width, step):
        m = mask_img.getpixel((x, y))
        
        # Only check where mask is white (foreground)
        if m > 250:
            p_orig = orig_img.getpixel((x, y))
            p_masked = masked_img.getpixel((x, y))
            
            # Distance
            d = sum([abs(p_orig[i] - p_masked[i]) for i in range(3)])
            # Average difference per channel
            d_avg = d / 3.0
            
            sum_diff += d_avg
            if d_avg > max_diff:
                max_diff = d_avg
            
            diffs.append(d_avg)
            count += 1

if count == 0:
    print("No foreground pixels found to compare (with step).")
else:
    mean_diff = sum_diff / count
    diffs.sort()
    p99_diff = diffs[int(len(diffs) * 0.99)]
    
    print(f"--- Foreground Stats (Sampled {count} pixels) ---")
    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"Max Difference: {max_diff:.4f}")
    print(f"99th Percentile Diff: {p99_diff:.4f}")

    if mean_diff > 5:
        print("\nSignificant difference detected!")
    else:
        print("\nDifferences are small (likely compression noise).")
