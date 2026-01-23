import os
from PIL import Image
import sys

base_path = "/home/wyk/data/wanou/wdd-jpg/images/frame000001"
img_name = "001.jpg"
mask_name = "001.png"

img_path = os.path.join(base_path, "images", img_name)
mask_path = os.path.join(base_path, "masks", mask_name)
masked_path = os.path.join(base_path, "images_masked", img_name)

orig_img = Image.open(img_path).convert("RGB")
masked_img = Image.open(masked_path).convert("RGB")
mask_img = Image.open(mask_path).convert("L")

# Resize mask to check the values used in computation
mask_resized = mask_img.resize(orig_img.size, Image.BILINEAR)

print(f"Checking Mask Values in 'Foreground' (center pixel)...")
w, h = orig_img.size
center_x, center_y = w // 2, h // 2

# Check a central region 10x10
max_val = 0
min_val = 255
for y in range(center_y - 5, center_y + 5):
    for x in range(center_x - 5, center_x + 5):
        m = mask_resized.getpixel((x, y))
        max_val = max(max_val, m)
        min_val = min(min_val, m)

print(f"Central Mask Range: {min_val} - {max_val}")

print("\nChecking Signed Difference (Orig - Masked) in center...")
pos_diff = 0
neg_diff = 0
zero_diff = 0

for y in range(center_y - 50, center_y + 50):
    for x in range(center_x - 50, center_x + 50):
        if mask_resized.getpixel((x, y)) > 250:
            po = orig_img.getpixel((x, y))
            pm = masked_img.getpixel((x, y))
            # Check Green channel for simplicity
            diff = po[1] - pm[1]
            if diff > 1: pos_diff += 1
            elif diff < -1: neg_diff += 1
            else: zero_diff += 1

print(f"Pixels where Orig > Masked: {pos_diff}")
print(f"Pixels where Orig < Masked: {neg_diff}")
print(f"Pixels where Orig ~= Masked: {zero_diff}")

if pos_diff > neg_diff * 10:
    print("\nCONCLUSION: The masked image is consistently DARKER than original.")
    print("This confirms the mask is not perfectly 1.0 (white).")
elif neg_diff > pos_diff * 10:
    print("\nCONCLUSION: The masked image is consistently BRIGHTER than original.")
else:
    print("\nCONCLUSION: Differences are random (Compression/Noise).")
