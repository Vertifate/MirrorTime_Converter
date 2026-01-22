import os
from PIL import Image


base_path = "/home/wyk/data/wanou/wdd-jpg/images/frame000001"
img_name = "001.jpg"
mask_name = "001.png"

img_path = os.path.join(base_path, "images", img_name)
mask_path = os.path.join(base_path, "masks", mask_name)
masked_path = os.path.join(base_path, "images_masked", img_name)

print(f"Checking files for {img_name}...")

if os.path.exists(img_path):
    img = Image.open(img_path)
    print(f"Original Image: {img.size} mode={img.mode}")
else:
    print(f"Original Image NOT FOUND: {img_path}")

if os.path.exists(mask_path):
    mask = Image.open(mask_path)
    print(f"Mask: {mask.size} mode={mask.mode}")
else:
    print(f"Mask NOT FOUND: {mask_path}")

if os.path.exists(masked_path):
    masked = Image.open(masked_path)
    print(f"Masked Image: {masked.size} mode={masked.mode}")
else:
    print(f"Masked Image NOT FOUND: {masked_path}")
