import os
import sys

def convert_cameras_txt(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Processing: {file_path}")
    
    new_lines = []
    modified_count = 0
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            new_lines.append(line)
            continue
            
        parts = line.split()
        # Colmap camera line format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
        if len(parts) >= 4:
            cam_id = parts[0]
            model = parts[1]
            width = parts[2]
            height = parts[3]
            params = parts[4:]
            
            if model == "SIMPLE_PINHOLE":
                # SIMPLE_PINHOLE params: f, cx, cy
                # PINHOLE params: fx, fy, cx, cy
                if len(params) == 3:
                    f = params[0]
                    cx = params[1]
                    cy = params[2]
                    
                    # Construct new line
                    # fx = f, fy = f
                    new_line = f"{cam_id} PINHOLE {width} {height} {f} {f} {cx} {cy}"
                    new_lines.append(new_line)
                    modified_count += 1
                    print(f"  Converted Camera {cam_id}: SIMPLE_PINHOLE -> PINHOLE")
                else:
                    print(f"  Warning: Camera {cam_id} is SIMPLE_PINHOLE but has {len(params)} params (expected 3). Skipping.")
                    new_lines.append(line)
            else:
                # Keep other models as is
                new_lines.append(line)
        else:
            new_lines.append(line)

    if modified_count > 0:
        with open(file_path, 'w') as f:
            for line in new_lines:
                f.write(line + "\n")
        print(f"Done. Modified {modified_count} cameras.")
    else:
        print("No SIMPLE_PINHOLE cameras found or modified.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default path requested by user
        default_path = "/home/wyk/data/wanou/wdd-jpg/input/sparse/0/cameras.txt"
        print(f"No path provided, using specific default: {default_path}")
        convert_cameras_txt(default_path)
    else:
        convert_cameras_txt(sys.argv[1])
