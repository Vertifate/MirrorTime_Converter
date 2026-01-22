#!/usr/bin/env python3
import struct
import sys

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            # PINHOLE model has 4 parameters: fx, fy, cx, cy
            num_params = 4
            params = struct.unpack(f"<{num_params}d", fid.read(8*num_params))
            
            cameras[camera_id] = {
                'id': camera_id,
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

if __name__ == "__main__":
    cameras = read_cameras_binary("/home/wyk/data/wanou/wdd-jpg/images/frame000001/sparse/0/cameras.bin")
    for cam_id, cam in cameras.items():
        fx, fy, cx, cy = cam['params']
        w, h = cam['width'], cam['height']
        print(f"Camera {cam_id}:")
        print(f"  Resolution: {w} x {h}")
        print(f"  fx={fx:.2f}, fy={fy:.2f}")
        print(f"  cx={cx:.2f}, cy={cy:.2f}")
        print(f"  Expected center: cx_ideal={w/2:.2f}, cy_ideal={h/2:.2f}")
        print(f"  Offset: dx={cx - w/2:.2f}, dy={cy - h/2:.2f}")
