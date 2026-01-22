import numpy as np
import argparse
import os
import sys
from plyfile import PlyData, PlyElement

# Add current directory to path to import litegs
sys.path.append(os.getcwd())
os.environ["QT_QPA_PLATFORM"] = "offscreen" #WDD 2026-01-02 避免Qt插件加载错误

from litegs.io_manager.colmap import load_frames
from litegs.io_manager.ply import load_ply, save_ply

from scipy.spatial import ConvexHull

def get_convex_hull_info(points, scale=1.0, center_offset=None, manual_radius=-1, manual_height=-1):
    """
    #WDD 2026-01-02 优化：通过添加虚拟端点强制凸包在轴向上具有足够高度（解决摄像头共面问题）
    #WDD 2026-01-22 Update: Support manual cylinder control (offset, radius, height)
    """
    if center_offset is None: center_offset = np.zeros(3)
    
    # 1. Base centroid from points
    centroid = np.mean(points, axis=0)
    
    # Apply manual offset to centroid
    centroid += np.array(center_offset)
    
    centered = points - centroid
    
    # 使用SVD估计圆柱轴线（法线方向）和水平面半径
    U, S, Vh = np.linalg.svd(centered)
    axis_vector = Vh[2] 
    
    # 计算摄像头距离轴心的平均半径，用作高度参考
    projections = centered - np.outer(centered @ axis_vector, axis_vector)
    radii = np.linalg.norm(projections, axis=1)
    
    # Determine Radius
    if manual_radius > 0:
        mean_radius = manual_radius
    else:
        mean_radius = np.mean(radii)
    
    # Determine Height (Half-height)
    if manual_height > 0:
        height_half = manual_height / 2.0
        # If manual height is set, we ignore "scale" for height, but still use scale for radius if manual_radius not set?
        # To keep it simple: manual_radius/height defines the base geometry. 'scale' arg is applied at the very end.
        # But if manual params are used, user likely expects exactness. 
        # Let's say: if manual params are set, we construct the hull to Match those params *before* scaling, 
        # or we treat manual params as the FINAL desired size (implying scale=1.0 effectively).
        # Decision: Use manual params to build the "unit" cylinder, then apply scale.
        # actually, if user gives radius=1.0, they expect 1.0. 
        # Let's apply scale normally. So if user wants exactly 1.0, they should set scale=1.0 or user input / scale.
        # Better: manual overrides are absolute. If set, they replace the auto-calc value used before scaling.
    else:
        height_half = mean_radius * 2 # Default logic
    
    # 构造变换后的点集
    height_offset = axis_vector * height_half
    
    # We construct a cylinder-like hull. 
    # To be robust, we project all points to the axis plane (circle), normalize them to radius, then extrude.
    # OR simpler: just add top/bottom virtual points at the Manual Radius distance?
    # No, 'points' are camera centers. We want to wrap them.
    # If we want to ENABLE manual control, we should probably ignore camera distribution if params are fixed?
    # BUT, axis_vector is still useful from cameras.
    
    # Let's stick to the previous 'virtual points' strategy but enforced with new radius/height
    # The previous code took 'points' (cameras), added virtual top/bottom, then scaled.
    # If we want a fixed cylinder, we might not want to depend on camera spread for the hull "width".
    # We should construct a synthetic cylinder instead.
    
    # Construct a synthetic cylinder vertices
    num_sides = 16
    angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
    
    # Create a circle on the plane perpendicular to axis_vector
    # Need two basis vectors perpendicular to axis_vector
    # axis_vector is normalized (from SVD Vh)
    
    v = axis_vector
    if abs(v[0]) > 0.9: 
        arbitrary = np.array([0, 1, 0])
    else: 
        arbitrary = np.array([1, 0, 0])
    
    u1 = np.cross(v, arbitrary)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    
    circle_points = []
    for theta in angles:
        p = mean_radius * (np.cos(theta) * u1 + np.sin(theta) * u2)
        circle_points.append(p)
    circle_points = np.array(circle_points)
    
    # Top and Bottom rings
    top_ring = circle_points + height_offset
    bottom_ring = circle_points - height_offset
    
    combined_points = np.vstack([top_ring, bottom_ring])
    
    # If we just use these points, the hull is a cylinder.
    # But wait, the original code combined 'centered' (cameras) with virtual points so the hull covers cameras + extension.
    # If user specifies radius/height, they likely want that EXACT cylinder, regardless of cameras.
    # So we should define the hull PURELY by these synthetic points.
    
    hull_points = combined_points
    
    # Apply scale (if user wants to shrink the MANUALLY defined cylinder via simple param? 
    # Usually manual > scale logic. Let's assume manual params are PRE-scale, 
    # OR simply: result = centroid + hull_points * scale. 
    # If user says radius=1.0, scale=0.8, result radius is 0.8.
    
    scaled_points = centroid + hull_points * scale
    
    hull = ConvexHull(scaled_points)
    return hull.equations, centroid, mean_radius, axis_vector

def filter_artifacts(xyz_np, sh_0_np, scale_np, opacity_np, centroid, axis_vector, mean_radius, k=25, std_ratio=2.0, brightness_thresh=0.015, opacity_thresh=0.02):
    """
    #WDD 2026-01-02 优化深度清理：核心绝对保护 + 边界强力清理
    针对用户反馈：
    1. 中间黑色被误删 -> 核心区黑色点绝对白名单。
    2. 边界小黑团没删掉 -> 边界区黑色点采用极其严苛的密度标准。
    """
    from scipy.spatial import cKDTree
    
    print(f"Spatially weighted deep cleaning (k={k}, std_ratio={std_ratio})...")
    
    # 1. 计算空间权重
    centered = xyz_np - centroid
    heights = centered @ axis_vector
    projections = centered - np.outer(heights, axis_vector)
    radial_dists = np.linalg.norm(projections, axis=1) # 点到轴线的径向距离
    
    # 放宽核心区范围到 50%，收紧边界区到 75%
    is_core_area = radial_dists < (mean_radius * 0.5)
    is_boundary_area = radial_dists > (mean_radius * 0.75)
    
    # 2. 计算属性
    C0 = 0.28209479177387814
    sh_dc = sh_0_np[0].T
    rgb = sh_dc * C0 + 0.5
    brightness = np.max(rgb, axis=1)
    
    tree = cKDTree(xyz_np)
    dists, _ = tree.query(xyz_np, k=k+1, workers=-1)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    
    # 3. 基础判定特征
    is_dark = brightness < brightness_thresh
    is_transparent = opacity_np < opacity_thresh
    is_foggy_scale = np.exp(scale_np).max(axis=0) > (np.mean(np.exp(scale_np).max(axis=0)) * 5)
    
    # 4. 动态阈值判定
    # 核心区：离群阈值极其宽松 (10倍 std)，基本不删
    # 边界区：离群阈值极其严格 (0.5倍 std)
    dynamic_std = np.full(xyz_np.shape[0], std_ratio)
    dynamic_std[is_core_area] = 10.0 
    dynamic_std[is_boundary_area] = 0.5 # 边界非常严格，稍有离群即删
    
    # 基础 SOR 删除掩码
    is_sor_outlier = mean_dists > (global_mean + dynamic_std * global_std)
    
    # 5. 针对性规则 (Override Rules)
    to_remove = is_sor_outlier.copy()
    
    # --- 规则 A: 边界小黑团强杀 ---
    # 如果在边界区，且是暗色或透明，必须非常致密才能存活
    # 判据：密度必须优于全局平均 (mean_dist < global_mean)，否则直接杀
    boundary_dark_kill = is_boundary_area & (is_dark | is_transparent) & (mean_dists > global_mean)
    to_remove[boundary_dark_kill] = True
    
    # --- 规则 B: 核心区绝对保护 ---
    # 只要在核心区，即使是离群点也不删（除极其离谱的飞出天际的点）
    # 尤其是暗色点，给予最高级别豁免
    # 豁免条件：(是核心区) 且 (不是超级巨大的雾块) 且 (不是超级离谱的距离 > 20 std)
    is_super_noise = is_foggy_scale | (mean_dists > (global_mean + 20.0 * global_std))
    to_remove[is_core_area & (~is_super_noise)] = False
    
    # 核心区的暗色点，即使是 Super noise 也再给一次机会（防止删掉黑衣服）
    # 除非真的离谱到家 (80倍平均距离)
    really_bad = mean_dists > (global_mean * 80)
    to_remove[is_core_area & is_dark & (~really_bad)] = False

    return ~to_remove

def filter_camera_proximity(xyz_np, scales_np, cam_centers, camera_radius):
    """
    #WDD 2026-01-02 考虑高斯大小的摄像机避让过滤
    如果高斯的包络球与摄像机避让球相交，则删除
    xyz_np: (N, 3) 
    scales_np: (3, N) 原始log scale
    """
    if camera_radius <= 0:
        return np.ones(xyz_np.shape[0], dtype=bool)
    
    from scipy.spatial import cKDTree
    print(f"Filtering points near cameras (camera_radius={camera_radius}, considering Gaussian scales)...")
    
    # 将 log scale 转换为实际 scale 值，并取三个轴的最大值作为包络球半径
    gs_radii = np.exp(scales_np).max(axis=0) # (N,)
    
    # 建立摄像机中心的KD树 (摄像机通常只有几十个，比高斯点少得多，查摄像机更快)
    cam_tree = cKDTree(cam_centers)
    
    # 查询每个高斯点离最近摄像机的距离
    # dists: 每个点到最近摄像机的距离
    dists, _ = cam_tree.query(xyz_np, workers=-1)
    
    # 判定逻辑：距离 < (摄像机避让半径 + 高斯半径)
    # 则发生相交，需要删除
    to_remove = dists < (camera_radius + gs_radii)
    
    return ~to_remove

def is_inside_convex_hull(points, equations):
    """
    Checks which points are inside the convex hull using its half-space equations.
    """
    # points: (N, 3), equations: (M, 4)
    # Result is (N,) boolean array. Point is inside if it satisfies ALL face equations.
    # To avoid memory issues with huge N, we process in chunks.
    n_points = points.shape[0]
    inside_mask = np.ones(n_points, dtype=bool)
    
    chunk_size = 100000
    for i in range(0, n_points, chunk_size):
        chunk = points[i:i+chunk_size]
        # (chunk_size, 3) @ (3, M) + (M,)
        # inside if all distances <= 1e-6 (buffer for float precision)
        distances = chunk @ equations[:, :3].T + equations[:, 3]
        inside_mask[i:i+chunk_size] = np.all(distances <= 1e-6, axis=1)
        
    return inside_mask

def main():
    parser = argparse.ArgumentParser(description="Clean GS PLY by Convex Hull and artifact filtering")
    parser.add_argument("-s", "--source_path", required=True, help="Colmap sparse directory (parent of sparse/0)")
    parser.add_argument("-i", "--image_dir", default="images", help="Image directory name")
    parser.add_argument("--input_ply", required=True, help="Input Gaussian PLY file")
    parser.add_argument("--output_ply", required=True, help="Output cleaned PLY file")
    parser.add_argument("--scale", type=float, default=0.8, help="Radius scale factor (default: 0.8)")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree of the input PLY")
    parser.add_argument("--clean", action="store_true", help="Enable artifact cleaning")
    parser.add_argument("--sor_k", type=int, default=20, help="K for SOR filtering")
    parser.add_argument("--sor_std", type=float, default=1.5, help="Std ratio for SOR filtering")
    parser.add_argument("--camera_radius", type=float, default=0.0, help="Ratio of cylinder radius for camera proximity filtering (e.g. 0.1)")
    
    # New arguments for manual control
    parser.add_argument("--center_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Offset for the cylinder center (x, y, z)")
    parser.add_argument("--cylinder_radius", type=float, default=-1.0, help="Manual cylinder radius override")
    parser.add_argument("--cylinder_height", type=float, default=-1.0, help="Manual cylinder height override")

    args = parser.parse_args()
    
    # #WDD 2026-01-02 读取摄像机参数并计算内角凸包，裁剪高斯点，过滤离群黑色噪声
    
    print(f"Loading cameras from {args.source_path}...")
    try:
        _, frames = load_frames(args.source_path, args.image_dir)
    except Exception as e:
        print(f"Error loading frames: {e}")
        return

    # Extract camera positions
    cam_centers = []
    for frame in frames:
        center = frame.get_camera_center()
        cam_centers.append(center)
    
    cam_centers = np.array(cam_centers)
    
    print(f"Found {len(cam_centers)} cameras. Computing convex hull...")
    print(f"Config: Offset={args.center_offset}, Radius={args.cylinder_radius}, Height={args.cylinder_height}")
    
    equations, centroid, mean_radius, axis_vector = get_convex_hull_info(
        cam_centers, 
        scale=args.scale, 
        center_offset=args.center_offset,
        manual_radius=args.cylinder_radius,
        manual_height=args.cylinder_height
    )
    print(f"Cylinder mean radius (after setup): {mean_radius:.4f}")
    print(f"Cylinder centroid: {centroid}")

    print(f"Loading PLY: {args.input_ply}")
    xyz, scale, rot, sh_0, sh_rest, opacity = load_ply(args.input_ply, args.sh_degree)
    
    # xyz is (3, N) in load_ply output
    xyz_np = xyz.transpose(1, 0) # (N, 3)
    
    print(f"Original point count: {xyz_np.shape[0]}")
    
    # 步骤1: 凸包裁剪
    spatial_mask = is_inside_convex_hull(xyz_np, equations)
    spatial_count = np.sum(spatial_mask)
    print(f"Spatial clipping: kept {spatial_count} / {xyz_np.shape[0]} points (removed {xyz_np.shape[0] - spatial_count})")
    
    # 应用裁剪掩码
    xyz_clipped = xyz[:, spatial_mask]
    scale_clipped = scale[:, spatial_mask]
    rot_clipped = rot[:, spatial_mask]
    sh_0_clipped = sh_0[:, :, spatial_mask]
    sh_rest_clipped = sh_rest[:, :, spatial_mask]
    opacity_clipped = opacity[:, spatial_mask]
    
    xyz_np_clipped = xyz_clipped.transpose(1, 0)
    
    # 步骤1.5: 摄像机近距离过滤
    if args.camera_radius > 0:
        #WDD 2026-01-02 将输入比例转换为实际距离
        actual_radius = args.camera_radius * mean_radius
        cam_mask = filter_camera_proximity(xyz_np_clipped, scale_clipped, cam_centers, actual_radius)
        cam_remove_count = np.sum(~cam_mask)
        print(f"Camera proximity: kept {np.sum(cam_mask)} / {spatial_count} points (removed {cam_remove_count})")
        
        xyz_clipped = xyz_clipped[:, cam_mask]
        scale_clipped = scale_clipped[:, cam_mask]
        rot_clipped = rot_clipped[:, cam_mask]
        sh_0_clipped = sh_0_clipped[:, :, cam_mask]
        sh_rest_clipped = sh_rest_clipped[:, :, cam_mask]
        opacity_clipped = opacity_clipped[:, cam_mask]
        xyz_np_clipped = xyz_clipped.transpose(1, 0)
        spatial_count = xyz_np_clipped.shape[0]

    # 步骤2: 噪声过滤 (在裁剪后的结果基础上执行)
    if args.clean:
        #WDD 2026-01-02 深度清理飞絮 (带空间权重)
        artifact_mask = filter_artifacts(
            xyz_np_clipped, sh_0_clipped, scale_clipped, opacity_clipped[0], 
            centroid, axis_vector, mean_radius,
            k=args.sor_k, std_ratio=args.sor_std
        )
        artifact_count = np.sum(artifact_mask)
        print(f"Deep artifact cleaning: kept {artifact_count} / {spatial_count} points (removed {spatial_count - artifact_count})")
        
        xyz_new = xyz_clipped[:, artifact_mask]
        scale_new = scale_clipped[:, artifact_mask]
        rot_new = rot_clipped[:, artifact_mask]
        sh_0_new = sh_0_clipped[:, :, artifact_mask]
        sh_rest_new = sh_rest_clipped[:, :, artifact_mask]
        opacity_new = opacity_clipped[:, artifact_mask]
    else:
        xyz_new = xyz_clipped
        scale_new = scale_clipped
        rot_new = rot_clipped
        sh_0_new = sh_0_clipped
        sh_rest_new = sh_rest_clipped
        opacity_new = opacity_clipped
    
    print(f"New point count: {xyz_new.shape[1]}")
    
    print(f"Saving to {args.output_ply}...")
    save_ply(args.output_ply, xyz_new, scale_new, rot_new, sh_0_new, sh_rest_new, opacity_new)
    print("Done.")

if __name__ == "__main__":
    main()
