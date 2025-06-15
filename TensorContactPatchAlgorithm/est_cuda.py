import open3d as o3d
import numpy as np
from replay_realsense_tensor import read_RGB_D_folder
import time 
import cv2
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
import optix
import cupy as cp

from optix_castrays import OptiXRaycaster
#import kaolin as kao
import torch
import torch.nn.functional as F

width = 848
height = 480
border = 3 #pixel border
np_mask_invalid_points = np.ones((height,width), dtype = np.uint8)
np_mask_invalid_points[border:height-border,border:width-border] = 0
mask_invalid_points = cv2.cuda.GpuMat(rows = height, cols = width,type = cv2.CV_8U)
mask_invalid_points.upload(np_mask_invalid_points)

img = mask_invalid_points.download() * 255
cv2.imshow("mask",img)
cv2.waitKey(0)  
cv2.destroyAllWindows()

# ply_path = 'full_outer_inner_part_only.ply'
# pcd = o3d.t.io.read_point_cloud(ply_path)
# pcd.scale(scale = 0.03912, center = [0,0,0])
# pcd.estimate_normals()
# pcd.orient_normals_consistent_tangent_plane(k = 10)
# mesh = o3d.t.io.read_triangle_mesh(ply_path)
# mesh.scale(scale = 0.03912, center = [0,0,0])
# mesh.vertex.normals = pcd.point.normals
# #mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()
# mesh.normalize_normals()

# o3d.visualization.draw_geometries([pcd.to_legacy()])

# plane_coords = np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1]], dtype=np.float32)
# plane = o3d.t.geometry.TriangleMesh()
# plane.vertex.positions = o3d.core.Tensor.from_numpy(plane_coords)
# plane.triangle.indices = o3d.core.Tensor(np.array([[0,1,2],[1,2,3]]),dtype = o3d.core.int32)
# mesh = o3d.t.io.read_triangle_mesh("full_outer_outer_part_only.ply")
# mesh.scale(scale = 0.03912, center = [0,0,0])
# mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()

# mesh1 = o3d.t.io.read_triangle_mesh("full_outer_inner_part_only.ply")
# mesh1.scale(scale = 0.03912, center = [0,0,0])
# mesh1.compute_vertex_normals()
# mesh1.compute_triangle_normals()
# # o3d.visualization.draw([mesh1, mesh])
# # o3d.visualization.draw([plane, mesh])

# pcd = o3d.t.io.read_point_cloud("full_outer_outer_part_only.ply")
# pcd.estimate_normals()
# print(pcd.covariances)
# pcd = pcd.cuda()
# pcd.scale(scale = 0.03912, center = [0,0,0])



# #pcd = pcd.uniform_down_sample(every_k_points = 10)
# points = torch.utils.dlpack.from_dlpack(pcd.point.positions.contiguous().to_dlpack())

# def split_pointcloud_into_batches(points, batch_size):
#     """
#     Splits a (M, 3) point cloud into batches of shape (B, N, 3).

#     Args:
#         points (torch.Tensor): shape (M, 3)
#         batch_size (int): number of points per batch (N)

#     Returns:
#         torch.Tensor: shape (B, N, 3)
#     """
#     M = points.shape[0]
#     # Truncate M so it's divisible by batch_size
#     M_trunc = (M // batch_size) * batch_size
#     points = points[:M_trunc]

#     # Reshape to (B, N, 3)
#     batched_points = points.view(-1, batch_size, 3)
#     return batched_points

# # Example usage
# #points = torch.randn(100000, 3)  # M = 100,000
# batch_size = 2048*2*2*2*2*2*2*2*2 *2*2         # N = 2048 points per batch

# batched = split_pointcloud_into_batches(points, batch_size)  # shape (B, 2048, 3)
# print(batched.shape) 
# points = points.unsqueeze(0)  # shape: [1, N, 3]

# # Set voxel size
# voxel_size = 0.003  # in meters

# # Compute resolution from bounding box
# min_bound = points.min(dim=1)[0]
# max_bound = points.max(dim=1)[0]
# extent = max_bound - min_bound
# resolution = 128 #int(torch.ceil(extent.max() / voxel_size *extent.max()).item())
# print(resolution)

# def create_3d_sobel_kernels():

#     sobel_x = torch.tensor([
#         [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
#         [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
#         [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]
#     ], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)

#     sobel_y = sobel_x.permute(0,1,3,2,4)  # Rotate to Y
#     sobel_z = sobel_x.permute(0,1,4,3,2)  # Rotate to Z

#     return sobel_x, sobel_y, sobel_z

# sobel_x, sobel_y, sobel_z = create_3d_sobel_kernels()
# # Get kernels
# Gx, Gy, Gz = create_3d_sobel_kernels()

# for i in range(10):
#     t1 = time.perf_counter()
#     # spc = kao.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(pointcloud=points,
#     #                                                                     level=11,
#     #                                                                     )
#     # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(leg_pcd,
#     #                                                             voxel_size=0.05)
#     vox = kao.ops.conversions.pointclouds_to_voxelgrids(batched,
#                                                         resolution=resolution,
#                                                         origin=None,
#                                                         scale=None,
#                                                         return_sparse=False
#                                                         )   
    
#     # Dummy voxel data
#     voxel = vox

    

#     # Convolve
#     grad_x = F.conv3d(voxel, Gx, padding=1)
#     grad_y = F.conv3d(voxel, Gy, padding=1)
#     grad_z = F.conv3d(voxel, Gz, padding=1)

#     # Gradient magnitude
#     grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

#     print("Gradient magnitude shape:", grad_mag.shape)

#     t2 = time.perf_counter()
#     #print(vox)
#     # print(f'SPC keeps track of the following cells in levels of detail (parents + leaves):\n'
#     #       f' {spc.point_hierarchies}\n')
#     print("Voxel time:", t2-t1)
#     threshold = grad_mag.mean() + 2 * grad_mag.std() # tune this threshold
#     mask = grad_mag.squeeze() > threshold

#     coords = torch.nonzero(mask).float()  # shape (K, 3)

#     # Convert to Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())

#     o3d.visualization.draw_geometries([pcd])
# #o3d.visualization.draw_geometries([voxel_grid])
# num_partitions = pcd.pca_partition(max_points=40000)

# # print the partition ids and the number of points for each of them.
# print(np.unique(pcd.point.partition_ids.numpy(), return_counts=True))

# pcd1 = pcd.select_by_mask(pcd.point.partition_ids == 1)
# pcd2 = pcd.select_by_mask(pcd.point.partition_ids == 2)
# pcd3 = pcd.select_by_mask(pcd.point.partition_ids == 3)
# pcd4 = pcd.select_by_mask(pcd.point.partition_ids == 4)
# pcd5 = pcd.select_by_mask(pcd.point.partition_ids == 160)
# pcd6 = pcd.select_by_mask(pcd.point.partition_ids == 161)
# pcd7 = pcd.select_by_mask(pcd.point.partition_ids == 162)
# pcd8 = pcd.select_by_mask(pcd.point.partition_ids == 163)

# o3d.visualization.draw([pcd1,pcd2,pcd3,pcd4,pcd5,pcd6,pcd7,pcd8])

# # 1. Initialize once (outside your frame loop)
# raycaster = OptiXRaycaster("full_outer_inner_part_only.ply","full_outer_outer_part_only.ply", "raycast.cu", stream=cp.cuda.Stream())
# scale = 0.03912
# # outer = o3d.t.io.read_triangle_mesh("full_outer_inner_part_only.ply")
# # outer.scale(scale = scale, center = [0,0,0])
# # outer.compute_vertex_normals()
# # origins = outer.vertex.positions.numpy()
# # directions = outer.vertex.normals.numpy()
# outer = o3d.t.io.read_point_cloud("full_outer_inner_part_only.ply")
# mesh = o3d.t.io.read_triangle_mesh("full_outer_outer_part_only.ply")
# mesh.scale(scale = 0.03912, center = [0,0,0])
# mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()
# o3d.visualization.draw_geometries([mesh.to_legacy()])
# outer.scale(scale = scale, center = [0,0,0])
# outer.estimate_normals()
# #outer.orient_normals_consistent_tangent_plane(k = 10)

# outer = outer.cuda()
# origins = cp.from_dlpack(outer.point.positions.to_dlpack())
# directions = cp.from_dlpack(outer.point.normals.to_dlpack())
# # origins = np.asarray(outer.points)
# # directions = -np.asarray(outer.normals)
# # o3d.visualization.draw_geometries([outer,mesh.to_legacy()])

# rays = cp.concatenate([origins, -directions], axis=1)
# print(rays)
# hit_point, tri_id, t_hit = raycaster.cast(rays)
# print(hit_point)
# print(tri_id)
# print(t_hit)

# r_pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
# r_pcd.point.positions = o3d.core.Tensor.from_dlpack(hit_point.toDlpack())
# o3d.visualization.draw([r_pcd.cpu()])
# 2. In your frame loop

# imageStream = read_RGB_D_folder('realsense2',starting_index=150,depth_num=3,debug_mode=False)
# while imageStream.has_next():
#     start_cv2 = cv2.getTickCount()
#     if imageStream.has_next():
#         count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu= imageStream.get_next_frame()
#         origins = t2cam_pcd_cuda.point.positions.cpu().numpy()
#         directions = t2cam_pcd_cuda.point.normals.cpu().numpy()
#         rays = np.concatenate([origins, directions], axis=1)  # (N, 6)
#         hit_point, tri_id, t_hit = raycaster.cast(rays)
#     #t2cam_pcd.to_legacy()
#     end_cv2 = cv2.getTickCount()
#     time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     print("FPS:", time_sec)


# def load_mesh_from_ply(path):
#     mesh = o3d.io.read_triangle_mesh(path)
#     mesh.compute_vertex_normals()
#     vertices = np.asarray(mesh.vertices, dtype=np.float32)
#     triangles = np.asarray(mesh.triangles, dtype=np.uint32)
#     return vertices, triangles


# def prepare_rays(origins, directions):
#     n = origins.shape[0]
#     rays = np.zeros(n, dtype=optix.RayHit.dtype)
#     rays['origin'] = origins
#     rays['direction'] = directions
#     rays['tmin'] = 0.0
#     rays['tmax'] = 1e10
#     return rays


# def raycast_with_optix(vertices, triangles, rays):
#     ctx = optix.DeviceContext()

#     # Acceleration structure (BVH)
#     gas = optix.GeometryAccelerationStructure(ctx)
#     gas.set_triangles(vertices, triangles)
#     gas.build()

#     # Minimal raygen, miss, and hit programs
#     ptx = optix.utils.minimal_ptx()
#     module = ctx.create_module(ptx)
#     raygen = module.create_raygen_program("__raygen__minimal")
#     miss = module.create_miss_program("__miss__default")
#     hit = module.create_hitgroup_program("__closesthit__default")

#     pipeline = ctx.create_pipeline([raygen], miss, [hit])

#     # Output buffer
#     hit_results = np.zeros(len(rays), dtype=np.float32)

#     # Launch
#     sbt = pipeline.create_shader_binding_table()
#     pipeline.launch(sbt, rays, output=hit_results, width=len(rays), height=1)

#     return hit_results


# if __name__ == "__main__":
#     # Load mesh
#     mesh_path = "your_mesh.ply"
#     vertices, triangles = load_mesh_from_ply(mesh_path)

#     # Generate 400,000 rays (pointing along Z)
#     N = 400_000
#     origins = np.random.uniform(-1, 1, size=(N, 3)).astype(np.float32)
#     directions = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (N, 1))

#     rays = prepare_rays(origins, directions)

#     # Run OptiX raycasting
#     hit_dists = raycast_with_optix(vertices, triangles, rays)

#     # Inspect results
#     print("Hits:", np.count_nonzero(np.isfinite(hit_dists)))
#     print("Closest hit distance:", hit_dists[np.isfinite(hit_dists)].min())

# file_name = 'inner_to_outer_correspondences.npy'
# with open(file_name, 'rb') as f:
#     rays_hit_start_io = np.load(f)
#     rays_hit_end_io = np.load(f)

# model_pcd = o3d.t.geometry.PointCloud()
# model_pcd.point.positions = o3d.core.Tensor(rays_hit_start_io)

# scale = 0.03912
# file_path = 'full_outer_inner_part_only.ply'
# mesh = o3d.io.read_triangle_mesh(filename=file_path, print_progress = True)
# mesh.scale(scale = scale, center = [0,0,0])

# scene = o3d.t.geometry.RaycastingScene()
# scene.add_triangles(mesh)

# imageStream = read_RGB_D_folder('realsense2',starting_index=150,depth_num=3,debug_mode=False)
# while imageStream.has_next():
#     start_cv2 = cv2.getTickCount()
#     if imageStream.has_next():
#         count, depth_image, color_image, t2cam_pcd_cuda, vertex_map_gpu, normal_map_gpu= imageStream.get_next_frame()
#     #t2cam_pcd.to_legacy()
#     end_cv2 = cv2.getTickCount()
#     time_sec = (end_cv2-start_cv2)/cv2.getTickFrequency()
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#     print("FPS:", time_sec)

# intrinsic = o3d.core.Tensor([[433.4320983886719,0,416.3398132324219],[0,432.88079833984375,238.8269500732422],[0,0,1]]).cuda()
# #print(np.asarray(intrinsic))
# intrinsic_nt = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
# #print(intrinsic_nt)

# if imageStream.has_next():
#     count, depth_image, color_image, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
# if imageStream.has_next():
#     count1, depth_image1, color_image1, rgbd_image1, t2cam_pcd1 = imageStream.get_next_frame()

# t0 = time.time()
# depth_o3d = o3d.t.geometry.Image(depth_image).cuda()
# print(depth_image)
# color_o3d = o3d.t.geometry.Image(color_image).cuda()
# RGBD = o3d.t.geometry.RGBDImage(color_o3d,depth_o3d,10000)
# pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic)
# pcd_cpu = pcd.cpu()
# #pcd.estimate_normals() #bad for gpu
# #pcd.orient_normals_consistent_tangent_plane(k=10)
# #pcd.uniform_down_sample(every_k_points=10)
# pcd2 = pcd.clone()
# pcd2.translate(o3d.core.Tensor([0.01,0.1,0.01]))
# pcd2.estimate_normals()
# t1 = time.time()
# #filtered_depth = depth_o3d.filter_bilateral(kernel_size = 7, value_sigma= 10, dist_sigma = 20.0)
# vertex_map = depth_o3d.create_vertex_map(intrinsic)
# normal_map = vertex_map.create_normal_map()
# print((normal_map.as_tensor().cpu().numpy())[240,422]) #480, 848
# #o3d.visualization.draw([pcd,pcd2])
# # reg_res = o3d.t.pipelines.registration.icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=1000))

# # metric_params = o3d.t.geometry.MetricParameters()
# # metrics = pcd.compute_metrics(
# #     pcd2, [o3d.t.geometry.Metric.ChamferDistance],
# #     metric_params)

# # print(metrics.cpu().numpy())
# # np.testing.assert_allclose(
# #     metrics.cpu().numpy(),
# #     (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
# #     rtol=1e-6)
# t2 = time.time()

# print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

# p = o3d.t.geometry.PointCloud()
# p.point.positions = vertex_map.as_tensor().cpu().reshape((-1, 3))
# p.point.normals = normal_map.as_tensor().cpu().reshape((-1, 3))
# # o3d.visualization.draw([p])

# legacy_pcd = p.to_legacy()
# o3d.visualization.draw_geometries([legacy_pcd], point_show_normal=True)

# t0 = time.time()
# depth_o3d = o3d.t.geometry.Image(depth_image)
# color_o3d = o3d.t.geometry.Image(color_image)
# RGBD = o3d.t.geometry.RGBDImage(color_o3d,depth_o3d,10000)
# pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic)
# pcd_cuda = pcd.cuda()
# pcd.uniform_down_sample(every_k_points=10)
# #pcd.estimate_normals()
# pcd2 = pcd.translate([0.01,0.1,0.01])
# pcd2.estimate_normals()
# t1 = time.time()


# # reg_res = o3d.t.pipelines.registration.icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=1000))

# # pcd2 = pcd.translate([0.1,0.1,0.1])
# # metric_params = o3d.t.geometry.MetricParameters(fscore_radius=o3d.utility.DoubleVector((0.01, 0.11, 0.15, 0.18)))
# # metrics = pcd.compute_metrics(
# #     pcd2, [o3d.t.geometry.Metric.FScore],
# #     metric_params)

# # print(metrics.cpu().numpy())
# # np.testing.assert_allclose(
# #     metrics.cpu().numpy(),
# #     (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
# #     rtol=1e-6)
# t2 = time.time()

# print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

# t0 = time.time()
# depth_o3d = o3d.geometry.Image(depth_image)
# color_o3d = o3d.geometry.Image(color_image)
# RGBD = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d,depth_o3d,10000)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(RGBD,intrinsic_nt)
# #pcd.estimate_normals()
# pcd.uniform_down_sample(every_k_points=10)
# pcd2 = pcd.translate([0.1,0.1,0.1])
# pcd2.estimate_normals()
# t1 = time.time()

# # reg_res = o3d.pipelines.registration.registration_icp(source = pcd,
# #                                            target = pcd2,
# #                                            max_correspondence_distance = 0.02,
# #                                            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
# #                                            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1.000000e-08, max_iteration=1000))
# # print(reg_res)
# t2 = time.time()

# #print(depth_o3d.is_cuda, color_o3d.is_cuda,RGBD.is_cuda)
# print("Time 1", t1-t0)
# print("Time 2", t2-t1)

