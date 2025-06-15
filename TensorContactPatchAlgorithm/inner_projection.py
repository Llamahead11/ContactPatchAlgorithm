import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from rough_vis_deformation import load_rc_control_points, convert_rc_apriltag_hex_ids, convert_rc_control_points, find_tag_point_ID_correspondence

#model_inner_pcd = o3d.io.read_point_cloud("full_outer_inner_part_only.ply")
model_inner_ply = o3d.t.io.read_triangle_mesh("full_outer_inner_part_only.ply")
#model_outer_pcd = o3d.io.read_point_cloud("full_outer_outer_part_only.ply")
model_outer_ply = o3d.t.io.read_triangle_mesh("full_outer_outer_part_only.ply")
#model_outer_ply = o3d.t.io.read_triangle_mesh("full_outer_treads_part_only.ply")

model_detailed_inner = o3d.io.read_point_cloud("4_row_model_HighPoly_Smoothed.ply")
scale_new = 0.03757452 #0.03912#0.03805#0.0378047581618546 #0.0376047581618546 
scale_old = 0.019390745853434508

#model_inner_pcd.scale(scale = scale_new, center = [0,0,0])
#model_outer_pcd.scale(scale = scale_new, center = [0,0,0])
model_inner_ply.scale(scale = scale_new, center = [0,0,0])
model_outer_ply.scale(scale = scale_new, center = [0,0,0])
model_detailed_inner.scale(scale = scale_old, center = [0,0,0])

# o3d.visualization.draw_geometries([model_inner_pcd,model_detailed_inner])

# markers_old,m_points_old,numeric_markers_old = load_rc_control_points(file_path="./4_row_model_control_points.csv",scale_factor=scale_old)
# markers_new,m_points_new,numeric_markers_new = load_rc_control_points(file_path="./full_outer.csv",scale_factor=scale_new)
# normalTag, RCcorTag = convert_rc_apriltag_hex_ids(file_path="./RCtoTag.csv")
# correctTags_old = convert_rc_control_points(normalTag, numeric_markers_old)
# correctTags_new = convert_rc_control_points(normalTag, numeric_markers_new)

# tag_norm_old, model_correspondence_old = find_tag_point_ID_correspondence(model_detailed_inner,m_points_old)
# tag_norm_new, model_correspondence_new = find_tag_point_ID_correspondence(model_inner_pcd,m_points_new)

# #str_tags = [str(e) for e in correctTags_old]
# p = model_correspondence_old
# q = [model_correspondence_new[correctTags_new.index(tag)] for tag in correctTags_old]
# pq = np.asarray([[q[i], p[i]] for i in range(len(p))]) 
# print(pq) 
# corres = o3d.utility.Vector2iVector(pq)
# estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
# T = estimator.compute_transformation(model_inner_pcd,model_detailed_inner,corres)
# print(T)
# # Transform T2Cam point cloud
# model_inner_pcd.transform(T) 
# model_outer_pcd.transform(T)

# o3d.visualization.draw_geometries([model_inner_pcd,model_detailed_inner,model_outer_pcd])

model_outer_pcd = o3d.t.io.read_point_cloud("full_outer_treads_part_only.ply")
model_inner_pcd = o3d.t.io.read_point_cloud("full_outer_inner_part_only.ply")
model_detailed_inner = o3d.t.io.read_point_cloud("4_row_model_HighPoly_Smoothed.ply")
model_outer_pcd.scale(scale = scale_new, center = [0,0,0])
model_inner_pcd.scale(scale = scale_new, center = [0,0,0])
model_detailed_inner.scale(scale = scale_old, center = [0,0,0])
#model_outer_pcd.transform(o3d.core.Tensor(T))

device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
# prev = o3d.t.geometry.PointCloud(device)
# prev_pcd.point.positions = model.point.positions[0:2000000]
# prev_pcd.point.normals = model.point.normals[0:2000000]
print("orient")
model_inner_pcd.orient_normals_consistent_tangent_plane(k=10)
o3d.visualization.draw_geometries([model_inner_pcd.to_legacy()])

start_points = model_inner_pcd.point.positions.numpy()
normals = model_inner_pcd.point.normals.numpy()

t1 = time.time()
scene = o3d.t.geometry.RaycastingScene()
#inner_id = scene.add_triangles(model_ply)
outer_id = scene.add_triangles(model_outer_ply)
t2 = time.time()    
print("Add triangles",t2-t1)

# Cast rays with offset
offset_distance = 0.000000001 # small offset value
ray_origins_offset = start_points + offset_distance * (normals)
ray_data = np.hstack((ray_origins_offset, normals))  

ray_tensor = o3d.core.Tensor(ray_data, dtype=o3d.core.Dtype.Float32) 
print(ray_tensor.shape)

t3 = time.time()
results = scene.cast_rays(ray_tensor[:100000,:])
t4 = time.time()
print("cast rays", t4 - t3)

#hit = ((results['t_hit'].isfinite()) & (results['geometry_ids']==0)) & (results['t_hit'] < 0.1)
hit =  (results['geometry_ids']==0) & (results['t_hit'] < 0.1)
#print(results)

#print('Geometry hit',results['geometry_ids'].numpy())
#print(results['t_hit'].numpy())
poi = ray_tensor[hit][:,:3] + ray_tensor[hit][:,3:]*results['t_hit'][hit].reshape((-1,1))
poi1 = ray_tensor[hit][:,:3]

pcd = o3d.t.geometry.PointCloud(poi)
pcd1 = o3d.t.geometry.PointCloud(poi1)

rays_hit_start = ray_tensor[hit][:, :3].numpy()
rays_hit_end = poi.numpy()

lines = [[i, i + len(rays_hit_start)] for i in range(len(rays_hit_start))]
line_points = np.vstack((rays_hit_start, rays_hit_end))
ray_lines = o3d.geometry.LineSet()
ray_lines.points = o3d.utility.Vector3dVector(line_points)
ray_lines.lines = o3d.utility.Vector2iVector(lines)

o3d.visualization.draw_geometries([pcd.to_legacy(),pcd1.to_legacy()])

# with open('inner_to_treads_correspondences.npy', 'wb') as f:
#     np.save(f, rays_hit_start)
#     np.save(f, rays_hit_end)

# t0 = time.time()
# with open('inner_to_outer_correspondences.npy', 'rb') as f:
#     rays_hit_start = np.load(f)
#     rays_hit_end = np.load(f)
# t1 = time.time()    
# print(t1 - t0)
# print(rays_hit_start)