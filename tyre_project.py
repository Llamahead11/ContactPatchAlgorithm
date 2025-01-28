import open3d as o3d
import numpy as np
import open3d.visualization


pcd_outer = o3d.io.read_point_cloud("C:\\Users\\amalp\\Desktop\\MSS732\\tract_outide\\clean_output.ply")
translation_vector = np.array([0,0,-6])
pcd_outer.points = o3d.utility.Vector3dVector(np.asarray(pcd_outer.points) + translation_vector)

mesh_outer = o3d.io.read_triangle_mesh("C:\\Users\\amalp\\Desktop\\MSS732\\tract_outide\\clean_output.ply")
mesh_outer.translate(translation_vector)
mesh_outer.compute_vertex_normals()

#pcd_inner = o3d.io.read_point_cloud("C:\\Users\\amalp\\Desktop\\MSS732\\projected\\clean_output.ply")
pcd_inner = o3d.io.read_point_cloud("C:\\Users\\amalp\\Desktop\\MSS732\\betterinside\\scene\\cleaned_integrated.ply")
print(np.mean(pcd_inner.points,axis=0))
p = np.asarray(pcd_inner.points[::10])
n = np.asarray(pcd_inner.normals[::10])

#mesh_inner = o3d.io.read_triangle_mesh("C:\\Users\\amalp\\Desktop\\MSS732\\projected\\clean_output.ply")
mesh_inner = o3d.io.read_triangle_mesh("C:\\Users\\amalp\\Desktop\\MSS732\\betterinside\\scene\\cleaned_integrated.ply")
mesh_inner.compute_vertex_normals()

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])


scene = o3d.t.geometry.RaycastingScene()
inner_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_inner))
outer_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_outer))
print(inner_id,outer_id)

# Cast rays with offset
offset_distance = 0.1 # small offset value
ray_origins_offset = p + offset_distance * (-n)
ray_data = np.hstack((ray_origins_offset, -n))  

ray_tensor = o3d.core.Tensor(ray_data, dtype=o3d.core.Dtype.Float32) 
results = scene.cast_rays(ray_tensor)

# draw normals
normal_length = 0.1
line_starts = p
line_ends = p + (-n) * normal_length

lines = [[i, i + len(p)] for i in range(len(p))]
line_points = np.vstack((line_starts, line_ends))

# visualization inverted normals
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

o3d.visualization.draw_geometries([mesh_outer,mesh_inner,mesh_sphere,mesh_frame,line_set],zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


hit = (results['t_hit'].isfinite()) & (results['geometry_ids']!=0)
print(results)

print('Geometry hit',results['geometry_ids'].numpy())
print(results['t_hit'].numpy())
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

o3d.visualization.draw_geometries([pcd1.to_legacy(),pcd.to_legacy(),ray_lines,mesh_frame],
                                  front=[0.5, 0.86, 0.125],
                                  lookat=[0.23, 0.5, 2],
                                  up=[-0.63, 0.45, -0.63],
                                  zoom=0.7)

#Create mesh


# Select points:
# Visualize cloud and edit (Shift + Left Click)
vis = open3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd.to_legacy())
vis.add_geometry(pcd1.to_legacy())
vis.run()
vis.destroy_window()
print(vis.get_picked_points()) #[84, 119, 69]
print(vis.get_picked_points()[0])

#Select points in radius
pcd_outer_tree = o3d.geometry.KDTreeFlann(pcd.to_legacy())
print("Find its neighbors with distance less than 0.2, and paint them green.")
[k, idx, _] = pcd_outer_tree.search_radius_vector_3d(pcd.to_legacy().points[vis.get_picked_points()[0]], 1)
print(k,len(idx))
device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
pcd.paint_uniform_color([1.0,0.0,0.0])
pcd.estimate_normals()
pcd1.paint_uniform_color([1.0,0.0,0.0])
pcd1.estimate_normals()
for index in idx:
        pcd.point.colors[index] = o3d.core.Tensor([0.0, 1.0, 0.0], o3d.core.Dtype.Float32)  # Set to green
        pcd1.point.colors[index] = o3d.core.Tensor([0.0, 1.0, 0.0], o3d.core.Dtype.Float32)  # Set to green

centre_p = pcd.point.positions[vis.get_picked_points()[0]]
centre_n = pcd.point.normals[vis.get_picked_points()[0]]
print("centre_p,centre_n",centre_p,centre_n)

# Create a line showing the normal vector
n_start = centre_p
n_end = centre_p + centre_n * 1  # Scale the normal for visibility

# Define line set for visualization
dlines = [[0, 1]]
dpoints = [n_start.numpy(), n_end.numpy()]
print("dcsdcsdcsdc",dpoints)
dcolors = [[1, 0, 0]]  # Color the normal line red

line_set_n = o3d.t.geometry.LineSet()
line_set_n.point.positions = o3d.core.Tensor(dpoints, o3d.core.float32)
line_set_n.line.indices = o3d.core.Tensor(dlines, o3d.core.int32)
line_set_n.line.colors = o3d.core.Tensor(dcolors, o3d.core.float32)

#print(ray_tensor[hit][vis.get_picked_points()[0]][3:])
#print(results['t_hit'][hit][vis.get_picked_points()[0]].reshape((-1,1)))
#print((ray_tensor[hit][vis.get_picked_points()[0]][3:]*results['t_hit'][hit][vis.get_picked_points()[0]].reshape((-1,1)))[0])
#pcd.translate(-pcd.point.positions[vis.get_picked_points()[0]])
#pcd1.translate(-pcd1.point.positions[vis.get_picked_points()[0]]-(ray_tensor[hit][vis.get_picked_points()[0]][3:]*results['t_hit'][hit][vis.get_picked_points()[0]].reshape((-1,1)))[0])

#up_direction = np.array([0, 0, 1])

#pcd.rotate(pcd.point.normals[vis.get_picked_points()[0]])
# o3d.visualization.draw_geometries([line_set.to_legacy()],
#                                   zoom=0.5599,
#                                   front=[-0.4958, 0.8229, 0.2773],
#                                   lookat=[2.1126, 1.0163, -1.8543],
#                                   up=[0.1007, -0.2626, 0.9596])

o3d.visualization.draw_geometries([pcd.to_legacy(),pcd1.to_legacy(),mesh_frame,line_set_n.to_legacy()],
                                  zoom=0.5599,
                                  front=[-0.4958, 0.8229, 0.2773],
                                  lookat=[2.1126, 1.0163, -1.8543],
                                  up=[0.1007, -0.2626, 0.9596])

#Artificially Deform Outer and Compute Inner:
#use average normals of 5 closest points (to decrease noise) (normals point inwards)
#take green points and project them upwards using a function 
# function takes the outer points and keeps them there, rest of points are moved upwards where the inner most points are moved upwards

pcd1.point.positions[idx] = pcd1.point.positions[idx] + centre_n * (200)


o3d.visualization.draw_geometries([pcd.to_legacy(),mesh_frame],
                                  zoom=0.5599,
                                  front=[-0.4958, 0.8229, 0.2773],
                                  lookat=[2.1126, 1.0163, -1.8543],
                                  up=[0.1007, -0.2626, 0.9596])


#Convert to mesh using poisson surface reconstruction (points in pointcloud are equivalent to vertices in mesh)
print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh_ray_outer, densities_ray_outer = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd.to_legacy(), depth=12)
    mesh_ray_inner, densities_ray_inner = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd1.to_legacy(), depth=12)
print(mesh_ray_outer,mesh_ray_inner)

o3d.visualization.draw_geometries([mesh_ray_outer,mesh_ray_inner],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])




#Save projections
inner_p = ray_tensor[hit][:, :3].numpy()
outer_p = poi.numpy()
direction = ray_tensor[hit][:,3:].numpy()
distance = results['t_hit'][hit].numpy()


def_outer_p = 0
def_inner_p = def_outer_p + (-direction) * distance











