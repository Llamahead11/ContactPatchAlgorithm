import cv2
import open3d as o3d
import numpy as np
import robotpy_apriltag
import csv
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Open marker CSV file
sc_v = 1#0.0378047581618546#0.019390745853434508 #0.03468647377170447 #0.01934137612
markers = []
numeric_markers = []
m_points = []
#"../4_row_model/4_row_model_control_points.csv"
with open("./ContactPatchAlgorithm/full_outer.csv", 'r') as file:
    csv_reader = csv.reader(file)
    
    # Iterate through rows
    for row in csv_reader:
        print(row)
        markers.append(row[0])
        m_points.append([sc_v*float(row[1]),sc_v*float(row[2]),sc_v*float(row[3])])  
        numeric_markers.append(row[5])

# print(markers)
# print(m_points)

# Open csv that has reality capture tag correspondence to normal april tag 36h11 convention
normalTag = []
RCcorTag = []
with open("C://Users//amalp//Desktop//MSS732//projected//RCtoTag.csv", 'r') as file:
    csv_reader = csv.reader(file)
    
    # Iterate through rows
    for row in csv_reader:
        print(row)
        RCcorTag.append(row[0])
        normalTag.append(row[2])

correctTags = []; 
for n_m in numeric_markers:
    c_t = normalTag[int(n_m)-1]
    correctTags.append(c_t)


# print("Load a ply point cloud, print it, and render it")
#"../4_row_model/4_row_model_HighPoly_Smoothed.ply"
pcd = o3d.io.read_point_cloud("./ContactPatchAlgorithm/full_outer_inner_part_only.ply", print_progress = True)
#pcd.scale(scale = 0.0378047581618546, center = [0,0,0])
#pcd.scale(scale = 0.019390745853434508, center = [0,0,0])
#pcd.scale(scale = 0.03468647377170447, center = [0,0,0])
#Estimate normals
print("Estimating model normals")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
print("Completed")

# Find points near marker
print("Find points in model tags")
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
tag_norm = []
model_correspondence = []
for tag in range(0,len(m_points)):
    [k, idx, _] = pcd_tree.search_knn_vector_3d(m_points[tag],500)
    model_correspondence.append(idx[0])
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    m_norm = np.asarray(pcd.normals)[idx[1:], :]
    m_norm_ave = np.mean(m_norm, axis=0)
    tag_norm.append(m_norm_ave)
print("Completed")

tag_norm = np.array(tag_norm)    
print(tag_norm)

# draw normals
normal_length = 0.1
line_starts = m_points
line_ends = m_points + tag_norm * normal_length

lines = [[i, i + len(m_points)] for i in range(len(m_points))]
line_points = np.vstack((line_starts, line_ends))

# visualization inverted normals
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

# Visualisation Window (faster) (no tag IDs)
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0]))
# vis.add_geometry(line_set)
# # for i in range(len(m_points)):
# #     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=m_points[i])
# #     vis.add_geometry(mesh_frame)
# #     vis.update_renderer()
# #     vis.poll_events()

# vis.run()
# vis.destroy_window() 

#=====================================================================================================
# # Capture a single point cloud frame from the RealSense camera
# def capture_point_cloud(pipeline):
#     frames = pipeline.wait_for_frames()
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()

#     if not depth_frame or not color_frame:
#         raise RuntimeError("Could not capture frames")

#     depth_image = np.asanyarray(depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())

#     # Convert to Open3D format
#     depth_o3d = o3d.geometry.Image(depth_image)
#     color_o3d = o3d.geometry.Image(color_image)

#     # Create RGBD image
#     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image),
#         convert_rgb_to_intensity=False)

#     intrinsic = o3d.camera.PinholeCameraIntrinsic(
#         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

#     # Generate point cloud
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

#     # Flip the point cloud to correct for the RealSense coordinate system
#     pcd.transform([[1, 0, 0, 0],
#                    [0, -1, 0, 0],
#                    [0, 0, -1, 0],
#                    [0, 0, 0, 1]])

#     # Estimate normals for the point cloud
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#     return pcd, color_image

# # Configure and start the RealSense pipeline
# pipeline = rs.pipeline()
# config = rs.config()

# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# profile = pipeline.start(config)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_sensor.set_option(rs.option.visual_preset, 3)
# T2Cam, T2Cam_colour = capture_point_cloud(pipeline)

#===========================================================================================================
#image = cv2.imread(r'C:\Users\amalp\Desktop\MSS732\realsense\color_0_0\000310.jpg')
image = cv2.imread(r'C:\Users\amalp\Desktop\MSS732\realsense\color_3_18\000215.jpg')
#image = cv2.imread(r'C:\Users\amalp\Desktop\MSS732\projected\middle_in\color\000646.jpg')
#image = cv2.imread("./color/000001.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize AprilTag detector and configure the tag family
detector = robotpy_apriltag.AprilTagDetector()
detector.addFamily("tag36h11", 3)

# Detect AprilTags
detections = detector.detect(gray)
tags = []  # List to store detected tag IDs
tag_loc = [] # List to store detected tag locations [x,y,z]
outlineColor = (0, 255, 0)  # Color of tag outline
crossColor = (0, 0, 255)  # Color of cross

#intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json")
intrinsic = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
#intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

#source_color = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\realsense\color_0_0\000310.jpg')
#source_color = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\realsense\color_3_18\000215.jpg')
source_color = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\realsense2\color\000110.jpg')
#source_color = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\projected\middle_in\color\000646.jpg')
#source_depth = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\realsense\depth_0_0\000310.png')
#source_depth = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\realsense\depth_3_18\000215.png')
source_depth = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\realsense2\depth\000110.png')
#source_depth = o3d.io.read_image(r'C:\Users\amalp\Desktop\MSS732\projected\middle_in\depth\000646.png')
depth_scale = 0.1
depth_image_scaled = np.asarray(source_depth) * depth_scale
depth_image_scaled = o3d.geometry.Image(depth_image_scaled.astype(np.float32))
source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, depth_image_scaled,depth_trunc = 5.0)
source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
source_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

print("Estimating normals")
source_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
print("Completed")

print(np.asarray(source_rgbd_image.color)[0,0])
print("Making kDTree")
pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)
print("Completed")

origin = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(origin.to_legacy())
vis.add_geometry(source_pcd)
vis.update_renderer()
vis.poll_events()


mesh_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
t2cam_correspondence = []
for detection in detections:
    # Store detected tag ID
    tags.append(detection.getId())

    # Draw lines around the tag
    for i in range(4):
        j = (i + 1) % 4
        point1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
        point2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
        gray = cv2.line(gray, point1, point2, outlineColor, 2)

    # Mark the center of the tag with a cross
    cent_x = int(detection.getCenter().x)
    cent_y = int(detection.getCenter().y)
    
    depth_val = np.asarray(source_rgbd_image.depth)[cent_y,cent_x]
    print("Depth",depth_val)

    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    print([fx,fy,cx,cy])

    x = (cent_x - cx) * depth_val / fx
    y = (cent_y - cy) * depth_val / fy 
    z = depth_val

    tag_loc.append([x,-y,-z])
    print("Point Coords",[x,y,z])

    ll = 10
    gray = cv2.line(gray, (cent_x - ll, cent_y), (cent_x + ll, cent_y), crossColor, 2)
    gray = cv2.line(gray, (cent_x, cent_y - ll), (cent_x, cent_y + ll), crossColor, 2)

    # Display the tag ID near its center
    gray = cv2.putText(
        gray,
        str(detection.getId()),
        (cent_x + ll, cent_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        crossColor,
        3
    )
  
    mesh_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.1) 
    
    [k, idx, _] = pcd_tree.search_knn_vector_3d([x,-y,-z], 1)
    neighbor_normals = np.asarray(source_pcd.normals)[idx]
    t2cam_correspondence.append(idx[0])

    average_normal = np.mean(neighbor_normals, axis=0) # Normalize
    average_normal /= np.linalg.norm(average_normal)

    # Now align the z-axis of mesh_frame with this averaged normal
    # Define an arbitrary vector that is not parallel to the average normal
    reference_vector = np.array([0, 1, 0]) if abs(average_normal[1]) < 0.9 else np.array([1, 0, 0])

    # Compute the x-axis by taking the cross product of the average normal and the reference vector
    x_axis = np.cross(average_normal, reference_vector)
    x_axis /= np.linalg.norm(x_axis)  # Normalize

    # Compute the y-axis to complete the coordinate frame
    y_axis = np.cross(average_normal, x_axis)

    # Create a rotation matrix with z-axis aligned to the average normal
    rotation_matrix = np.column_stack((x_axis, y_axis, average_normal))
    print(idx)
    
    R_o3d = o3d.core.Tensor(rotation_matrix, dtype=o3d.core.float32)
    center = o3d.core.Tensor([0,0,0], dtype=o3d.core.float32)
    mesh_frame.rotate(R_o3d,center)
    
    mesh_frame.translate([x,y,z])
    mesh_frame.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    vis.add_geometry(mesh_frame.to_legacy())
    vis.poll_events()
    vis.update_renderer()
    
# Display the tags detected
print("Detected tags:", tags)
vis.run()
vis.destroy_window()

# Show the grayscale image with overlays
cv2.imshow("gray", gray)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window when done

#===========================================================================================================
# Visualisation Window (slow) (Tag IDs shown) (Before Registration)
app = o3d.visualization.gui.Application.instance
app.initialize()
o3dvis = o3d.visualization.O3DVisualizer()
o3dvis.show_settings = True

for t_l,t in zip(tag_loc,tags):
    o3dvis.add_3d_label(pos=t_l, text = f'{t}')

o3dvis.add_geometry("T2Cam",source_pcd)
o3dvis.add_geometry("my points",pcd)
o3dvis.add_geometry("tag normals", line_set)

for mark,id in zip(m_points,correctTags):
    o3dvis.add_3d_label(pos=mark, text = f'{id}')
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:        
    '''visualize'''
    o3dvis.reset_camera_to_default()
    app.add_window(o3dvis)
    app.run()

#===========================================================================================================
# Registration and scaling
str_tags = [str(e) for e in tags]

# Scaling model (not efficient)
model_idx_corres = np.array([[idx,int(ctag)] for idx,ctag in enumerate(correctTags) if ctag in str_tags])
sorted_idx_corres = model_idx_corres[model_idx_corres[:, 1].argsort()][:,0]
print(sorted_idx_corres)

scale_arr = []
for tag_idx,idx in enumerate(sorted_idx_corres[:-1]):
    d_t2cam_id_curr_next = np.linalg.norm(np.array(tag_loc[tag_idx]) - np.array(tag_loc[tag_idx+1]))
    d_model_id_curr_next = np.linalg.norm(np.array(m_points[idx]) - np.array(m_points[sorted_idx_corres[tag_idx+1]]))
    scale = d_t2cam_id_curr_next/d_model_id_curr_next
    print(tag_idx,idx,d_t2cam_id_curr_next,d_model_id_curr_next)
    scale_arr.append(scale)

scale = np.mean(scale_arr)
print("SCALE ++++++++++++++++++++++++++++")
print(scale)
print(scale_arr)
pcd.scale(scale = scale, center = [0,0,0])
m_points = scale*np.array(m_points)

# define 3d transformation from T2Cam coordinate system to 3D model coordinate system
print("t2cam: ", t2cam_correspondence)
print("model: ", model_correspondence)

p = t2cam_correspondence
q = [model_correspondence[correctTags.index(tag)] for tag in str_tags]
pq = np.asarray([[p[i], q[i]] for i in range(len(p))]) 
print(pq) 
corres = o3d.utility.Vector2iVector(pq)
estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
T = estimator.compute_transformation(source_pcd,pcd,corres)

# Transform T2Cam point cloud
source_pcd.transform(T) 

#===========================================================================================================
# Visualisation Window (slow) (Tag IDs shown) (After Registration)
app = o3d.visualization.gui.Application.instance
app.initialize()
o3dvis = o3d.visualization.O3DVisualizer()
o3dvis.show_settings = True

for t_l,t in zip(tag_loc,tags):
    o3dvis.add_3d_label(pos=t_l, text = f'{t}')

o3dvis.add_geometry("T2Cam",source_pcd)
o3dvis.add_geometry("my points",pcd)
o3dvis.add_geometry("tag normals", line_set)

for mark,id in zip(m_points,correctTags):
    o3dvis.add_3d_label(pos=mark, text = f'{id}')
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:        
    '''visualize'''
    o3dvis.reset_camera_to_default()
    app.add_window(o3dvis)
    app.run()

# print(correctTags)
# print(m_points)
# print(tags)
# print(tag_loc)

# for idx,tag in enumerate(tags):
#     print(tag,idx,tag_loc[idx])
    
# for idx,ctag in enumerate(correctTags):
#     if ctag in str_tags:
#         print(ctag,idx,m_points[idx])

# FIX THIS INDICES NOT CORRECT< CORRESPOND TO OLD MODEL

# d_t2cam_34_35 = np.linalg.norm(np.array(tag_loc[0]) - np.array(tag_loc[1]))
# print("Distance between tag 34 and 35 on t2cam:", d_t2cam_34_35)

# d_t2cam_35_36 = np.linalg.norm(np.array(tag_loc[1]) - np.array(tag_loc[2]))
# print("Distance between tag 34 and 35 on t2cam:", d_t2cam_35_36)

# d_model_34_35 = np.linalg.norm(np.array(m_points[89]) - np.array(m_points[187]))
# print("Distance between tag 34 and 35 on model:", d_model_34_35)

# d_model_35_36 = np.linalg.norm(np.array(m_points[187]) - np.array(m_points[184]))
# print("Distance between tag 35 and 36 on model:", d_model_35_36)

# s_34_36 = np.linalg.norm(np.array(tag_loc[0]) - np.array(tag_loc[2]))/np.linalg.norm(np.array(m_points[89]) - np.array(m_points[184]))
# s_84_85 = np.linalg.norm(np.array(tag_loc[3]) - np.array(tag_loc[4]))/np.linalg.norm(np.array(m_points[93]) - np.array(m_points[114]))
# s_85_86 = np.linalg.norm(np.array(tag_loc[4]) - np.array(tag_loc[5]))/np.linalg.norm(np.array(m_points[114]) - np.array(m_points[2]))
# s_84_86 = np.linalg.norm(np.array(tag_loc[3]) - np.array(tag_loc[5]))/np.linalg.norm(np.array(m_points[93]) - np.array(m_points[2]))

# print("scale for 34 to 35 is: ", d_t2cam_34_35/d_model_34_35)
# print("scale for 35 to 36 is: ", d_t2cam_35_36/d_model_35_36)
# print("scale for 34 to 36 is: ", s_34_36)

# print("scale for 84 to 85 is: ", s_84_85)
# print("scale for 85 to 86 is: ", s_85_86)
# print("scale for 84 to 86 is: ", s_84_86)

# print(np.mean(np.array([d_t2cam_35_36/d_model_35_36,d_t2cam_34_35/d_model_34_35,s_34_36,s_84_85,s_84_86,s_85_86])))

# print(scale_arr)

# print(np.mean(np.array(scale_arr)))

#=====================================================================================================================
# Inner deformed to inner undeformed
#=====================================================================================================================

# Segment the model point cloud to contain that section of t2Cam point cloud
# calculate bounding box of t2cam_pcd and crop model according to it
t2cam_bounding_box = source_pcd.get_oriented_bounding_box() #.get_axis_aligned_bounding_box() 
cropped_model = pcd.crop(t2cam_bounding_box)

app = o3d.visualization.gui.Application.instance
app.initialize()
o3dvis = o3d.visualization.O3DVisualizer()
o3dvis.show_settings = True

for t_l,t in zip(tag_loc,tags):
    o3dvis.add_3d_label(pos=t_l, text = f'{t}')

o3dvis.add_geometry("T2Cam",source_pcd)
o3dvis.add_geometry("my points",cropped_model)
o3dvis.add_geometry("tag normals", line_set)

for mark,id in zip(m_points,correctTags):
    o3dvis.add_3d_label(pos=mark, text = f'{id}')
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:        
    '''visualize'''
    o3dvis.reset_camera_to_default()
    app.add_window(o3dvis)
    app.run()

# Calculate the distances between the source point cloud and target point cloud

deformed_distances = source_pcd.compute_point_cloud_distance(cropped_model)
deformed_distances = np.array(deformed_distances)
print(deformed_distances)
num_bins = 20 
plt.hist(deformed_distances, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')
plt.title("Histogram of Deformed Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

print(np.max(deformed_distances), np.min(deformed_distances))

d_dist = np.asarray(deformed_distances)
d_colors = plt.get_cmap('plasma')((d_dist - d_dist.min()) / (d_dist.max() - d_dist.min()))
d_colors = d_colors[:, :3]

print(d_colors)
print(np.size(d_colors))
print(np.size(np.asarray(source_pcd.points)))
t2_d_pcd = o3d.geometry.PointCloud()
t2_d_pcd.points = source_pcd.points
t2_d_pcd.colors = o3d.utility.Vector3dVector(d_colors)

o3d.visualization.draw_geometries([t2_d_pcd])

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(t2_d_pcd)

# for idx,region in enumerate(grid[:-1]):
#     ind = np.where((deformed_distances > region) & (deformed_distances < grid[idx+1]))[0]
#     select_deformation = cropped_model.select_by_index(ind)
#     select_deformation.paint_uniform_color([color_def[idx], color_def[idx], 1-color_def[idx]])
#     vis.add_geometry(select_deformation)

# vis.run()
# vis.destroy_window()

#OR

# # Raycasting
# # Shoot rays from the t2cam to undeformed, make undeformed a mesh and find point correspondences
# # Might shoot rays in both directions
# # Check that rays dont hit objects with the same id or just take out the starting ray mesh from the scene

# mesh_model = o3d.io.read_triangle_mesh("../4_row_model/4_row_model_HighPoly_Smoothed.ply", print_progress = True)
# mesh_model.scale(scale = 0.019390745853434508, center = [0,0,0])
# cropped_mesh_model = mesh_model.crop(t2cam_bounding_box)
# print("Compute normals")
# cropped_mesh_model.compute_vertex_normals()
# print("Completed")

# # mesh_t2cam , densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
# #         source_pcd, depth=9)

# #get normals and points from t2cam pointcloud
# p = np.asarray(source_pcd.points[::1])
# n = np.asarray(source_pcd.normals[::1])

# scene = o3d.t.geometry.RaycastingScene()
# #deformed_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_t2cam))
# undeformed_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cropped_mesh_model))
# print(undeformed_id)

# # Cast rays with offset
# offset_distance = 0.00001 # small offset value hereeeeeeeeeeeeeee
# ray_origins_offset = p + offset_distance * (-n)
# ray_data = np.hstack((ray_origins_offset, -n)) 

# ray_tensor = o3d.core.Tensor(ray_data, dtype=o3d.core.Dtype.Float32) 
# results = scene.cast_rays(ray_tensor)

# # draw normals
# normal_length = 0.01
# line_starts = p
# line_ends = p + (-n) * normal_length

# lines = [[i, i + len(p)] for i in range(len(p))]
# line_points = np.vstack((line_starts, line_ends))

# # visualization inverted normals
# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(line_points)
# line_set.lines = o3d.utility.Vector2iVector(lines)
# line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

# o3d.visualization.draw_geometries([cropped_mesh_model,line_set])

# # Raycast info
# hit = (results['t_hit'].isfinite()) & (results['geometry_ids']==0)
# #print(results)

# #print('Geometry hit',results['geometry_ids'].numpy())
# #print(results['t_hit'].numpy())
# dist_ray = ray_tensor[hit][:,3:]*results['t_hit'][hit].reshape((-1,1))
# poi = ray_tensor[hit][:,:3] + dist_ray
# poi1 = ray_tensor[hit][:,:3]

# #print(dist_ray.numpy())
# # Color according to distance
# densities = np.linalg.norm(dist_ray.numpy(),axis=1) ## here
# density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
# density_colors = density_colors[:, :3]

# print(density_colors)

# density_pcd = o3d.geometry.PointCloud()
# density_pcd.points = o3d.utility.Vector3dVector(poi1.numpy())
# density_pcd.colors = o3d.utility.Vector3dVector(density_colors)

# pcd_1 = o3d.t.geometry.PointCloud(poi)
# pcd_2 = o3d.t.geometry.PointCloud(poi1)

# rays_hit_start = ray_tensor[hit][:, :3].numpy()
# rays_hit_end = poi.numpy()

# lines = [[i, i + len(rays_hit_start)] for i in range(len(rays_hit_start))]
# line_points = np.vstack((rays_hit_start, rays_hit_end))
# ray_lines = o3d.geometry.LineSet()
# ray_lines.points = o3d.utility.Vector3dVector(line_points)
# ray_lines.lines = o3d.utility.Vector2iVector(lines)

# o3d.visualization.draw_geometries([density_pcd,ray_lines])