import cv2
import open3d as o3d
import numpy as np
import robotpy_apriltag
import csv
import pyrealsense2 as rs

# Open marker CSV file
sc_v = 0.01934137612
markers = []
numeric_markers = []
m_points = []
with open("C://Users//amalp//Desktop//MSS732//better//test3best_capture_with_edges.csv", 'r') as file:
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

# Find and print the correct convention Apriltag IDs in my 3D inner tyre model 
correctTags = []; 
for n_m in numeric_markers:
    c_t = normalTag[int(n_m)-1]
    #print(n_m,c_t)
    correctTags.append(c_t)
print(correctTags)

# print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("C://Users//amalp//Desktop//MSS732//better//best_capture_with_edges.ply")
pcd.scale(scale = 0.01934137612, center = [0,0,0])
#Estimate normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

# Find points near marker
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
image = cv2.imread("000115.jpg")
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

intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json")
#intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
# Configure the pose estimator
poseEstConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
        0.0115,  # Tag size in meters
        intrinsic.intrinsic_matrix[0, 0],
        intrinsic.intrinsic_matrix[1, 1],
        intrinsic.intrinsic_matrix[0, 2],
        intrinsic.intrinsic_matrix[1, 2]
    )
estimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstConfig)

source_color = o3d.io.read_image(f"000115.jpg")
source_depth = o3d.io.read_image(f"000115.png")
depth_scale = 0.1
depth_image_scaled = np.asarray(source_depth) * depth_scale
depth_image_scaled = o3d.geometry.Image(depth_image_scaled.astype(np.float32))
source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, depth_image_scaled,depth_trunc = 5.0)
source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
source_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
source_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

print(np.asarray(source_rgbd_image.color)[0,0])
pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)

origin = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(origin.to_legacy())
vis.add_geometry(source_pcd)
vis.update_renderer()
vis.poll_events()
transform_array = []

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

    # Estimate and display the pose of the tag
    pose = estimator.estimate(detection)
    rot = pose.rotation()
    
    print(rot)
    print(f"Pose for Tag {detection.getId()}: X={pose.X()}, Y={pose.Y()}, Z={pose.Z()}, RotX={rot.X()}, RotY={rot.Y()}, RotZ={rot.Z()}")
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
    #ro = np.asarray(source_pcd.normals)[idx].flatten()
    ro = np.array([rot.X(),rot.Y(),rot.Z()]).flatten()
    print(ro)
    #print(np.asarray(source_pcd.normals)[idx])
    #np.array([rot.X(),rot.Y(),rot.Z()])
    # z_axis = np.array([0, 0, 1])
    # rotation_axis = np.cross(z_axis, ro)
    # rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    # angle = np.arccos(np.dot(z_axis, ro) / (np.linalg.norm(z_axis) * np.linalg.norm(ro)))
    # rotation_matrix, _ = cv2.Rodrigues(rotation_axis*angle)
    R_o3d = o3d.core.Tensor(rotation_matrix, dtype=o3d.core.float32)
    center = o3d.core.Tensor([0,0,0], dtype=o3d.core.float32)
    mesh_frame.rotate(R_o3d,center)
    
    mesh_frame.translate([x,y,z])
    #mesh_frame.translate([pose.X(), pose.Y(), pose.Z()])
    mesh_frame.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    vis.add_geometry(mesh_frame.to_legacy())
    vis.poll_events()
    vis.update_renderer()
    #R = mesh_frame.get_rotation_matrix_from_xyz((rot.X(),rot.Y(),rot.Z()))
    #R = mesh_frame.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    #print(R)
    #o3d.core.Tensor([rot.X(),rot.Y(),rot.Z()],o3d.core.float64)
    #o3d.core.Tensor(R, o3d.core.Dtype.Float32)
    #o3d.core.Tensor([[-0.19649191 , 0.74538074, -0.63702315],[-0.80862165 ,-0.49063371 ,-0.32466843],[-0.55454662 , 0.45131599 , 0.69913656]]),o3d.core.float32
    #mesh_frame.rotate(R = o3d.core.Tensor([0.43,0.69,-1.8],o3d.core.float32),centre = o3d.core.Tensor([0,0,0],o3d.core.float32))
    #mesh_frame.rotate(R)

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
# Registration

# define 3d affine transformation from T2Cam coordinate system to 3D model coordinate system
print("t2cam: ", t2cam_correspondence)
print("model: ", model_correspondence)
str_tags = [str(e) for e in tags]
p = t2cam_correspondence[0:4] 
q = [model_correspondence[correctTags.index(str_tags[0])],model_correspondence[correctTags.index(str_tags[1])],model_correspondence[correctTags.index(str_tags[2])],model_correspondence[correctTags.index(str_tags[3])]]
pq = np.asarray([[p[i], q[i]] for i in range(4)]) 
print(pq)
corres = o3d.utility.Vector2iVector(pq)
estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
T = estimator.compute_transformation(source_pcd,pcd,corres)
# p1,p2,p3,p4 = tag_loc[0:4]
# print(tags[0:3])
# str_tags = [str(e) for e in tags]
# q1 = m_points[correctTags.index(str_tags[0])]
# q2 = m_points[correctTags.index(str_tags[1])]
# q3 = m_points[correctTags.index(str_tags[2])]
# q4 = m_points[correctTags.index(str_tags[3])]
# #print(p1,p2,p3,p4)
# #print(q1,q2,q3,q4)
# T = cv2.estimateAffine3D(np.asarray([p1, p2, p3, p4]), np.asarray([q1, q2, q3, q4]), force_rotation = True)
# print(T)
# T_mat = np.vstack([T[0].reshape(3,4), [0, 0, 0, 1]])
# print(T_mat)

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



