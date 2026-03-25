import cv2
import robotpy_apriltag
import numpy as np
import open3d as o3d
import copy
import time

t0= time.time()
print(t0)
# Load the image and convert to grayscale
image = cv2.imread("000115.jpg")
#image = cv2.imread("./color/000001.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize AprilTag detector and configure the tag family
detector = robotpy_apriltag.AprilTagDetector()
detector.addFamily("tag36h11", 3)

# Detect AprilTags
detections = detector.detect(gray)
tags = []  # List to store detected tag IDs
outlineColor = (0, 255, 0)  # Color of tag outline
crossColor = (0, 0, 255)  # Color of cross

intrinsic = o3d.io.read_pinhole_camera_intrinsic( "camera_intrinsic.json")
# Configure the pose estimator
poseEstConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
        0.115,  # Tag size in meters
        intrinsic.intrinsic_matrix[0, 0],
        intrinsic.intrinsic_matrix[1, 1],
        intrinsic.intrinsic_matrix[0, 2],
        intrinsic.intrinsic_matrix[1, 2]
    )
estimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstConfig)

source_color = o3d.io.read_image(f"000115.jpg")
source_depth = o3d.io.read_image(f"000115.png")

source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth, depth_trunc = 6)
source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
#o3d.io.write_point_cloud("./T2CAM_1.pcd",source_pcd)
source_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
source_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

print(np.asarray(source_rgbd_image.color)[0,0])
pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)

t1 = time.time()
print(t0 - t1)

#o3d.visualization.draw(source_pcd)

origin = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(origin.to_legacy())
vis.add_geometry(source_pcd)
vis.update_renderer()
vis.poll_events()
transform_array = []

mesh_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)

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

    # fx = 958.658
    # fy = 958.658
    # cx = 943.963
    # cy = 534.253
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    print([fx,fy,cx,cy])

    x = (cent_x - cx) * depth_val / fx
    y = (cent_y - cy) * depth_val / fy
    z = depth_val

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
    
    [k, idx, _] = pcd_tree.search_knn_vector_3d([x,y,z], 1)
    neighbor_normals = np.asarray(source_pcd.normals)[idx]
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





