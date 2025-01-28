import open3d as o3d
import numpy as np
import cv2
import copy
import robotpy_apriltag

def april():
    # Load the image and convert to grayscale
    image = cv2.imread("000115.jpg")
    #image = cv2.imread("./color/000001.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize AprilTag detector and configure the tag family
    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily("tag36h11", 1)

    # Detect AprilTags
    detections = detector.detect(gray)
    tags = []  # List to store detected tag IDs
    outlineColor = (0, 255, 0)  # Color of tag outline
    crossColor = (0, 0, 255)  # Color of cross

    # Configure the pose estimator
    poseEstConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
        0.0115,  # Tag size in meters
        958.658,  # Camera focal length (fx)
        958.658,  # Camera focal length (fy)
        943.963,  # Principal point (cx)
        534.253  # Principal point (cy)
    )
    estimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstConfig)

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
        cx = int(detection.getCenter().x)
        cy = int(detection.getCenter().y)
        ll = 10
        gray = cv2.line(gray, (cx - ll, cy), (cx + ll, cy), crossColor, 2)
        gray = cv2.line(gray, (cx, cy - ll), (cx, cy + ll), crossColor, 2)

        # Display the tag ID near its center
        gray = cv2.putText(
            gray,
            str(detection.getId()),
            (cx + ll, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            crossColor,
            3
        )

        # Estimate and display the pose of the tag
        pose = estimator.estimate(detection)
        rot = pose.rotation()
        print(f"Pose for Tag {detection.getId()}: X={pose.X()}, Y={pose.Y()}, Z={pose.Z()}, RotX={rot.X()}, RotY={rot.Y()}, RotZ={rot.Z()}")

    # Display the tags detected
    print("Detected tags:", tags)

    # Show the grayscale image with overlays
    cv2.imshow("gray", gray)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the image window when done
    return 1

if __name__ == "__main__":
    N = 10
    axis = []
    origin = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis = o3d.visualization.Visualizer()

    
    vis.create_window()
    
    vis.add_geometry(origin.to_legacy())
    vis.update_renderer()
    vis.poll_events()
    transform_array = []

    mesh_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
    
    for i in range(1,N):
        N1 = i+1
        intrinsic = o3d.io.read_pinhole_camera_intrinsic( "camera_intrinsic.json")
        source_color = o3d.io.read_image(f"color/{str(i).zfill(6)}.jpg")
        source_depth = o3d.io.read_image(f"depth/{str(i).zfill(6)}.png")
        target_color = o3d.io.read_image(f"color/{str(i+1).zfill(6)}.jpg")
        target_depth = o3d.io.read_image(f"depth/{str(i+1).zfill(6)}.png")
        print(f"color/{str(i).zfill(6)}.jpg")
        

        # source_color = o3d.io.read_image("color/001683.jpg")
        # source_depth = o3d.io.read_image("depth/001683.png")
        # target_color = o3d.io.read_image("000013.jpg")
        # target_depth = o3d.io.read_image("000013.png")

        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth, depth_trunc = 6)
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_color, target_depth, depth_trunc = 6)
        source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd_image, intrinsic)

        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = 0.3
        option.depth_max = 4
        option.depth_min = 0
        odo_init = np.identity(4)
        #print(option)

        [success_color_term, trans_color_term,
        info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image,
            intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
        # [success_hybrid_term, trans_hybrid_term,
        #  info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        #      source_rgbd_image, target_rgbd_image,
        #      intrinsic, odo_init,
        #      o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

        if success_color_term:
            print("Using RGB-D Odometry")
            print(trans_color_term)
            source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
                source_rgbd_image, intrinsic)
            source_pcd_color_term.transform(trans_color_term)
            if i == 1:
                vis.add_geometry(source_pcd_color_term)
            #o3d.visualization.draw([target_pcd, source_pcd_color_term])

        # if success_hybrid_term:
        #     print("Using Hybrid RGB-D Odometry")
        #     print(trans_hybrid_term)
        #     source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
        #         source_rgbd_image, intrinsic)
        #     source_pcd_hybrid_term.transform(trans_hybrid_term)
        #     o3d.visualization.draw([target_pcd, source_pcd_hybrid_term])

        transform_array.append(source_pcd_color_term)
        
        mesh_frame.transform(trans_color_term)
        #axis.append(mesh_frame)
        vis.add_geometry(mesh_frame.to_legacy())
        vis.update_geometry(source_pcd_color_term)
        vis.poll_events()
        vis.update_renderer()
        
        #o3d.visualization.draw([origin, mesh_frame, source_pcd.paint_uniform_color([1,0,0]), target_pcd.paint_uniform_color([0,1,0])])
    
    #o3d.visualization.draw(axis)
    vis.run()
    vis.destroy_window()

