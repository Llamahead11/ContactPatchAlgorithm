import cv2
import robotpy_apriltag
import numpy as np
import open3d as o3d
import copy
import time

t0= time.time()
print(t0)
# Load the image and convert to grayscale
#image = cv2.imread("apriltags_page_1.png")
image = cv2.imread("t2test5_Color.png")
#image = cv2.imread("sheet6.JPG")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize AprilTag detector and configure the tag family
detector = robotpy_apriltag.AprilTagDetector()
detector.addFamily("tag36h11", 3)

# Detect AprilTags
detections = detector.detect(gray)
tags = []  # List to store detected tag IDs
outlineColor = (0, 255, 0)  # Color of tag outline
crossColor = (0, 255, 0)  # Color of cross

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
    

    # fx = 958.658
    # fy = 958.658
    # cx = 943.963
    # cy = 534.253
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    print([fx,fy,cx,cy])

    ll = 10
    gray = cv2.line(gray, (cent_x - ll, cent_y), (cent_x + ll, cent_y), crossColor, 2)
    gray = cv2.line(gray, (cent_x, cent_y - ll), (cent_x, cent_y + ll), crossColor, 2)

    # Display the tag ID near its center
    gray = cv2.putText(
        gray,
        str(detection.getId()),
        (cent_x - 4*ll, cent_y - 4*ll),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (128,128,128),
        3
    )

    # Estimate and display the pose of the tag
    pose = estimator.estimate(detection)
    rot = pose.rotation()
    
 
    

  

# Display the tags detected
print("Detected tags:", tags)


# Show the grayscale image with overlays
cv2.imshow("gray", gray)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window when done