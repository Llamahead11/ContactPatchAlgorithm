import cv2
import numpy as np
import open3d as o3d
import robotpy_apriltag
from capture_realsense_tensor import RealSenseManager
from numba import njit

class DetectAprilTagsWindows():
    def __init__(self, depth_profile, debug_mode = True):
        self.depth_num = depth_profile
        self.debug_mode = debug_mode
        self.detector = robotpy_apriltag.AprilTagDetector()
        self.detector.addFamily("tag36h11",1)

        if self.depth_num == 3:
            self.intrinsic = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
        elif self.depth_num == 0:
            self.intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json")

        self.fx = self.intrinsic.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic.intrinsic_matrix[1, 2]
        self.outlineColor = (0, 255, 0) # Color of tag outline GBR
        self.crossColor = (0, 255, 255) # Color of cross GBR

    def input_frame(self, color_image):
        self.color_image = (color_image.copy()*255).astype(np.uint8)
        self.gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        self.detections = self.detector.detect(self.gray)
        self.tag_IDs = []
        self.tag_locations = []
        self.fail_count_per_frame = 0
        self.number_of_detections = len(self.detections)
        self.pcd_IDs = []

    def process_3D_locations(self, vertex_map_np):
        for detection in self.detections:
            # Determine the center of the tag with a cross
            cent_x = int(detection.getCenter().x)
            cent_y = int(detection.getCenter().y)

            x,y,z = vertex_map_np[cent_y,cent_x]

            if z == 0.0:
                self.fail_count_per_frame += 1
            else:
                self.tag_IDs.append(detection.getId())
                self.tag_locations.append([x,y,z])
                self.pcd_IDs.append(cent_x+cent_y*848)

            # Draw around the tag
            if self.debug_mode:
                for i in range(4):
                    j = (i + 1) % 4
                    point1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
                    point2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
                    self.color_image = cv2.line(self.color_image, point1, point2, self.outlineColor, 2)

                ll = 10
                self.color_image = cv2.line(self.color_image, (cent_x - ll, cent_y), (cent_x + ll, cent_y), self.crossColor, 2)
                self.color_image = cv2.line(self.color_image, (cent_x, cent_y - ll), (cent_x, cent_y + ll), self.crossColor, 2)

                # Display the tag ID near its center
                self.color_image = cv2.putText(
                    self.color_image,
                    str(detection.getId()),
                    (cent_x + ll, cent_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.crossColor,
                    2
                )
        return self.tag_IDs, self.tag_locations, self.pcd_IDs






        
        
