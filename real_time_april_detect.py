import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import sys
import robotpy_apriltag
from enum import IntEnum

#================================================
# USE THIS CODE TO IMPORT OTHER FUNCTIONS FROM SCRIPTS IN THE SAME DIRECTORY
# from os.path import exists, join, abspath
# import sys
# sys.path.append(abspath(__file__))
# from realsense_helper import get_profiles
#================================================

def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()

    color_profiles = [] 
    depth_profiles = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print('Sensor: {}, {}'.format(name, serial))
        print('Supported video formats:')
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split('.')[-1]
                    print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                        video_type, w, h, fps, fmt))
                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

    return color_profiles, depth_profiles

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()

    color_profiles, depth_profiles = get_profiles()

    w, h, fps, fmt = depth_profiles[3]
    config.enable_stream(rs.stream.depth, w, h, fmt, fps)
    w, h, fps, fmt = color_profiles[18]
    config.enable_stream(rs.stream.color, w, h, fmt, fps)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
    depth_sensor.set_option(rs.option.exposure,1000)
    depth_sensor.set_option(rs.option.gain, 248)

    spatial_filter = rs.spatial_filter()  # Spatial filter
    spatial_filter.set_option(rs.option.filter_magnitude, 2)  # Adjust filter magnitude
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)  # Adjust smooth alpha
    spatial_filter.set_option(rs.option.filter_smooth_delta, 20)  # Adjust smooth delta

    temporal_filter = rs.temporal_filter()  # Temporal filter
    temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)  # Adjust smooth alpha
    temporal_filter.set_option(rs.option.filter_smooth_delta, 20)  # Adjust smooth delta

    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale", depth_scale)

    clipping_distance_in_meters = 3
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    frame_count = 0
    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily("tag36h11",1)
    outlineColor = (0, 255, 0)  # Color of tag outline
    crossColor = (0, 0, 255)  # Color of cross

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue

            # Apply spatial filter
            filtered_depth = spatial_filter.process(aligned_depth_frame)

            # Apply temporal filter
            filtered_depth = temporal_filter.process(filtered_depth)

            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)
            for detection in detections:
            # Store detected tag ID
            # tags.append(detection.getId())

            # Draw lines around the tag
                for i in range(4):
                    j = (i + 1) % 4
                    point1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
                    point2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
                    color_image = cv2.line(color_image, point1, point2, outlineColor, 2)

                # Mark the center of the tag with a cross
                cent_x = int(detection.getCenter().x)
                cent_y = int(detection.getCenter().y)
                
                # depth_val = np.asarray(source_rgbd_image.depth)[cent_y,cent_x]
                # print("Depth",depth_val)

                # fx = intrinsic.intrinsic_matrix[0, 0]
                # fy = intrinsic.intrinsic_matrix[1, 1]
                # cx = intrinsic.intrinsic_matrix[0, 2]
                # cy = intrinsic.intrinsic_matrix[1, 2]

                # print([fx,fy,cx,cy])

                # x = (cent_x - cx) * depth_val / fx
                # y = (cent_y - cy) * depth_val / fy 
                # z = depth_val

                # tag_loc.append([x,-y,-z])
                # print("Point Coords",[x,y,z])

                ll = 10
                color_image = cv2.line(color_image, (cent_x - ll, cent_y), (cent_x + ll, cent_y), crossColor, 2)
                color_image = cv2.line(color_image, (cent_x, cent_y - ll), (cent_x, cent_y + ll), crossColor, 2)

                # Display the tag ID near its center
                color_image = cv2.putText(
                    color_image,
                    str(detection.getId()),
                    (cent_x + ll, cent_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    crossColor,
                    3
                )

            key = cv2.waitKey(1)
            # if frame_count == 0:
            #     break
            frame_count += 1

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | \
                    (depth_image_3d <= 0), grey_color, color_image)
            
            #Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((color_image,depth_colormap))
            cv2.namedWindow('AprilTag Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AprilTag Detection', 1000, 400)
            cv2.imshow('AprilTag Detection', images)

            if key == 27:
                cv2.destroyAllWindows()
                break 

    finally:
        pipeline.stop()
