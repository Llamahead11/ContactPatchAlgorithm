import open3d as o3d
import cv2
import numpy as np
import apriltag
import time
import pyrealsense2 as rs

#import matplotlib.pyplot as plt
import sys
from enum import IntEnum

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
    options = apriltag.DetectorOptions(families = "tag36h11")
    # options = apriltag.DetectorOptions(families='tag36h11',
    #                              border=1,
    #                              nthreads=4,
    #                              quad_decimate=1.0,
    #                              quad_blur=0.0,
    #                              refine_edges=True,
    #                              refine_decode=False,
    #                              refine_pose=False,
    #                              debug=False,
    #                              quad_contours=True)
    detector = apriltag.Detector(options)
    
    outlineColor = (0, 255, 0) # Color of tag outline
    crossColor = (0, 255, 255) # Color of cross
    
    intrinsic = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
    depth_scale = 0.1
    origin = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    source_pcd = o3d.geometry.PointCloud()
    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    #vis.add_geometry(origin.to_legacy())
    #create_o3d_obj = False
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

            depth_image = np.asarray(filtered_depth.get_data())
            color_image = np.asarray(color_frame.get_data())

            #Generate T2Cam 
            
            depth_image_scaled = depth_image #* depth_scale
            depth_image_scaled = o3d.geometry.Image(depth_image_scaled)
            color_image_o3d = o3d.geometry.Image(color_image)
            source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_scaled)
            # source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)
            # source_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)

            # if not create_o3d_obj:
            #     vis.add_geometry(source_pcd)  # add point cloud
            #     create_o3d_obj = True  # change flag
            # else:
            #     vis.update_geometry(source_pcd)  # update point cloud
            
            # if not vis.poll_events():
            #     break
            # vis.update_renderer()
            # time.sleep(1)
            # transform_array = []

            # mesh_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
            
            tags = []
            tag_loc = []
            # t2cam_correspondence = []

            # Detect Tags from image stream
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)
            for detection in detections:
                # Store detected tag ID
                #tags.append(detection.tag_id)

                # Draw lines around the tag
                #(ptA, ptB, ptC, ptD) = detection.corners
                for i in range(4):
                    j = (i + 1) % 4
                    point1 = (int(detection.corners[i][0]), int(detection.corners[i][1]))
                    point2 = (int(detection.corners[j][0]), int(detection.corners[j][1]))
                    color_image = cv2.line(color_image, point1, point2, outlineColor, 2)

                # Mark the center of the tag with a cross
                cent_x = int(detection.center[0])
                cent_y = int(detection.center[1])
                
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
                color_image = cv2.line(color_image, (cent_x - ll, cent_y), (cent_x + ll, cent_y), crossColor, 2)
                color_image = cv2.line(color_image, (cent_x, cent_y - ll), (cent_x, cent_y + ll), crossColor, 2)

                # Display the tag ID near its center
                color_image = cv2.putText(
                    color_image,
                    str(detection.tag_id),
                    (cent_x + ll, cent_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    crossColor,
                    2
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
            
            # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed,depth_colormap))
            cv2.namedWindow('AprilTag Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AprilTag Detection', 1000, 400)
            cv2.imshow('AprilTag Detection', images)

            if key == 27:
                cv2.destroyAllWindows()
                # vis.close()
                # vis.destroy_window()
                break 

    finally:
        pipeline.stop()
        
