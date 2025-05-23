import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join, abspath
import shutil
import json
from enum import IntEnum

# import sys
# sys.path.append(abspath(__file__))

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

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


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        # user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        # if user_input.lower() == 'y':
        #     shutil.rmtree(path_folder)
        #     makedirs(path_folder)
        # else:
        #     exit()
        exit()


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)

def save_time(filename,time):
    with open(filename, 'wb') as f:
        np.save(f, time)

if __name__ == "__main__":
    depth_num = 0
    color_num = 0
    data_path = "../DATA"
    path_output = join(data_path,"test_eg_horizontal_10mm_disp_no_slip_d{}_c{}".format(depth_num,color_num))  # ../ for parent directory, ./ for current directory
    path_depth = join(path_output,"depth")
    path_color = join(path_output,"color")  

    make_clean_folder(path_output)
    make_clean_folder(path_depth)
    make_clean_folder(path_color)
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    color_profiles, depth_profiles = get_profiles()

    
    # note: using 640 x 480 depth resolution produces smooth depth boundaries
    #       using rs.format.bgr8 for color image format for OpenCV based image visualization
    print('Using the default profiles: \n  color:{}, depth:{}'.format(
        color_profiles[color_num], depth_profiles[depth_num]))
    w, h, fps, fmt = depth_profiles[depth_num] #6
    config.enable_stream(rs.stream.depth, w, h, fmt, fps)
    w, h, fps, fmt = color_profiles[color_num] #21
    config.enable_stream(rs.stream.color, w, h, fmt, fps)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
    depth_sensor.set_option(rs.option.exposure,6000)
    depth_sensor.set_option(rs.option.gain, 48)
    # depth_sensor.set_option(rs.option.depth_units,0.001)
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale",depth_scale)

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    frame_count = 0
    prev_time = 0
    time = []
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            time_ms = frames.get_timestamp()
            if len(time) == 0:
                time.append(prev_time)
            else:
                time.append(time_ms - prev_time)
            prev_time = time_ms
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32)
            color_image = np.asanyarray(color_frame.get_data()).astype(np.float32)
            key = cv2.waitKey(1)
            #and key == ord('p')

            if frame_count == 0:
                save_intrinsic_as_json(
                    join(path_output,"camera_intrinsic.json"),
                    color_frame)
                
            cv2.imwrite("%s/%06d.png" % \
                    (path_depth, frame_count), depth_image)
            cv2.imwrite("%s/%06d.jpg" % \
                    (path_color, frame_count), color_image)
            print("Saved color + depth image %06d" % frame_count)
            frame_count += 1

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | \
                    (depth_image_3d <= 0), grey_color, color_image)

            # Render images
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Recorder Realsense', 1000,400)
            cv2.imshow('Recorder Realsense', images)
            
            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                cv2.destroyAllWindows()
                save_time(join(path_output,'time.npy'),time)
                print(np.array(time))
                print("SAVED SUCCESSFULLY")
                break

    finally:
        pipeline.stop()
