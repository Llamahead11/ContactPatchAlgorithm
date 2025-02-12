import open3d as o3d
import numpy as np
import pyrealsense2 as rs
from enum import IntEnum

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

class RealSenseManager:
    """
    Class to manage io of realsense D405 camera
    """
    def __init__(self,depth_profile=3,color_profile=18,exposure=1000,gain=248,enable_spatial=False,enable_temporal=False):
        self.depth_num = depth_profile
        self.color_num = color_profile
        self.exposure = exposure
        self.gain = gain
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal

        color_profiles, depth_profiles = get_profiles()

        w, h, fps, fmt = depth_profiles[self.depth_num]
        self.config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = color_profiles[self.color_num]
        self.config.enable_stream(rs.stream.color, w, h, fmt, fps)

        profile = self.pipeline.start(self.config)
        self.depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
        self.depth_sensor.set_option(rs.option.exposure,self.exposure)
        self.depth_sensor.set_option(rs.option.gain, self.gain)

        if self.enable_spatial:
            self.spatial_filter = rs.spatial_filter()  # Spatial filter
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)  # Adjust filter magnitude
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)  # Adjust smooth alpha
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)  # Adjust smooth delta

        if self.enable_temporal:
            self.temporal_filter = rs.temporal_filter()  # Temporal filter
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)  # Adjust smooth alpha
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)  # Adjust smooth delta

        depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale", depth_scale)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frames(self):
        """
        Returns depth and color image as NumPy array
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return None, None

        if self.enable_spatial:
            # Apply spatial filter
            filtered_depth = self.spatial_filter.process(aligned_depth_frame)

        if self.enable_temporal:
            # Apply temporal filter
            filtered_depth = self.temporal_filter.process(filtered_depth)
        
        if self.enable_spatial or self.enable_temporal:
            depth_image = np.asanyarray(filtered_depth.get_data())
        else:
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())

        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(color_image)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(depth_o3d, color_o3d)

        if self.depth_num == 3:
            intrinsic = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
        elif self.depth_num == 0:
            intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json")

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return depth_image, color_image, rgbd_image, pcd
        
    def stop(self):
        self.pipeline.stop()

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