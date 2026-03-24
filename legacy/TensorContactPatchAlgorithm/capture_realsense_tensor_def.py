import open3d as o3d
import numpy as np
import pyrealsense2 as rs
from enum import IntEnum
import numba

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

        if self.depth_num == 3:
            self.intrinsic = o3d.core.Tensor(o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json").intrinsic_matrix).cuda()
        elif self.depth_num == 0:
            self.intrinsic = o3d.core.Tensor(o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json").intrinsic_matrix).cuda()


        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.count = -1
        self.prev_time = 0

        self.cu = o3d.core.Device("CUDA:0")
        self.cpu = o3d.core.Device("CPU:0")
        self.dtype = o3d.core.float32
        self.arr_inc = np.arange(407040)
        self.old_to_new = -np.ones(407040, dtype=int)

        # Step 2: build triangle indices
        self.triangles = []
        H = 480
        W = 848
        for i in range(H - 1):
            for j in range(W - 1):
                idx = i * W + j
                v0 = idx
                v1 = idx + 1
                v2 = idx + W
                v3 = idx + W + 1
                self.triangles.append([v0, v1, v2])
                self.triangles.append([v1, v3, v2])

        self.triangles = np.array(self.triangles)

    def get_frames(self):
        """
        Returns depth and color image as NumPy array
        """
        self.count+=1
        frames = self.pipeline.wait_for_frames()
        time_global = frames.get_timestamp()
        if self.count == 0:
            time_ms = 0
        else:
            time_ms = time_global - self.prev_time
        self.prev_time = time_global
        
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
            depth_image = np.asanyarray(filtered_depth.get_data()).astype(np.float32)
            depth_image /= 10000
        else:
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32)
            depth_image /= 10000

        color_image = np.asanyarray(color_frame.get_data()).astype(np.float32)
        color_image /= 255

        current_depth = o3d.t.geometry.Image(o3d.core.Tensor(depth_image))
        current_color = o3d.t.geometry.Image(o3d.core.Tensor(color_image))

        current_depth_cuda = current_depth.cuda()
        current_color_cuda = current_color.cuda()

        
        #filtered_depth = depth_o3d.filter_bilateral(kernel_size = 7, value_sigma= 10, dist_sigma = 20.0)
        vertex_map = current_depth_cuda.create_vertex_map(self.intrinsic)
        normal_map = vertex_map.create_normal_map()

        pcd_cuda = o3d.t.geometry.PointCloud()
        pcd_cuda.point.positions = vertex_map.as_tensor().reshape((-1, 3))
        pcd_cuda.point.normals = normal_map.as_tensor().reshape((-1, 3))
        pcd_cuda.point.colors = current_color_cuda.as_tensor().reshape((-1, 3))
        #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        vertex_map_np = vertex_map.as_tensor().cpu().numpy()
        normal_map_np = normal_map.as_tensor().cpu().numpy()

        #pcd = pcd_cuda.cpu()

        vertices = vertex_map_np.reshape(-1, 3)
        normals = normal_map_np.reshape(-1, 3)

        mask = (vertices[:,2] != 0)
        v = np.count_nonzero(mask)

        valid_triangles = filter_triangles(self.triangles, mask)

        mesh = o3d.t.geometry.TriangleMesh(o3d.core.Device("CPU:0"))
        mesh.vertex.positions = o3d.core.Tensor(vertices, o3d.core.float32, o3d.core.Device("CPU:0"))
        mesh.triangle.indices = o3d.core.Tensor(valid_triangles, o3d.core.int32, o3d.core.Device("CPU:0"))
        mesh.vertex.normals = o3d.core.Tensor(normals,o3d.core.float32, o3d.core.Device("CPU:0"))


        #return time_ms, self.count, depth_image, color_image, pcd, pcd_cuda, vertex_map_np, vertex_map, normal_map_np, normal_map, mesh
        return time_ms, self.count, depth_image, color_image, pcd_cuda, vertex_map_np, vertex_map, normal_map, mesh

    def stop(self):
        self.pipeline.stop()

@numba.jit(nopython=True)
def filter_triangles(triangles, mask):
    out = np.empty((triangles.shape[0], 3), dtype=np.int32)
    count = 0
    for i in range(triangles.shape[0]):
        v0, v1, v2 = triangles[i]
        if mask[v0] and mask[v1] and mask[v2]:
            out[count, 0] = v0
            out[count, 1] = v1
            out[count, 2] = v2
            count += 1
    return out[:count]

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