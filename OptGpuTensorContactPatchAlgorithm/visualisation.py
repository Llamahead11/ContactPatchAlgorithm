
import pyrealsense2 as rs
import time
import threading
from threading import Thread
import numpy as np
import multiprocessing
from queue import Queue
import time
import open3d as o3d
from enum import IntEnum
import cv2

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

class CameraProcess(Thread):
    def __init__(self,frameQueue,processedQueue, color_depthQueue, depth_profile=3,color_profile=18,enable_spatial=False,enable_temporal=False):
        Thread.__init__(self)
        self.frameQueue=frameQueue
        self.color_depthQueue=color_depthQueue
        self.processedQueue = processedQueue
        self.depth_num = depth_profile
        self.color_num = color_profile
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal

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

    def run(self):
        t1=time.perf_counter()
        while True:
            frameset=self.frameQueue.get()
            output = self.getProcessedFrame(frameset)
            if output is not None:
                self.processedQueue.put(output)
                print(f"Looper Thread | qsize {self.processedQueue.qsize()}, Processed frame {frameset.frame_number} in {np.round((time.perf_counter()-t1),3)}")
                t1=time.perf_counter()
    
    def getProcessedFrame(self,frameset):
        proc=self.processFrame(frameset)
        return proc
    
    def processFrame(self,frameset):
        '''
        Function that performs processing to the frameset, such as alignment, depth post-processing and others
        '''
        #time.sleep(0.01)
        time_global = frameset.get_timestamp()
        if self.count == 0:
            time_ms = 0
        else:
            time_ms = time_global - self.prev_time
        self.prev_time = time_global

        aligned_frames = self.align.process(frameset)
        self.count+=1

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
            self.depth_image = np.asanyarray(filtered_depth.get_data()).astype(np.float32)
            self.depth_image /= 10000
        else:
            self.depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32)
            self.depth_image /= 10000

        self.color_image = np.asanyarray(color_frame.get_data()).astype(np.float32)
        self.color_image /= 255

        self.color_depthQueue.put((self.color_image,self.depth_image))

        current_depth = o3d.t.geometry.Image(o3d.core.Tensor(self.depth_image))
        current_color = o3d.t.geometry.Image(o3d.core.Tensor(self.color_image))

        current_depth_cuda = current_depth.cuda()
        current_color_cuda = current_color.cuda()

        
        #filtered_depth = depth_o3d.filter_bilateral(kernel_size = 7, value_sigma= 10, dist_sigma = 20.0)
        vertex_map = current_depth_cuda.create_vertex_map(self.intrinsic)
        normal_map = vertex_map.create_normal_map()

        pcd_cuda = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        pcd_cuda.point.positions = vertex_map.as_tensor().reshape((-1, 3))
        pcd_cuda.point.normals = normal_map.as_tensor().reshape((-1, 3))
        pcd_cuda.point.colors = current_color_cuda.as_tensor().reshape((-1, 3))

        return time_ms, self.count, self.depth_image, current_color_cuda, pcd_cuda, vertex_map, normal_map 
        #return frameset

class CameraStreamer(Thread):
    depthShape = [848,480]
    rgbShape=[848,480]
    def __init__(self,frameQueue,depth_profile=3,color_profile=18,exposure=1000,gain=248,enable_spatial=False,enable_temporal=False):
        Thread.__init__(self)
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_stream(rs.stream.color, self.rgbShape[0], self.rgbShape[1], rs.format.bgr8, 30)
        #self.config.enable_stream(rs.stream.depth, self.depthShape[0], self.depthShape[1], rs.format.z16, 30)
        self.frameQueue=frameQueue

        self.depth_num = depth_profile
        self.color_num = color_profile
        self.exposure = exposure
        self.gain = gain
        #self.pipeline = rs.pipeline()
        #self.config = rs.config()
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal
        #self.frame_queue = rs.frame_queue(100)

        color_profiles, depth_profiles = get_profiles()

        w, h, fps, fmt = depth_profiles[self.depth_num]
        self.config.enable_stream(rs.stream.depth, w, h, fmt, 30)
        w, h, fps, fmt = color_profiles[self.color_num]
        self.config.enable_stream(rs.stream.color, w, h, fmt, 30)

        #self.pipeline.stop()
        
    def run(self):
        #self.profile = self.pipeline.start(self.config)
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
        self.loopFrames()
        
    def loopFrames(self):
        t1=time.perf_counter()
        while True:
            frameset = self.pipeline.wait_for_frames()
            self.frameQueue.put(frameset)
            print(f"Looper Thread | qsize {self.frameQueue.qsize()}, Acquired frame {frameset.frame_number} in {np.round((time.perf_counter()-t1),3)}")
            t1=time.perf_counter()

class CameraVis(Thread):
    def __init__(self,color_depthQueue):
        Thread.__init__(self)
        self.color_depthQueue=color_depthQueue
        clipping_distance_in_meters = 3  # 3 meter
        self.clipping_distance = clipping_distance_in_meters / 10000 #depth_scale

    def run(self):
        cv2.startWindowThread()
        cv2.namedWindow('Recorder Realsense', cv2.WINDOW_NORMAL)
        t1 = time.perf_counter()
        while True:
            color_image, depth_image = self.color_depthQueue.get()
            self.vis(color_image, depth_image)
            print(f"Looper Thread | qsize {self.color_depthQueue.qsize()}, Visualized frame in {np.round((time.perf_counter()-t1),3)}")
            t1=time.perf_counter()

    def vis(self, color_image, depth_image):
        #Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        #depth image is 1 channel, color is 3 channels
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        # bg_removed = np.where((depth_image_3d > self.clipping_distance) | \
        #         (depth_image_3d <= 0), grey_color, color_image)

        # # Render images
        # depth_colormap = cv2.applyColorMap(
        #     cv2.convertScaleAbs(bg_removed, alpha=0.09), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_image_3d))
        #cv2.namedWindow('Recorder Realsense', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Recorder Realsense', 1000,400)
        cv2.imshow('Recorder Realsense', images)
        
        # if 'esc' button pressed, escape loop and exit program
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()

# class CameraSave():
#     def __init__(self,color_depthQueue):
#         Thread.__init__(self)
#         self.color_depthQueue=color_depthQueue

#     def run(self):
#         pass

#     def save_cv2(self):
#         while True:



frameQueue=Queue(50)
color_depthQueue=Queue(50)
processedQueue = Queue(50)

#start threads
cameraStreamer=CameraStreamer(frameQueue, depth_profile=3,color_profile=18,exposure=21000,gain=80,enable_spatial=False,enable_temporal=False)
cameraProcess = CameraProcess(frameQueue,processedQueue,color_depthQueue, depth_profile=3,color_profile=18,enable_spatial=False,enable_temporal=False)
cameraVis = CameraVis(color_depthQueue)

#start threads
# num_consumers = 2  # or more depending on your CPU/GPU balance
# for _ in range(num_consumers):
#     processor = CameraProcess(frameQueue,processedQueue,color_depthQueue, depth_profile=3,color_profile=18,enable_spatial=False,enable_temporal=False)
#     processor.start()
cameraProcess.start()
cameraStreamer.start()
cameraVis.start()

nframes=0
iniTime=time.perf_counter()
t1=iniTime
while True:
    output = processedQueue.get()
    time_ms, count, depth_image, current_color_cuda, pcd_cuda, vertex_map, normal_map = output
    nframes+=1
    print(f"Time {time_ms} {count} since last frame {np.round((time.perf_counter()-t1)*1000,3)}ms, fps: {np.round(nframes/(time.perf_counter()-iniTime),3)}")
    t1=time.perf_counter()
#c.pipeline.stop()
