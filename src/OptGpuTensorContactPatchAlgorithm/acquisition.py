
import pyrealsense2 as rs
import time
from threading import Event
from threading import Thread
import numpy as np
import multiprocessing
from queue import Queue
from queue import Empty
import time
import open3d as o3d
from enum import IntEnum
import cv2
import cupy as cp
from os import makedirs
from os.path import exists, join, abspath
import json
from app_vis import Viewer3D

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
    def __init__(self, stop_event,frameQueue,processedQueue, color_depthQueue, depth_profile=3,color_profile=18,enable_spatial=False,enable_temporal=False):
        Thread.__init__(self)
        self.stop_event = stop_event
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.frameQueue=frameQueue
        self.color_depthQueue=color_depthQueue
        self.processedQueue = processedQueue
        self.depth_num = depth_profile
        self.color_num = color_profile
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal

        self.pinned_depth = cp.cuda.alloc_pinned_memory( 848 * 480 * 4)
        self.np_pinned_depth = np.frombuffer(self.pinned_depth, dtype=np.float32, count=480*848).reshape((480, 848)) # Numpy View of buffer
        self.pinned_color = cp.cuda.alloc_pinned_memory( 848 * 480 * 3 * 4)
        self.np_pinned_color = np.frombuffer(self.pinned_color, dtype=np.float32, count=480*848*3).reshape((480,848,3)) # Numpy View of buffer

        self.cp_depth = cp.asarray(self.np_pinned_depth)
        self.cp_color = cp.asarray(self.np_pinned_color)

        self.cp_depth.set(self.np_pinned_depth)
        self.cp_color.set(self.np_pinned_color)

        # self.np_pinned_depth = np_pinned_depth
        # self.np_pinned_color = np_pinned_color


        # self.dl_depth = cp_depth.toDlpack()
        # self.dl_color = cp_color.toDlpack()

        # self.o3d_depth = o3d.core.Tensor.from_dlpack(self.dl_depth)
        # self.o3d_color = o3d.core.Tensor.from_dlpack(self.dl_color)

        # self.o3d_depth_image = o3d.t.geometry.Image(self.o3d_depth)
        # self.o3d_color_image = o3d.t.geometry.Image(self.o3d_color)


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

        self.pcd_cuda = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))

    def run(self):
        t1=time.perf_counter()
        while not self.stop_event.is_set():
            frameset=self.frameQueue.get()
            # try:
            #     frameset = self.frameQueue.get_nowait()
                
            # except Empty:
            #     pass
            output = self.getProcessedFrame(frameset)
            if output is not None:
                self.processedQueue.put(output)
                # print(f"Looper Thread | qsize {self.processedQueue.qsize()}, Processed frame {frameset.frame_number} in {np.round((time.perf_counter()-t1),3)}")
                print(f"Looper Thread {self.name} | qsize {self.processedQueue.qsize()}, Processed frame {frameset.frame_number} in  {np.round((time.perf_counter()-t1),3)}")
                t1=time.perf_counter()
    
    def getProcessedFrame(self,frameset):
        proc=self.processFrame(frameset)
        return proc
    
    def processFrame(self,frameset):
        '''
        Function that performs processing to the frameset, such as alignment, depth post-processing and others
        '''
        #time.sleep(0.01)

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

        np.copyto(self.np_pinned_depth, self.depth_image)
        np.copyto(self.np_pinned_color, self.color_image)

        self.color_depthQueue.put((self.color_image,self.depth_image))

        # try:
        #     self.color_depthQueue.put_nowait((self.color_image,self.depth_image))
        # except self.frameQueue.Full:
        #     print("Queue full, dropped frame")

        with self.stream:
            self.cp_depth.set(self.np_pinned_depth)
            self.cp_color.set(self.np_pinned_color)
            # regenerate dlpack / o3d tensors here
            self.dl_depth = self.cp_depth.toDlpack()
            self.dl_color = self.cp_color.toDlpack()
            self.o3d_depth = o3d.core.Tensor.from_dlpack(self.dl_depth)
            self.o3d_color = o3d.core.Tensor.from_dlpack(self.dl_color)

            self.o3d_depth_image = o3d.t.geometry.Image(self.o3d_depth)
            self.o3d_color_image = o3d.t.geometry.Image(self.o3d_color)

            vertex_map = self.o3d_depth_image.create_vertex_map(self.intrinsic)
            normal_map = vertex_map.create_normal_map()

            self.pcd_cuda.point.positions = vertex_map.as_tensor().reshape((-1, 3))
            self.pcd_cuda.point.normals = normal_map.as_tensor().reshape((-1, 3))
            self.pcd_cuda.point.colors = self.o3d_color.reshape((-1, 3))

            self.stream.synchronize()

        # cp_depth = cp.asarray(self.np_pinned_depth)
        # cp_color = cp.asarray(self.np_pinned_color)

        # o3d_depth = o3d.core.Tensor.from_dlpack(cp_depth.toDlpack())
        # o3d_color = o3d.core.Tensor.from_dlpack(cp_color.toDlpack())

        # o3d_depth_image = o3d.t.geometry.Image(o3d_depth)
        # o3d_color_image = o3d.t.geometry.Image(o3d_color)

        # vertex_map = o3d_depth_image.create_vertex_map(self.intrinsic)
        # normal_map = vertex_map.create_normal_map()

        # self.pcd_cuda.point.positions = vertex_map.as_tensor().reshape((-1, 3))
        # self.pcd_cuda.point.normals = normal_map.as_tensor().reshape((-1, 3))
        # self.pcd_cuda.point.colors = o3d_color.reshape((-1, 3))

        return self.count, self.depth_image, self.o3d_color_image, self.pcd_cuda, vertex_map, normal_map 
        #return frameset

class CameraStreamer(Thread):
    depthShape = [848,480]
    rgbShape=[848,480]
    def __init__(self,stop_event,path_output,saveQueue,frameQueue,depth_profile=3,color_profile=18,exposure=1000,gain=248,enable_spatial=False,enable_temporal=False):
        Thread.__init__(self)
        self.stop_event = stop_event
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_stream(rs.stream.color, self.rgbShape[0], self.rgbShape[1], rs.format.bgr8, 30)
        #self.config.enable_stream(rs.stream.depth, self.depthShape[0], self.depthShape[1], rs.format.z16, 30)
        self.frameQueue=frameQueue
        self.saveQueue = saveQueue
        self.depth_num = depth_profile
        self.color_num = color_profile
        self.exposure = exposure
        self.gain = gain
        #self.pipeline = rs.pipeline()
        #self.config = rs.config()
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal
        #self.frame_queue = rs.frame_queue(100)
        self.prev_time = 0
        self.path_output = path_output
        color_profiles, depth_profiles = get_profiles()

        w, h, fps, fmt = depth_profiles[self.depth_num]
        self.config.enable_stream(rs.stream.depth, w, h, fmt, 60)
        w, h, fps, fmt = color_profiles[self.color_num]
        self.config.enable_stream(rs.stream.color, w, h, fmt, 60)
        self.time_arr = []
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
        while not self.stop_event.is_set():
            frameset = self.pipeline.wait_for_frames()
            time_global = frameset.get_timestamp()
            if self.prev_time == 0:
                time_ms = self.prev_time
            else:
                time_ms = time_global - self.prev_time
            self.prev_time = time_global
            # try:
            #     self.frameQueue.put_nowait(frameset)
            # except self.frameQueue.Full:
            #     print("Queue full, dropped frame")
            self.frameQueue.put(frameset)
            self.saveQueue.put(frameset)
            self.time_arr.append(time_ms)
            print(f"Looper Thread | qsize {self.frameQueue.qsize()}, Acquired frame {frameset.frame_number} in {time_ms} {np.round((time.perf_counter()-t1),3)}")
            t1=time.perf_counter()
        
        if self.stop_event.is_set():
            save_time(join(self.path_output,'time.npy'),self.time_arr)

class CameraVis(Thread):
    def __init__(self,stop_event,color_depthQueue):
        Thread.__init__(self)
        self.stop_event = stop_event
        self.color_depthQueue=color_depthQueue
        clipping_distance_in_meters = 3  # 3 meter
        self.clipping_distance = clipping_distance_in_meters / 10000 #depth_scale

    def run(self):
        cv2.startWindowThread()
        cv2.namedWindow('Recorder Realsense Color', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Recorder Realsense Depth', cv2.WINDOW_NORMAL)
        cv2.moveWindow("Recorder Realsense Color", 100, 100)
        cv2.moveWindow("Recorder Realsense Depth", 1000, 100)
        t1 = time.perf_counter()
        while not self.stop_event.is_set():
            #color_image, depth_image = self.color_depthQueue.get()
            while self.color_depthQueue.qsize() > 1:
                _ = self.color_depthQueue.get_nowait()
            color_image, depth_image = self.color_depthQueue.get()  # get most recent
            self.vis(color_image, depth_image)
            print(f"Looper Thread | qsize {self.color_depthQueue.qsize()}, Visualized frame in {np.round((time.perf_counter()-t1),3)}")
            t1=time.perf_counter()
            # try:
            #     #frameset = self.frameQueue.get_nowait()
                
            # except Empty:
            #     pass
            
    def vis(self, color_image, depth_image):
        cv2.imshow('Recorder Realsense Color', color_image)
        cv2.imshow("Recorder Realsense Depth", depth_image)
        
        # if 'esc' button pressed, escape loop and exit program
        key = cv2.waitKey(1)
        if key == 27:
            self.stop_event.set()
            cv2.destroyAllWindows()

class CameraSave(Thread):
    def __init__(self,stop_event,path_depth,path_color,path_output,saveQueue):
        Thread.__init__(self)
        self.stop_event = stop_event
        self.path_depth = path_depth
        self.path_color = path_color
        self.path_output = path_output
        self.saveQueue=saveQueue
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def run(self):
        t1 = time.perf_counter()
        while not self.stop_event.is_set():
            frameset = self.saveQueue.get()  # get most recent
            self.save_cv2(frameset)
            print(f"Looper Thread | qsize {self.saveQueue.qsize()}, Saved frame {frameset.frame_number} in {np.round((time.perf_counter()-t1),3)}")
            t1=time.perf_counter()

    def save_cv2(self,frameset):
        aligned_frames = self.align.process(frameset)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        cv2.imwrite("%s/%06d.png" % \
                (self.path_depth,frameset.frame_number), np.asanyarray(aligned_depth_frame.get_data()))
        cv2.imwrite("%s/%06d.jpg" % \
                (self.path_color,frameset.frame_number), np.asanyarray(color_frame.get_data()))
        
        if self.stop_event.is_set():
            save_intrinsic_as_json(
                    join(self.path_output,"camera_intrinsic.json"),
                    color_frame)

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

def save_time(filename,time_arr):
    with open(filename, 'wb') as f:
        np.save(f, time_arr)

def main():
    depth_num = 3 #0
    color_num = 18 #0
    data_path = "../DATA"
    path_output = join(data_path,"new_pipeline_test_1_d{}_c{}".format(depth_num,color_num))  # ../ for parent directory, ./ for current directory
    #test_eg_horizontal_10mm_disp_no_slip
    path_depth = join(path_output,"depth")
    path_color = join(path_output,"color")  

    make_clean_folder(path_output)
    make_clean_folder(path_depth)
    make_clean_folder(path_color)

    frameQueue=Queue(50)
    color_depthQueue=Queue(50)
    processedQueue = Queue(50)
    saveQueue = Queue(50)
    stop_event = Event()

    #start threads
    cameraStreamer=CameraStreamer(stop_event, path_output,saveQueue, frameQueue, depth_profile=3,color_profile=18,exposure=3000,gain=240,enable_spatial=False,enable_temporal=False)
    #cameraProcess = CameraProcess(frameQueue,processedQueue,color_depthQueue, depth_profile=3,color_profile=18,enable_spatial=False,enable_temporal=False)
    cameraVis = CameraVis(stop_event, color_depthQueue)
    #cameraSave = CameraSave(stop_event,path_depth, path_color,path_output,saveQueue)

    #start threads
    processors = []
    for i in range(2):
        processor = CameraProcess(stop_event, frameQueue, processedQueue, color_depthQueue,
                                depth_profile=3, color_profile=18,
                                enable_spatial=False, enable_temporal=False)
        processor.name = f"Processor-{i}"
        processor.start()
        processors.append(processor)
    #cameraProcess.start()
    cameraStreamer.start()
    cameraVis.start()
    #cameraSave.start()
    savers = []
    for i in range(3):
        saver = CameraSave(stop_event,path_depth, path_color,path_output,saveQueue)
        saver.name = f"Saver-{i}"
        saver.start()
        savers.append(saver)

    nframes=0
    viewer3d = Viewer3D("Outer Deformation and Tracking")
    iniTime=time.perf_counter()
    t1=iniTime
    try:
        while not stop_event.is_set():
            output = processedQueue.get()
            count, depth_image, current_color_cuda, pcd_cuda, vertex_map, normal_map = output
            nframes += 1
            # viewer3d.update_cloud(pcd_cuda = pcd_cuda.uniform_down_sample(20).cpu())
            # viewer3d.tick()
            print(f"Time {count} since last frame {np.round((time.perf_counter()-t1)*1000,3)}ms, fps: {np.round(nframes/(time.perf_counter()-iniTime),3)}")
            t1 = time.perf_counter()
    except Exception as e:
        print("Main thread error or exit:", e)
    finally:
        print("Stopping threads...")
        stop_event.set()
        print("set")
        cameraStreamer.join()
        print("joined stream")
        #cameraSave.join()
        for s in savers:
            s.join()
        print("joined save")
        cameraVis.join()
        print("joined vis")
        for p in processors:
            p.join()
        print("All threads stopped.")
    #c.pipeline.stop()

if __name__ == "__main__":
    main()