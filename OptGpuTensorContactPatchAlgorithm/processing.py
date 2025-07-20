import multiprocessing as mp
import time
import numpy as np
import open3d as o3d
import cupy as cp

class FrameGrabber(mp.Process):
    def __init__(self, queues, stop_event):
        super().__init__()
        self.queues = queues  # [save_q, vis_q, proc_q]
        self.stop_event = stop_event

    def run(self):
        import pyrealsense2 as rs
        import numpy as np

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 90)
        pipeline.start(config)

        try:
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames()
                frame_id = frames.get_frame_number()
                depth = np.asanyarray(frames.get_depth_frame().get_data())
                color = np.asanyarray(frames.get_color_frame().get_data())

                frame = (frame_id, depth, color)
                for q in self.queues:
                    q.put(frame)
                print(f"Aquired Frame {frame_id}")
        finally:
            pipeline.stop()

class SaveProcess(mp.Process):
    def __init__(self, queue, path_depth, path_color, stop_event):
        super().__init__()
        self.q = queue
        self.path_depth = path_depth
        self.path_color = path_color
        self.stop_event = stop_event

    def run(self):
        import cv2
        import os
        while not self.stop_event.is_set():
            try:
                frame_id, depth, color = self.q.get()
                cv2.imwrite(os.path.join(self.path_depth, f"{frame_id:06}.png"), depth)
                cv2.imwrite(os.path.join(self.path_color, f"{frame_id:06}.jpg"), color)
                print(f"Saved Frame {frame_id} with Queue size:", self.q.qsize())
            except:
                continue

class VisProcess(mp.Process):
    def __init__(self, queue, stop_event):
        super().__init__()
        self.q = queue
        self.stop_event = stop_event

    def run(self):
        import cv2
        cv2.startWindowThread()
        cv2.namedWindow("Color")
        cv2.startWindowThread()
        cv2.namedWindow("Depth")
        while not self.stop_event.is_set():
            try:
                frame_id, depth, color = self.q.get()
                cv2.imshow("Color", color)
                cv2.imshow("Depth", depth)
                if cv2.waitKey(1) == 27:
                    self.stop_event.set()
                print(f"Vis Frame {frame_id} with Queue size:", self.q.qsize())
            except:
                continue

class CUDAProcess(mp.Process):
    def __init__(self, queue, result_queue, stop_event):
        super().__init__()
        
        self.q = queue
        self.result_q = result_queue
        self.stop_event = stop_event
        self.depth_num = 3

    def run(self):
        import cupy as cp
        import open3d as o3d
        self.stream = cp.cuda.Stream(non_blocking=True)
        if self.depth_num == 3:
            self.intrinsic = o3d.core.Tensor(o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json").intrinsic_matrix).cuda()
        elif self.depth_num == 0:
            self.intrinsic = o3d.core.Tensor(o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json").intrinsic_matrix).cuda()

        self.pcd_cuda = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
        while not self.stop_event.is_set():
            try:
                frame_id, depth_np, color_np = self.q.get()
                # GPU work here
                # Return to result_q if needed
                with self.stream:
                    self.cp_depth = cp.asarray(depth_np.astype(np.float32)/10000)
                    self.cp_color = cp.asarray(color_np.astype(np.float32)/255)
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
                self.result_q.put((frame_id, depth_np,self.pcd_cuda.cpu()))
                print(f"Preprocessed Frame {frame_id} with Queue size:", self.q.qsize())
            except:
                continue


def main():
    mp.set_start_method("spawn")  # Safe for CUDA
    stop_event = mp.Event()

    save_q = mp.Queue(50)
    vis_q = mp.Queue(50)
    proc_q = mp.Queue(50)
    result_q = mp.Queue(50)

    grabber = FrameGrabber([save_q, vis_q, proc_q], stop_event)
    saver = SaveProcess(save_q, "./depth", "./color", stop_event)
    visualizer = VisProcess(vis_q, stop_event)
    processor = CUDAProcess(proc_q, result_q, stop_event)

    for p in [grabber, saver, visualizer, processor]:
        p.start()

    nframes=0
    iniTime=time.perf_counter()
    t1=iniTime
    try:
        while not stop_event.is_set():
            frame_id,_,pcd_cpu = result_q.get()
            nframes += 1
            print(f"Time {frame_id} since last frame {np.round((time.perf_counter()-t1)*1000,3)}ms, fps: {np.round(nframes/(time.perf_counter()-iniTime),3)}")
            t1 = time.perf_counter()
            # if not result_q.empty():
            #     result = result_q.get()
            #     print("Processed:", result)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        for p in [grabber, saver, visualizer, processor]:
            p.join()

if __name__ == "__main__":
    main()