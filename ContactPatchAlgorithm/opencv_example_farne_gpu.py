import cv2
import numpy as np
import time
# init video capture with video
from replay_realsense import read_RGB_D_folder
from capture_realsense import RealSenseManager


#imageStream = read_RGB_D_folder('realsense2',starting_index=100,depth_num=3,debug_mode=3)
rsManager = RealSenseManager(depth_profile=0,color_profile=0,exposure=21000,gain=20,enable_spatial=False,enable_temporal=False)
# video = 't2cam_vid.avi'
# cap = cv2.VideoCapture(video)
 
# # get default video FPS
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30
 
# # get total number of video frames
# num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

num_frames = 0

# read the first frame
ret = True 
#previous_frame = cap.read()

#count, depth_image, previous_frame, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
time_ms, count, depth_image, previous_frame, rgbd_image, pcd = rsManager.get_frames()
timers = {}
device = "gpu"
start_time = time.time()
 
if device == "cpu":
 
    # proceed if frame reading was successful
    if ret:
        # resize frame
        frame = cv2.resize(previous_frame, (960, 540))
 
        # convert to gray
        previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # create hsv output for optical flow
        hsv = np.zeros_like(frame, np.float32)
 
        # set saturation to 1
        hsv[..., 1] = 1.0

    while True:
        # start full pipeline timer
        start_full_time = time.time()
        
        # start reading timer
        start_read_time = time.time()
        
        # capture frame-by-frame
        #count, depth_image, frame, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
        #time_ms, depth_image, frame, rgbd_image, pcd = rsManager.get_frames()
        # end reading timer
        end_read_time = time.time()
        
        # add elapsed iteration time
        #timers["reading"].append(end_read_time - start_read_time)
        
        # if frame reading was not successful, break
        # if not imageStream.has_next():
        #     break
        
        # start pre-process timer
        start_pre_time = time.time()
        # resize frame
        frame = cv2.resize(frame, (960, 540))
        
        # convert to gray
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # end pre-process timer
        end_pre_time = time.time()
        
        # add elapsed iteration time
        #timers["pre-process"].append(end_pre_time - start_pre_time)

        # start optical flow timer
        start_of = time.time()
        
        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            previous_frame, current_frame, None, 0.5, 5, 15, 3, 5, 1.2, 0,
        )
        # end of timer
        end_of = time.time()
        
        # add elapsed iteration time
        #timers["optical flow"].append(end_of - start_of)

        # start post-process timer
        start_post_time = time.time()
        
        # convert from cartesian to polar coordinates to get magnitude and angle
        magnitude, angle = cv2.cartToPolar(
            flow[..., 0], flow[..., 1], angleInDegrees=True,
        )
        
        # set hue according to the angle of optical flow
        hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))
        
        # set value according to the normalized magnitude of optical flow
        hsv[..., 2] = cv2.normalize(
            magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
        )
        
        # multiply each pixel value to 255
        hsv_8u = np.uint8(hsv * 255.0)
        
        # convert hsv to bgr
        bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)
        
        # update previous_frame value
        previous_frame = current_frame
        
        # end post-process timer
        end_post_time = time.time()
        
        # add elapsed iteration time
        #timers["post-process"].append(end_post_time - start_post_time)
        
        # end full pipeline timer
        end_full_time = time.time()
        num_frames+=1
        print("FPS:", num_frames/(end_full_time-start_time))
        # add elapsed iteration time
        #timers["full pipeline"].append(end_full_time - start_full_time)

        # visualization
        cv2.imshow("original", frame)
        cv2.imshow("result", bgr)
        k = cv2.waitKey(1)
        if k == 27:
            break

# # elapsed time at each stage
# print("Elapsed time")
# for stage, seconds in timers.items():
#     print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))

# # calculate frames per second
# print("Default video FPS : {:0.3f}".format(fps))

# of_fps = (num_frames - 1) / sum(timers["optical flow"])
# print("Optical flow FPS : {:0.3f}".format(of_fps))

# full_fps = (num_frames - 1) / sum(timers["full pipeline"])
# print("Full pipeline FPS : {:0.3f}".format(full_fps))


if device == "gpu":
    # proceed if frame reading was successful
    if ret:
        # resize frame
        #frame = cv2.resize(previous_frame, (960, 540))
    
        # upload resized frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(previous_frame)
    
        # convert to gray
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
        # upload pre-processed frame to GPU
        gpu_previous = cv2.cuda_GpuMat()
        gpu_previous.upload(previous_frame)
    
        # create gpu_hsv output for optical flow
        gpu_hsv = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC3)
        gpu_hsv_8u = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)
    
        gpu_h = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
        gpu_s = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
        gpu_v = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
    
        # set saturation to 1
        gpu_s.upload(np.ones_like(previous_frame, np.float32))

    while True:
        # start full pipeline timer
        start_full_time = time.time()
    
        # start reading timer
        start_read_time = time.time()
    
        # capture frame-by-frame
        #count, depth_image, frame, rgbd_image, t2cam_pcd = imageStream.get_next_frame()
        time_ms, count, depth_image, frame, rgbd_image, pcd = rsManager.get_frames()
        # upload frame to GPU
        gpu_frame.upload(frame)
    
        # end reading timer
        end_read_time = time.time()
    
        # add elapsed iteration time
        #timers["reading"].append(end_read_time - start_read_time)
    
        # if frame reading was not successful, break
        # if not imageStream.has_next():
        #     break
    
        # start pre-process timer
        start_pre_time = time.time()
    
        # resize frame
        #gpu_frame = cv2.cuda.resize(gpu_frame, (960, 540))
    
        # convert to gray
        gpu_current = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
    
        # end pre-process timer
        end_pre_time = time.time()

        # start optical flow timer
        start_of = time.time()
        
        # create optical flow instance
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
            15, 0.5, False, 5, 6, 7, 1.5, 0,
        )
        # calculate optical flow
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
            gpu_flow, gpu_previous, gpu_current, None,
        )
        
        # end of timer
        end_of = time.time()
        
        # add elapsed iteration time
        #timers["optical flow"].append(end_of - start_of)

        # start post-process timer
        start_post_time = time.time()
        
        gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])
        
        # convert from cartesian to polar coordinates to get magnitude and angle
        gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
            gpu_flow_x, gpu_flow_y, angleInDegrees=True,
        )
        
        # set value to normalized magnitude from 0 to 1
        gpu_v = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)
        
        # get angle of optical flow
        angle = gpu_angle.download()
        angle *= (1 / 360.0) * (180 / 255.0)
        gpu_mean_std = cv2.cuda.GpuMat((2,1),cv2.CV_64FC1)
        cv2.cuda.meanStdDev(mtx=gpu_angle,dst=gpu_mean_std)

        mean,std = np.array(gpu_mean_std.download()).flatten()
        print("ANGLE:",mean,std)
        # set hue according to the angle of optical flow
        gpu_h.upload(angle)
        
        # merge h,s,v channels
        cv2.cuda.merge([gpu_h, gpu_s, gpu_v], gpu_hsv)
        
        # multiply each pixel value to 255
        #gpu_hsv.convertTo(gpu_hsv_8u, cv2.CV_16FC3, 255.0, 0.0)
        gpu_hsv.convertTo(rtype = cv2.CV_8U, dst = gpu_hsv_8u, alpha=255.0)
        
        # convert hsv to bgr
        gpu_bgr = cv2.cuda.cvtColor(gpu_hsv_8u, cv2.COLOR_HSV2BGR)
        
        # send original frame from GPU back to CPU
        frame = gpu_frame.download()
        
        # send result from GPU back to CPU
        bgr = gpu_bgr.download()
        
        # update previous_frame value
        gpu_previous = gpu_current


        
        # end post-process timer
        end_post_time = time.time()
        
        # add elapsed iteration time
        #timers["post-process"].append(end_post_time - start_post_time)
        
        # end full pipeline timer
        end_full_time = time.time()
        num_frames+=1
        print("FPS:", num_frames/(end_full_time-start_time))
        print("TIME:", end_full_time - start_full_time)
        # add elapsed iteration time
        #timers["full pipeline"].append(end_full_time - start_full_time)

        # visualization
        cv2.imshow("original", frame)
        cv2.imshow("result", bgr)
        k = cv2.waitKey(1)
        if k == 27:
            break

# # elapsed time at each stage
# print("Elapsed time")
# for stage, seconds in #.items():
#     print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))

# # calculate frames per second
# print("Default video FPS : {:0.3f}".format(fps))

# of_fps = (num_frames - 1) / sum(#["optical flow"])
# print("Optical flow FPS : {:0.3f}".format(of_fps))

# full_fps = (num_frames - 1) / sum(timers["full pipeline"])
# print("Full pipeline FPS : {:0.3f}".format(full_fps))
