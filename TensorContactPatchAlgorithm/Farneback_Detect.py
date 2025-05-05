import cv2
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt
import vidVisualiser as vV

class farnebackDetect:
    def __init__(self, depth_profile, debug_mode):
        
        self.farneback = cv2.cuda.FarnebackOpticalFlow.create(numLevels=15,
                                                        pyrScale=0.5,
                                                        fastPyramids=True,
                                                        winSize=5,
                                                        numIters=6,
                                                        polyN=7,
                                                        polySigma=1.5,
                                                        flags=0
        )

        self.debug_mode = debug_mode
        self.depth_num = depth_profile
    
        if self.depth_num == 3:
            self.intrinsic = o3d.io.read_pinhole_camera_intrinsic("real_time_camera_intrinsic.json")
        elif self.depth_num == 0:
            self.intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_intrinsic.json")

        self.fx = self.intrinsic.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic.intrinsic_matrix[1, 2]
        self.width = self.intrinsic.width
        self.height = self.intrinsic.height

        self.gpu_flow = cv2.cuda.GpuMat(rows = self.height, cols = self.width, type = cv2.CV_32FC2)
        #cv2.cuda.GpuMat()
        self.gpu_flow_x = cv2.cuda_GpuMat(self.gpu_flow.size(), cv2.CV_32FC1)
        self.gpu_flow_y = cv2.cuda_GpuMat(self.gpu_flow.size(), cv2.CV_32FC1)
        
        

    def detect(self, gpu_prev_gray, gpu_curr_gray):
        self.farneback.calc(
            gpu_prev_gray, gpu_curr_gray, self.gpu_flow, None
        )

        cv2.cuda.split(self.gpu_flow, [self.gpu_flow_x, self.gpu_flow_y])

        gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
            self.gpu_flow_x, self.gpu_flow_y, angleInDegrees=True,
        )

        frame = gpu_curr_gray.download()
        return frame, gpu_magnitude, gpu_angle,self.gpu_flow_x, self.gpu_flow_y
    
        # vV.plotAndSavePlt(local_point_displacements,prev_3D_points,vmin,vmax)

        # # Show the figure in a non-blocking way
        # plt.savefig("./Farne_video/{}.png".format(self.count))
        # #plt.show(block=False)
        # self.count +=1
        
      
    
