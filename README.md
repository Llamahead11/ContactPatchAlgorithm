# Contact Patch Algorithm using GPU-Accelerated 3D Processing
## Overview
- This repository holds the development of an algorithm that is able to dynamically measure the contact patch of the inner and outer tyre.
- Real-time capable pipeline for estimating tyre contact patch and deformation unver quasi-static and low speed rolling
- Uses Intel Realsense D405 Depth Camera and Nvidia Jetson Orin Nano 8GB (Hardware-in-the-loop)
- Primary Output: 3D spatio-temporally tracked point cloud of the inner and outer tyre surface as well as the contact patch

## Key Features
- CUDA-Accelerated Processing (CuPy, OptiX, OpenCV, GPU pipelines)
- Real-time processing (2Hz - online) and (5Hz - offline)
- 3D point cloud generation and tracking
- Deformation estimation as per tyre incompressibility assumption
- Net Force estimation as per inverse continuum mechanics

## Technical Highlights
- 3D Spatio-temporal tracking achieved through dense optical flow and depth maps
- Inner-to-Outer tyre deformation acheived through BVH Ray Casting (Optix - offline, PyTorch - online)
- Point cloud registration achieved through AprilTags and spatially constrained alignment
- Deformation comparison made against a 3D geometric tyre model created in Reality Capture (Epic Games)

## Results

### Static Loading

### Low Speed Rolling

### Static Steering
