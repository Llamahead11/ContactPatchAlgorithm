import cv2
import numpy as np

img_rgb = cv2.imread('realsense2/color/000150.jpg')  # Assume test.jpg is feature-rich
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img_bgr)

gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

surf = cv2.cuda.SURF_CUDA_create(100,5,4)
kp, des = surf.detectWithDescriptors(gpu_gray, None)

cpu_kp = surf.downloadKeypoints(kp)

print("Keypoints:", len(cpu_kp))

img_with_keypoints = cv2.drawKeypoints(img_bgr, cpu_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
cv2.imshow("Keypoints", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print("Descriptor shape:", des.shape if des is not None else "None")