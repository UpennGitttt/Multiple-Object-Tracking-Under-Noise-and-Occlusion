# Multiple-Object-Tracking-Under-Noise-and-Occlusion

Employed augmented belief updates and model the positions and uncertainties of occluders and tracked targets using Gaussian distributions, subsequently applying Kalman Filters and Extended Kalman Filters for tracking. For occluders, we use a hand recognition algorithm to distinguish between static and dynamic occlusions (such as hands). Used Iphone for video capture, 1920 * 1080 with 30 fps. 

1. Intrinsics Parameter Calibration: Utilized a checkerboard pattern with a grid size of 2.3 cm as a calibration target. After converting the images to grayscale, the corners of each checkerboard grid are detected and passed the corner coordinates to the cv2.calibrateCamera function in OpenCV.


2. Obtain Observation: In the project setup, April tags are used to obtain location and orientation information for observations.The April tag with tag ID 0 is designated as the world frame. A code that utilizes the camera’s intrinsic matrix and the dimensions of the April tags to determine their positions and orientations relative to the camera frame is developed as the implementation of calibrating iPhone and intrinsic matrix

3. EKF for state estimation: Propagate the dynamics:
In the two-dimensional context, we establish the state representation as [x, y, θ, v], where x and y denote the coordinates of the tag, θ signifies the tag’s orientation and v corresponds to the velocity of the tracked tag. The dynamic system equation, which is derived from Fig.2, can be expressed as follows:
<img width="325" alt="Screen Shot 2023-05-02 at 11 05 34 AM" src="https://user-images.githubusercontent.com/98191838/235706871-21cbdded-6b46-4285-9be5-1001b8c709fa.png">
And Jacobian matrix is:
<img width="388" alt="Screen Shot 2023-05-02 at 11 06 23 AM" src="https://user-images.githubusercontent.com/98191838/235707053-a4d6ebe8-598d-4416-a46e-19fd4a18ebcc.png">
Calculate mean and covariance at dynamic step:
<img width="247" alt="Screen Shot 2023-05-02 at 11 06 57 AM" src="https://user-images.githubusercontent.com/98191838/235707201-bb7ade2c-ffce-459a-9ae2-f78be6390c33.png">
calculate mean and covariance after incorporate our observations:

<img width="386" alt="Screen Shot 2023-05-02 at 11 07 38 AM" src="https://user-images.githubusercontent.com/98191838/235707373-ff6ebef0-8863-47d6-af32-2b799cb0d630.png">

A few results for EKF and KF tracker under 2D and 3D cases(Blue trajectory is Apriltag ground truth trajectory and green trajectory is the filter prediction):

![Screen Shot 2023-05-02 at 11 10 26 AM](https://user-images.githubusercontent.com/98191838/235708124-85401178-e363-4331-af99-3f864e40d02f.png)

![kf](https://user-images.githubusercontent.com/98191838/235705593-1a912201-32e8-4db0-bc2a-234e37ce716c.png)
![kf_2d](https://user-images.githubusercontent.com/98191838/235705624-fc7aacd2-607c-4b8c-a9bf-9b21ee384ec6.png)
![3d_ekf](https://user-images.githubusercontent.com/98191838/235705630-b85b3ad3-dc85-4323-b8c0-a10995471d7d.png)
