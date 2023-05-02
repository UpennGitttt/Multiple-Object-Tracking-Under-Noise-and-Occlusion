import cv2
import numpy as np
import mediapipe as mp
import apriltag
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from ekf import *

from kf import *
from utils import *


def bhattacharyya(mean1, cov1, mean2, cov2):
    cov=(1/2)*(cov1+cov2)
    t1=(1/8)*np.sqrt((mean1-mean2)@np.linalg.inv(cov)@(mean1-mean2).T)
    t2=(1/2)*np.log(np.linalg.det(cov)/np.sqrt(np.linalg.det(cov1)*np.linalg.det(cov2)))
    return t1+t2


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

mtx = np.array([[1.66016657e+03, 0.00000000e+00, 5.45114552e+02],
                [0.00000000e+00, 1.66228213e+03, 9.94387658e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

image_path = '/Users/yuhaoyou/PycharmProjects/pythonProject1/output_frame_100.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

options_1 = apriltag.DetectorOptions(families='tag36h11',
                                     border=1,
                                     nthreads=4,
                                     quad_decimate=5.0,
                                     quad_blur=0.0,
                                     refine_edges=True,
                                     refine_decode=False,
                                     refine_pose=False,
                                     debug=False,
                                     quad_contours=True)


def main():
    video_path = "/Users/yuhaoyou/PycharmProjects/pythonProject1/IMG_6003.MOV"
    # video_path = "/Users/yuhaoyou/PycharmProjects/pythonProject1/IMG_5940.MOV"
    # read_videos(video_path)
    cap = cv2.VideoCapture(video_path)  # Use 0 for the default camera# 创建AprilTag检测器
    tag_trajectories = {}
    ekf_trajectories = {}
    hand_traj = []
    # flag for initialize
    flag_initial = False
    # initial setting
    epi = 0.00001
    tag_states = np.zeros((5, 4))  # target number x state dimension
    tag_states_p = np.zeros((5, 4))
    error_cov = np.array([np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4)]) * epi
    hand_world_coor_p = np.zeros((3, 1))
    # define observation covirance
    obs_cov = np.eye(3) * 0.0005
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Unable to read the frame.")
                break

            fimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            # results = hands.process(frame_rgb)

            detector_1 = apriltag.Detector(options_1)

            # 对图像进行AprilTag检测
            detections_1 = detector_1.detect(fimage)

            # define tag size
            tag_size = 3.1 * 0.01
            camera_params = (mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
            pose, e0, e1 = detector_1.detection_pose(detections_1[0], camera_params, tag_size)
            T = pose[:3, -1].reshape(-1, 1)  # T^C_W
            R_base = pose[:3, :3]  # R^C_W
            # pixel_world = np.zeros(())
            pixel_world = {}

            target = {0, 1, 2, 3, 4}
            detected = []

            for detection in detections_1:
                pose_cur, e0, e1 = detector_1.detection_pose(detection, camera_params, tag_size)
                tag_id = detection.tag_id
                vec_1 = pose_cur[:3, -1].reshape(-1, 1)

                R_cur = R_base[:3, :3].T @ pose_cur[:3, :3]
                # 从旋转矩阵创建一个Rotation对象
                rotation = R.from_matrix(R_cur)
                # 将旋转对象转换为欧拉角（使用ZYX顺序，也称为yaw-pitch-roll顺序）
                euler_angles = rotation.as_euler('ZYX', degrees=False)
                theta = euler_angles[0]
                pixel_world[tag_id] = (R_base.T @ (vec_1 - T))

                if not flag_initial:
                    tag_states[tag_id][:2] = pixel_world[tag_id][:2].reshape(-1)
                    tag_states[tag_id][2] = theta
                    if tag_id == 4:
                        flag_initial = True  # means that we've finished the initialization.
                else:
                    v = np.linalg.norm((tag_states[tag_id][:2] - tag_states_p[tag_id][:2])) / (1 / 30)
                    cur_obs = np.hstack((pixel_world[tag_id][:2].reshape(-1), theta, np.array(v))).squeeze()
                    tag_states_p[tag_id] = tag_states[tag_id]
                    tag_states[tag_id], error_cov[tag_id] = kf(cur_obs, tag_states[tag_id], 1 / 30, error_cov[tag_id])

                detected.append(tag_id)
                if tag_id not in tag_trajectories:
                    tag_trajectories[tag_id] = []

                if tag_id not in ekf_trajectories:
                    ekf_trajectories[tag_id] = []

                tag_trajectories[tag_id].append([int(detection.center[0]), int(detection.center[1])])
                ekf_trajectories[tag_id].append(
                    world_to_image_coordinates([tag_states[tag_id][0], tag_states[tag_id][1], 0], mtx, pose))

            detected = set(detected)
            # missed is missed tag_id
            missed = target - detected

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    middle_finger_mcp = hand_landmarks.landmark[9]

                    # Extract normalized x and y coordinates
                    normalized_x = middle_finger_mcp.x
                    normalized_y = middle_finger_mcp.y
                    image_height, image_width, _ = frame.shape
                    x = int(normalized_x * image_width)
                    y = int(normalized_y * image_height)

                    hand_homo_coor = np.vstack((x, y, 1))
                    hand_world_coor = np.linalg.inv(mtx) @ hand_homo_coor * 0.7
                    hand_world_coor = np.linalg.inv(pose[0:3, 0:3]) @ (hand_world_coor - pose[0:3, -1].reshape(3, 1))
                    hand_traj.append([x, y])

                    v_hand = np.linalg.norm((hand_world_coor[:2] - hand_world_coor_p[:2])) / (1 / 30)
                    hand_world_coor_p = hand_world_coor

            if len(missed) != 0:
                # for tag_id in missed:
                #     cur_obs = np.hstack((hand_world_coor[:2].reshape(-1), np.array(v_hand))).squeeze()
                #     tag_states[tag_id], error_cov[tag_id] = ekf_missed(cur_obs, tag_states[tag_id], 1 / 30, error_cov[tag_id])
                #     ekf_trajectories[tag_id].append(world_to_image_coordinates([tag_states[tag_id][0], tag_states[tag_id][1], 0], mtx, pose))
                dis_collection = []
                for tag_id in missed:  # 3, 5, 6
                    dis_collection.append(
                        bhattacharyya(tag_states[tag_id][:3], error_cov[tag_id][:3, :3], hand_world_coor.reshape(-1),
                                      obs_cov))
                # check the tag id blocked by hand
                missed = list(missed)
                blocked_index = missed[np.argmin(np.array(dis_collection))]
                print('tag ' + str(blocked_index) + ' has been blocked by hand')
                for tag_id in missed:
                    if tag_id == blocked_index:
                        # print('tag ' + str(blocked_index) + ' has been blocked by hand')
                        cur_obs = np.hstack((hand_world_coor[:2].reshape(-1), np.array(v_hand))).squeeze()
                        tag_states[blocked_index], error_cov[blocked_index] = kf(cur_obs, tag_states[blocked_index], 1 / 30, error_cov[blocked_index])
                        ekf_trajectories[blocked_index].append(world_to_image_coordinates([tag_states[tag_id][0], tag_states[tag_id][1], 0], mtx, pose))
                    else:
                        ekf_trajectories[tag_id].append(world_to_image_coordinates([tag_states[tag_id][0], tag_states[tag_id][1], 0], mtx, pose))

            plt.figure(figsize=(10, 6))

            for tag_id, trajectory in ekf_trajectories.items():
                for idx in range(len(trajectory) - 1):
                    ekf_last_frame_pixel = trajectory[idx]
                    ekf_cur_frame_pixel = trajectory[idx + 1]
                    cv2.line(frame, ekf_last_frame_pixel, ekf_cur_frame_pixel, (0, 255, 0), 2)

            for tag_id, trajectory in tag_trajectories.items():
                for idx in range(len(trajectory) - 1):
                    tag_last_frame_pixel = trajectory[idx]
                    tag_cur_frame_pixel = trajectory[idx + 1]
                    cv2.line(frame, tag_last_frame_pixel, tag_cur_frame_pixel, (136, 20, 8), 2)

            # for idx in range(len(hand_traj) - 1):
            #     hand_last_frame_pixel = hand_traj[idx]
            #     hand_cur_frame_pixel = hand_traj[idx + 1]
            #     cv2.line(frame, hand_last_frame_pixel, hand_cur_frame_pixel, (255, 255, 255), 2)

            # Display the video frame with the trajectories
            cv2.imshow('Trajectories', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
