import cv2
import numpy as np
from typing import Tuple

def read_videos(video_file_path):
    camera = cv2.VideoCapture(video_file_path)
    i = 1
    while i < int(camera.get(cv2.CAP_PROP_FRAME_COUNT)):
        _, frame = camera.read()
        cv2.imwrite(
            "/Users/yuhaoyou/PycharmProjects/pythonProject1/IMG_5941.MOV" + str(
                i) + '.jpg', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imshow('frame', frame)
        i += 10
        if cv2.waitKey(200) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def world_to_image_coordinates(world_coordinates: np.ndarray, camera_matrix: np.ndarray, pose: np.ndarray) -> Tuple[
    int, int]:
    # Multiply the world coordinates by the camera matrix and pose
    helper = pose[:3, :3] @ world_coordinates
    image_coordinates = camera_matrix @ (pose[:3, :3] @ world_coordinates + pose[:3, -1])
    # Divide by the third coordinate to obtain homogeneous coordinates
    image_coordinates /= image_coordinates[2]

    return [int(image_coordinates[0]), int(image_coordinates[1])]

def update_3d_plot(trajectories, ax):
    ax.clear()

    for tag_id, trajectory in trajectories.items():
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o', linestyle='-', label=f'Tag {tag_id}')

    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_zlabel('Z-coordinate')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.4, 0.4)

    ax.legend()