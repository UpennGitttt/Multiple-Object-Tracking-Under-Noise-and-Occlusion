import cv2

video_file = '/Users/haihui_gao/Documents/my_scripts/vedio/IMG_7664.MOV'
frame_interval = 50
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_number in range(0, total_frames, frame_interval):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read the frame {frame_number}.")
        continue

    output_image_file = f'output_frame_{frame_number}.jpg'
    cv2.imwrite(output_image_file, frame)

cap.release()
cv2.destroyAllWindows()
