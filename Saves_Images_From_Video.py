import cv2
import os

# User parameters
VIDEOS_PATHS = "Saved_Videos" # Change this to the path of your video files
VIDEO_PATH = "Saved_Videos/2022_11_09-16_27_15-Human.mp4"  # Change this to the path of your video file
OUTPUT_FOLDER = "Screenshot_Images"  # Change this to the desired output folder
FRAMES_PER_SECOND = 1  # Change this to control how many frames to save per second

def save_frames(video_path, output_folder, frames_per_second, video_index):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frame_interval = int(fps / frames_per_second)
    if frame_interval < 1:
        frame_interval = 1

    for frame_number in range(0, total_frames, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if not ret:
            break

        output_filename = os.path.join(output_folder, f"Vid_{video_index}-Frame_{frame_number:04d}.jpg")
        cv2.imwrite(output_filename, frame)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = VIDEO_PATH  # Change this to the path of your video file
    output_folder = OUTPUT_FOLDER  # Change this to the desired output folder
    frames_per_second = FRAMES_PER_SECOND  # Change this to control how many frames to save per second

    # save_frames(video_path, output_folder, frames_per_second, 0)

    for video_index, video_name in enumerate(os.listdir(VIDEOS_PATHS)):
        video_path = os.path.join(VIDEOS_PATHS, video_name)
        save_frames(video_path, output_folder, frames_per_second, video_index)
