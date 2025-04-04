import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate frame interval based on the specified frame rate
    frame_interval = int(fps * frame_rate)
    
    # Initialize frame counter
    count = 0
    frame_count = 0
    
    # Read until video is completed
    while True:
        ret, frame = video.read()
        
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        
        # Save frame at the specified interval
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Extracted: {frame_filename}")
            frame_count += 1
        
        count += 1
    
    # Release the video capture object
    video.release()
    
    print(f"Extraction complete. {frame_count} frames extracted to {output_dir}")

def get_file_paths():
    file_paths = []

    for root, dirs, files in os.walk("videos"):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

if __name__ == "__main__":
    file_paths = get_file_paths()

    for file_path in file_paths:
        extract_frames(file_path, os.path.join("frames", os.path.basename(file_path)))