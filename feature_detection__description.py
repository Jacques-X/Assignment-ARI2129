import cv2
import os
import time
import matplotlib.pyplot as plt

#feature detection using ORB, SIFT, and BRISK
def load_images_from_all_videos(base_folder, max_images_per_video=10):
    all_images = []
    video_folders = sorted(os.listdir(base_folder))

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for video_folder in video_folders:
        full_path = os.path.join(base_folder, video_folder)
        if not os.path.isdir(full_path):
            continue  # skip non-folder entries

        image_filenames = sorted(os.listdir(full_path))
        image_files = [f for f in image_filenames if os.path.splitext(f)[1].lower() in image_extensions]

        for filename in image_files[:max_images_per_video]:
            img_path = os.path.join(full_path, filename)
            img_color = cv2.imread(img_path)
            if img_color is None:
                print(f"Warning: Couldn't read {img_path}")
                continue
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            all_images.append((video_folder + "/" + filename, img_color, img_gray))

    return all_images

#detect features using ORB, SIFT, and BRISK
def detect_features(detector, images, use_gray=True):
    results = []
    for filename, img_color, img_gray in images:
        img = img_gray if use_gray else img_color
        start_time = time.time()
        keypoints, descriptors = detector.detectAndCompute(img, None)
        time_taken = time.time() - start_time
        results.append({
            'filename': filename,
            'image': img_color,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'time': time_taken
        })
    return results

#draw keypoints on images
def draw_keypoints(results, alg_name, output_dir="output_keypoints"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for res in results:
        img_with_kp = cv2.drawKeypoints(res['image'], res['keypoints'], None, color=(0, 255, 0))
        out_path = os.path.join(output_dir, f"{alg_name}_{res['filename'].replace('/', '_')}")
        cv2.imwrite(out_path, img_with_kp)

#analyze results
def analysis(results, alg_name):
    print(f"\n - - {alg_name} - -")
    total_kps = 0
    total_time = 0

    #print summary of results
    for res in results:
        print(f"{res['filename']}: {len(res['keypoints'])} keypoints in {res['time']:.4f}s")
        total_kps += len(res['keypoints'])
        total_time += res['time']
    if len(results) > 0:
        avg_kps = total_kps / len(results)
        avg_time = total_time / len(results)
        print(f"Average keypoints: {avg_kps:.2f} | Average Time: {avg_time:.2f}s")
    else:
        print("No images were processed.")

def main():
    frame_dir = "frames"  #path to frames
    images = load_images_from_all_videos(frame_dir, max_images_per_video=10)

    if not images:
        print("No images found in any of the video subfolders.")
        return

    #initialize detectors
    orb = cv2.ORB_create(nfeatures=1000)
    sift = cv2.SIFT_create()
    brisk = cv2.BRISK_create()

    #detect features
    orb_results = detect_features(orb, images)
    sift_results = detect_features(sift, images)
    brisk_results = detect_features(brisk, images)

    #save keypoint visualizations
    draw_keypoints(orb_results, "ORB")
    draw_keypoints(sift_results, "SIFT")
    draw_keypoints(brisk_results, "BRISK")

    #print comparison summary
    analysis(orb_results, "ORB")
    analysis(sift_results, "SIFT")
    analysis(brisk_results, "BRISK")

if __name__ == "__main__":
    main()
