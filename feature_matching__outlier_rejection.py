import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#load images from all videos
def load_image_pairs(base_folder, step=1, max_pairs_per_video=3):
    pairs = []
    video_folders = sorted(os.listdir(base_folder))

    #filter out non-directory entries
    for video_folder in video_folders:
        folder_path = os.path.join(base_folder, video_folder)
        if not os.path.isdir(folder_path):
            continue

        filenames = sorted([f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))])
        num_pairs = min(len(filenames) - step, max_pairs_per_video)

        for i in range(num_pairs):
            img1_path = os.path.join(folder_path, filenames[i])
            img2_path = os.path.join(folder_path, filenames[i + step])

            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            if img1 is not None and img2 is not None:
                pairs.append((f"{video_folder}/{filenames[i]}", f"{video_folder}/{filenames[i+step]}", img1, img2))
    return pairs

#get feature detector
def get_detector(name="SIFT"):
    if name == "ORB":
        return cv2.ORB_create(nfeatures=1000)
    elif name == "BRISK":
        return cv2.BRISK_create()
    else:
        return cv2.SIFT_create()

#get matcher based on descriptor type
def get_matcher(desc_type):
    if desc_type == "float":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#match features between two images
def match_features(detector, img1, img2):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    #check descriptor type
    desc_type = "float" if des1.dtype == np.float32 else "binary"
    matcher = get_matcher(desc_type)

    if desc_type == "float":
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    else:
        matches = matcher.match(des1, des2)
        good_matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, good_matches

#ransac filter to remove outliers
def ransac_filter(kp1, kp2, matches):
    if len(matches) < 8:
        return [], None

    #extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)
    inliers = [m for i, m in enumerate(matches) if mask[i]]

    return inliers, F

#draw matches between two images
def draw_matches(img1, kp1, img2, kp2, matches, title, out_folder="output_matches", filename_prefix="match"):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    #draw matches
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    #save figure with title in filename
    safe_title = title.replace(" ", "_").lower()
    out_path = os.path.join(out_folder, f"{filename_prefix}_{safe_title}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    #set parameters
    frame_dir = "frames"
    detector_name = "SIFT"
    detector = get_detector(detector_name)

    pairs = load_image_pairs(frame_dir, step=1, max_pairs_per_video=3)

    #check if pairs are loaded
    for fname1, fname2, img1, img2 in pairs:
        print(f"\nProcessing pair: {fname1} & {fname2}")
        kp1, kp2, matches = match_features(detector, img1, img2)

        print(f"Total matches before RANSAC: {len(matches)}")
        draw_matches(img1, kp1, img2, kp2, matches, title="Before RANSAC", filename_prefix=f"{fname1.replace('/', '_')}_vs_{fname2.replace('/', '_')}")

        inliers, F = ransac_filter(kp1, kp2, matches)
        print(f"Matches after RANSAC: {len(inliers)}")
        draw_matches(img1, kp1, img2, kp2, inliers, title="After RANSAC", filename_prefix=f"{fname1.replace('/', '_')}_vs_{fname2.replace('/', '_')}")

if __name__ == "__main__":
    main()
