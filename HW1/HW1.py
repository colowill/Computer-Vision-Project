import cv2
import matplotlib.pyplot as plt
import numpy as np

## Step 1: Feature Detection ##
PATH_PREFIX = 'HW1/images/'

yosemite_image_paths = [
    f'{PATH_PREFIX}yosemite1.jpg',
    f'{PATH_PREFIX}yosemite2.jpg',
    f'{PATH_PREFIX}yosemite3.jpg',
    f'{PATH_PREFIX}yosemite4.jpg'
]

def display_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
    for ax, item in zip(axes, images):
        # Handle path strings
        if isinstance(item, str):
            img = cv2.imread(item)
            if img is None:
                raise FileNotFoundError(item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Handle step1 dictionary output
        elif isinstance(item, dict):
            img = item['rgb']
        # Handle raw image arrays
        else:
            img = item      
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def step1(path_array):
    """
    Returns an array of images with their features detected
    """
    processed_images = []
    
    # Using SIFT to detect features
    sift = cv2.SIFT_create()

    for path in path_array:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)

        h, w = img.shape[:2]
        aspect_ratio = w / h  # Corrected to width / height
        target_height = 480
        target_width = int(target_height * aspect_ratio)

        # Resizing Image
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Corrected to BGR2GRAY
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        keypoints, features = sift.detectAndCompute(img_gray, None)

        # Draw the detected keypoints directly onto the RGB image
        img_sift = cv2.drawKeypoints(
            img_rgb, 
            keypoints, 
            None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        processed_images.append({
            'rgb': img_sift,
            'clean_rgb': img_rgb,
            'gray': img_gray,
            'keypoints': keypoints,
            'features': features
        })
        
    return processed_images


def step2(img1_dict, img2_dict):
    """
    Takes two images and roughly stitches them together by detecting where there features match

    Originally, step2 hard-pasted img1 directly over warped img2 into a single result image, 
    which lost the individual image boundary data needed for blending. It now returns two separate, 
    identical-sized canvases (canvas1 and canvas2) so step3 can compute independent distance weights and create a seamless overlap.
    """
    
    img1_pts = []
    img2_pts = []
    
    # Brute force match object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Compute matches between the features of the two images
    matches = bf.match(img1_dict['features'], img2_dict['features'])
    
    # Sort them in order by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    for match in matches:
        
        # Sorting through the matches between each image, index1 is the index of keypoints from first image 
        # that corresponds to index2 of keypoints in the second image
        
        img1_index = match.queryIdx
        img2_index = match.trainIdx
        
        # Setting the actual x,y coord from each image of the keypoint match
        img1_coord = img1_dict['keypoints'][img1_index].pt
        img2_coord = img2_dict['keypoints'][img2_index].pt
        
        img1_pts.append(img1_coord)
        img2_pts.append(img2_coord)
    
    pts1_np = np.float32(img1_pts)
    pts2_np = np.float32(img2_pts)

    # Set homography matrix using RANSAC to reject outliers
    homography_matrix, mask = cv2.findHomography(pts2_np, pts1_np, cv2.RANSAC, 5.0)
    
    # Set new image width
    h1, w1 = img1_dict['clean_rgb'].shape[:2]
    h2, w2 = img2_dict['clean_rgb'].shape[:2]    
    new_width = w1 + w2
    new_height = max(h1, h2)
    
    # Warp img1 onto img2
    canvas_img1 = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    canvas_img1[0:h1, 0:w1] = img1_dict['clean_rgb']
    
    # Warp img2 onto its own canvas of size (new_height, new_width)
    canvas_img2 = cv2.warpPerspective(img2_dict['clean_rgb'], homography_matrix, (new_width, new_height))
        
    # MODIFIED: Returned both separate canvases so step3 can compute distance transform weight maps
    return canvas_img1, canvas_img2

post_step1_images = step1(yosemite_image_paths)

## Display original Images
# display_images(yosemite_image_paths)
## Display images after applying SIFT in Step 1
# display_images(post_step1_images)

post_step2_canvas1, post_step2_canvas2 = step2(post_step1_images[0], post_step1_images[1])

display_images([post_step2_canvas1])
display_images([post_step2_canvas2])

def step3(base_canvas, warped_canvas):
    """
    Blends two aligned canvases using distance transform weights to create a seamless stitch
    """
    def get_distance_transform(img_rgb):
        thresh = cv2.threshold(img_rgb, 0, 255, cv2.THRESH_BINARY)[1]
        thresh = thresh.any(axis=2)
        thresh = np.pad(thresh, 1)
        thresh = thresh.astype(np.uint8) * 255
        
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)[1:-1, 1:-1]
        dist = dist[:, :, None]
        
        max_val = dist.max()
        if max_val == 0:
            return dist
        return dist / max_val * 255.0

    # 1. Convert canvases to float32 for blending calculations
    c1_f = base_canvas.astype(np.float32)
    c2_f = warped_canvas.astype(np.float32)

    # 2. Compute distance weight maps using the inner function
    w1 = get_distance_transform(base_canvas)
    w2 = get_distance_transform(warped_canvas)

    # 3. Calculate total weight map and prevent division by zero
    denom = np.maximum(w1 + w2, 1.0)

    # 4. Blend using weighted average: (img1 * w1 + img2 * w2) / (w1 + w2)
    blended = (c1_f * w1 + c2_f * w2) / denom

    # 5. Clip pixel values to [0, 255] and return as uint8
    return np.clip(blended, 0, 255).astype(np.uint8)

blended_panorama = step3(post_step2_canvas1, post_step2_canvas2)
display_images([blended_panorama])


def final_panorama(image_array):
    
    