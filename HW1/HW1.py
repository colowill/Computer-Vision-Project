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
    img1_height, img1_width = img1_dict['clean_rgb'].shape[:2]
    new_width = img1_width * 2  

    panorama = cv2.warpPerspective(img2_dict['clean_rgb'], homography_matrix, (new_width, img1_height))
    
    # Overlay img2 onto the left side of the canvas
    panorama[0:img1_height, 0:img1_width] = img1_dict['clean_rgb']
    
    return panorama


step1_images = step1(yosemite_image_paths)

## Display original Images
# display_images(yosemite_image_paths)
## Display images after applying SIFT in Step 1
# display_images(step1_images)

step2_image1 = step1_images[0]
step2_image2 = step1_images[1]

step2_images = step2(step1_images[0], step1_images[1])

display_images([step2_images])