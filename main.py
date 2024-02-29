import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import morphology, transform
import cv2
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import types
import random
import optuna
from argparse import ArgumentParser
# from skimage.filters import threshold_otsu
# from aicsimageio import AICSImage
# import pandas as pd

def main(
    level: int, 
    thresh: int,
    kernel_size: int,
    holes_thresh: int,
    scale_factor: int,
    padding: int,
    connect: int
):

    img_dir = Path('raw_data/IMGS')
    with open('raw_data/channelnames.txt', 'r') as file:
        channelnames = [line.strip() for line in file]

    #### OPTION PARAMETERS ####
    # level = 3 #downsample level - 3
    # thresh = 30 # needs to be percentage of max value 
    # kernel_size = 0 # 0
    # holes_thresh = 300 #300
    # scale_factor = 10 #10
    # padding = 20
    # connect = 2
    ###########################

    # Load images
    img_list = []
    img_list_sorted =  sorted(img_dir.glob('*.qptiff'))
    for img_path in img_list_sorted:
        img_list.append(tiff.TiffFile(img_path))

    img_arr = []
    for img in img_list:
        img_arr.append(img.series[0].levels[level].asarray())

    img_2D = sum_channels(img_arr)

    # plot_img_from_list(img_2D)

    # otsu for threshold for automation
    if 
    # thresh = threshold_otsu(img_2D[0])

    #downsample images
    img_2D_downsample = [transform.downscale_local_mean(img, (scale_factor,scale_factor)) for img in img_2D]

    # Thresholding by otsu
    # thresh_imgs = [threshold_otsu(img) for img in img_2D_downsample]
    # binary_imgs = [img > thresh_imgs[i] for i, img in enumerate(img_2D_downsample)]

    # convert to binary image by thresholding > 0 
    binary_imgs = [img > thresh for img in img_2D_downsample]
    binary_imgs = [(img * 255).astype(np.uint8) for img in binary_imgs]

    # plot_img_from_list(binary_imgs)

    # close images
    closed_imgs = [morphological_operation(img, kernel_size, 'closing') for img in binary_imgs]
    filled_imgs = [morphology.remove_small_holes(img, area_threshold=holes_thresh) for img in closed_imgs] 
    filled_imgs = [(img * 255).astype(np.uint8) for img in filled_imgs]

    # erode image
    eroded_imgs = [morphological_operation(img, kernel_size, 'erosion') for img in filled_imgs]

    # dilate image back
    dilated_imgs = [morphological_operation(img, kernel_size, 'dilation') for img in eroded_imgs]

    connected_comp_imgs = [cv2.connectedComponentsWithStats(img, connect, cv2.CV_32S) for img in dilated_imgs]

    filtered_imgs = [detect_tissues(img) for img in connected_comp_imgs]

    ### COLOR LABEL IMAGES ###
    # filtered_labels = [detect_tissues(img)[1] for img in connected_comp_imgs]
    # save_arrays_as_images(filtered_labels, use_colormap=True, output_folder='figures', file_prefix="labels", file_extension=".png")
    ##########################

    assert_same_length(filtered_imgs)

    # find biggest bounding box and mask
    tissue_bbox, centroid_slices, bbox_slices = find_biggest_bound_box(filtered_imgs)


    # crop images
    cropped_imgs = crop_imgs(img_arr, tissue_bbox, centroid_slices, scale_factor, padding, filtered_imgs)
    # cropped_imgs = crop_imgs([img_arr[4]], tissue_bbox, [centroid_slices[4]], scale_factor, padding, [filtered_imgs[4]])
    
    # stack images
    stacked_imgs = stack_images(cropped_imgs)

    aligned_tissue_list = []
    for img in stacked_imgs:

        # Normal alignment#
        aligned_tissue = align_z_slices(img)
        dice_list, d_avg = calculate_dice_coefficients(aligned_tissue)
        print(f"Average Dice coefficient: {d_avg}")
        plot_stack(aligned_tissue)

        aligned_tissue_list.append(aligned_tissue)

        # OPTUNA - hyperparameter tuning
        # optimize_sift_parameters(img)
        


        
    # save stack images
        
    pps = types.PhysicalPixelSizes(X=2.0, Y=2.0, Z=2.0)

    for i, img in enumerate(aligned_tissue):
        OmeTiffWriter.save(
            img,
            f"./outputs/aligned_3D_CODEX_59C_ZCYX_Tissue_{i}.ome.tif",
            dim_order="ZCYX",
            channel_names=channelnames,
            physical_pixel_sizes=pps,
        )



    # stack images index 0 - for testing and protyping
    # stacked_imgs_0 = np.stack((cropped_imgs[0][0], cropped_imgs[1][0], cropped_imgs[2][0], cropped_imgs[3][0], cropped_imgs[4][0]), axis=0)
    # stacked_imgs_0_DAPI = stacked_imgs_0[:, 0, :, :]

    # save stack images
    # tiff.imwrite('./outputs/stacked_imgs_0_DAPI.tif', stacked_imgs_0_DAPI)


    #physical_pixel_sizes
    # pps = types.PhysicalPixelSizes(X=16559.97, Y=30684.65, Z=1.0)

#     OmeTiffWriter.save(
#         stacked_imgs_0,
#         "./outputs/new_3D_CODEX_TEST_ZCYX.ome.tif",
#         dim_order="ZCYX",
#         channel_names=channelnames,
#         physical_pixel_sizes=pps,
# )
    

    
    print('DONE')


def align_z_slices(image_4d, reference_z=0, align_channel=0, params=None):
    """
    Align all z-slices in a 4D image array to a reference z-slice.

    :param image_4d: 4D image array with shape (z-slice, channels, height, width).
    :param reference_z: Index of the z-slice to use as the reference.
    :return: Aligned 4D image array.
    """
    # Initialize SIFT
    # Initialize SIFT with adjusted parameters

    if params:
        n_features = params['n_features']
        n_octave_layers = params['n_octave_layers']
        contrast_threshold = params['contrast_threshold']
        edge_threshold = params['edge_threshold']
        sigma = params['sigma']
        ratio_threshold = params['ratio_threshold']

    else:
        # n_features = 0  # 0 means no limit
        # n_octave_layers = 3  # Default value
        # contrast_threshold = 0.03  # Lowering this might result in more features being detected
        # edge_threshold = 10  # Default value
        # sigma = 1.6  # Default value
        # ratio_threshold = 0.75  # Lowe's ratio test

        #best params from optuna
        n_features = 600
        contrast_threshold = 0.07033053754007414
        edge_threshold = 15.243986448031832
        sigma = 1.7055546594751954
        n_octave_layers = 1
        ratio_threshold = 0.8961605990597056

    sift = cv2.SIFT_create(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma)

    # sift = cv2.SIFT_create()
    # Convert the reference z-slice to grayscale for feature detection
    # ref_slice = cv2.cvtColor(image_4d[reference_z].transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)

    #average or total sum of all channels
    # ref_slice = convert_to_grayscale_sum(image_4d[reference_z])
    # ref_slice = np.sum(image_4d[reference_z].transpose(1, 2, 0), axis=2)
    # ref_slice = np.mean(image_4d[reference_z], axis=0).astype(np.uint8)

    #one channel
    ref_slice = image_4d[reference_z, align_channel].astype(np.uint8)

    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_slice, None)

    # Initialize FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Initialize an empty array for the aligned z-slices
    aligned_image_4d = np.zeros_like(image_4d)

    # Align each z-slice to the reference
    for z in range(image_4d.shape[0]):
        if z != reference_z:
            # Convert z-slice to grayscale
            # current_slice = convert_to_grayscale_sum(image_4d[z])
            # current_slice = cv2.cvtColor(image_4d[z].transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
            # current_slice = np.mean(image_4d[z], axis=0).astype(np.uint8)
            # current_slice = np.sum(image_4d[z].transpose(1, 2, 0), axis=2)

            current_slice = image_4d[z, align_channel].astype(np.uint8)

            keypoints, descriptors = sift.detectAndCompute(current_slice, None)

            # Match descriptors
            matches = flann.knnMatch(descriptors_ref, descriptors, k=2)

            # Filter matches using Lowe's ratio test
            # good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

            # Find homography matrix
            src_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            try: 
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                aligned_slice = cv2.warpPerspective(image_4d[z].transpose(1, 2, 0), M, (image_4d.shape[3], image_4d.shape[2]))
            except Exception as e:
                print(e)
                return None
                

            # Apply the transformation to align the z-slice
            aligned_image_4d[z] = aligned_slice.transpose(2, 0, 1)
            # Update the keypoints and descriptors for the reference slice as the currrent aligned slice
            keypoints_ref, descriptors_ref = sift.detectAndCompute(aligned_image_4d[z, align_channel], None)

        else:
            # Copy the reference slice as-is
            aligned_image_4d[z] = image_4d[z]

    return aligned_image_4d

def generate_binary_mask(image):
    """
    Convert an image to a binary mask based on a threshold.
    
    :param image: Input image (2D array).
    :param threshold: Threshold value for binarization.
    :return: Binary mask of the image.
    """
    # _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

def objective(trial, image_4d):
    # Define the range of values for each parameter you want to optimize
    n_features = trial.suggest_int('n_features', 0, 1000)
    contrast_threshold = trial.suggest_float('contrast_threshold', 0.01, 0.1)
    edge_threshold = trial.suggest_float('edge_threshold', 10, 20)
    sigma = trial.suggest_float('sigma', 1.0, 2.0)
    n_octave_layers = trial.suggest_int('n_octave_layers', 1, 10)
    ratio_threshold = trial.suggest_float('ratio_threshold', 0.5, 0.9)

    #put parameters in a dictionary
    params = {
        'n_features': n_features,
        'n_octave_layers': n_octave_layers,
        'contrast_threshold': contrast_threshold,
        'edge_threshold': edge_threshold,
        'sigma': sigma,
        'ratio_threshold': ratio_threshold
    }

    # Initialize SIFT with suggested parameters
    # sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)

    # Perform alignment using the current SIFT configuration
    # Note: You'll need to modify your alignment function to accept the `sift` parameter
    aligned_image_4d = align_z_slices(image_4d, params=params)
    
    # Evaluate alignment quality
    _, average_dice = calculate_dice_coefficients(aligned_image_4d)
    
    # Since Optuna minimizes the objective, return a negative value of the metric if higher is better
    return average_dice

# Example usage
def optimize_sift_parameters(image_4d):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, image_4d), n_trials=100)  # Adjust n_trials to your preference

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

def calculate_dice_coefficients(aligned_image_4d, align_channel=0, threshold=127):
    """
    Calculate the Dice coefficient between consecutive slices in a 4D image array
    after converting each slice into a binary image.
    
    :param aligned_image_4d: 4D image array with shape (z-slice, channels, height, width).
    :param align_channel: The channel index to use for creating binary images.
    :param threshold: Threshold value for binarization.
    :return: List of Dice coefficients between consecutive slices.
    """

    if aligned_image_4d is None:
        return [], 0

    dice_coefficients = []
    
    # Convert the first slice to a binary mask and store as the previous slice's mask
    prev_slice_mask = generate_binary_mask(aligned_image_4d[0, align_channel])
    
    for z in range(1, aligned_image_4d.shape[0]):
        # Convert the current slice to a binary mask
        current_slice_mask = generate_binary_mask(aligned_image_4d[z, align_channel])
        
        # Calculate Dice coefficient
        intersection = np.logical_and(prev_slice_mask, current_slice_mask)
        dice = 2. * intersection.sum() / (prev_slice_mask.sum() + current_slice_mask.sum())
        
        dice_coefficients.append(dice)
        
        # Update the previous slice mask
        prev_slice_mask = current_slice_mask

    average_dice = np.mean(dice_coefficients) if dice_coefficients else 0
    
    return dice_coefficients, average_dice

def plot_stack(image_4d, align_channel=0):
    fig, axes = plt.subplots(1, image_4d.shape[0], figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(image_4d[i, align_channel], cmap='gray')
        ax.set_title(f'Slice {i+1}')
    plt.tight_layout()
    plt.show()

def convert_to_grayscale_sum(slice_img):
    # Sum the channels
    summed_img = np.sum(slice_img, axis=0)
    # Normalize to the 8-bit range and convert to uint8
    normalized_img = cv2.normalize(summed_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_img

def stack_images(list_of_lists_of_images):
    # Determine the number of images per inner list (assuming all inner lists are of the same length)
    num_images = len(list_of_lists_of_images[0])

    tissue_imgs = [[list_of_lists_of_images[i][j] for i in range(len(list_of_lists_of_images))] for j in range(num_images)]

    #assert that all arrays in a list are the same size
    idx_mismatch = check_inner_lists_sizes(tissue_imgs)

    # Pad images to the same size if needed
    if idx_mismatch is not None:
        for i in idx_mismatch:
            tissue_imgs[i] = pad_images_to_max_size(tissue_imgs[i])
        
    # Stack images of the same index from each inner list
    # stacked_images = [np.stack([inner_list[i] for inner_list in list_of_lists_of_images], axis=0) for i in range(num_images)]
    stacked_images = [np.stack(tissue_imgs[i], axis=0) for i in range(num_images-3)]

    return stacked_images

def check_inner_lists_sizes(list_of_lists):
    """
    Check if all arrays in each inner list of a list have the same size. 

    :param list_of_lists: List of lists, where each inner list contains numpy arrays.
    :return: List of indices of inner lists with mismatched sizes, or None if all inner lists have arrays of the same size.
    """

    idx_list = []
    
    for idx, inner_list in enumerate(list_of_lists):
        if inner_list:  # Check if the inner list is not empty
            first_shape = inner_list[0].shape
            for array in inner_list[1:]:
                if array.shape != first_shape:
                    idx_list.append(idx)  # Return the index of the inner list with mismatched sizes
    
    if idx_list:
        return idx_list
    else:
        return None  # All inner lists have arrays of the same size

def pad_image(image, max_width, max_height, pad_value=0):
    height_pad = (max_height - image.shape[1], 0)  # Pad only before, none after
    width_pad = (max_width - image.shape[2], 0)  # Pad only before, none after
    padding = [(0, 0), height_pad, width_pad]  # No padding for channels
    return np.pad(image, padding, mode='constant', constant_values=pad_value)

def pad_images_to_max_size(image_list, pad_value=0):
    max_width, max_height = find_max_dimensions(image_list)
    return [pad_image(image, max_width, max_height, pad_value) for image in image_list]


def find_max_dimensions(images):
    max_width = max(image.shape[2] for image in images)
    max_height = max(image.shape[1] for image in images)
    return max_width, max_height


def close_blob(binary_image, kernel_size=20):
    """
    Close holes and connect components in a binary image.

    :param binary_image: Input binary image (numpy array) where objects are white and the background is black.
    :param kernel_size: Size of the structuring element used for closing.
    :return: The closed binary image.
    """
    
    # Create the structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply closing (dilation followed by erosion)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return closed_image

def upsample_image(image, upsample_factor):
    """
    Upsample the image by the specified factor.

    :param image: Input image to upsample.
    :param upsample_factor: Factor to upsample the image by.
    :return: Upsampled image.
    """
    # Calculate the new dimensions
    new_height = int(image.shape[0] * upsample_factor)
    new_width = int(image.shape[1] * upsample_factor)

    # Upsample the image using cubic interpolation
    upsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return upsampled_image

def crop_imgs(imgs, bbox, centroids, sf, padding, filtered_imgs):
    
    cropped_slices = []

    for i, img in enumerate(imgs):
        cropped_imgs = []
        for j in range(len(bbox)):
            w, h = bbox[j]
            cX, cY = centroids[i][j]
            # x, y, _, _ = bbox_slices[i][j]
            x1, y1, x2, y2 = create_bounding_box(cX, cY, w, h, img, sf, padding)

            #use upsample mask on original image
            upsampled_mask = upsample_image(filtered_imgs[i][0][j], sf)
            upsampled_mask = (upsampled_mask > 0).astype(np.uint8) #convert to [0, 1]

            #apply mask to original image
            mask_img = img * upsampled_mask

            #crop image
            new_img = mask_img[:, y1:y2, x1:x2]
            cropped_imgs.append(new_img)
            # cropped_imgs.append(img[y:y+h, x:x+w])

        cropped_slices.append(cropped_imgs) 

    
    return cropped_slices

def create_bounding_box(cX, cY, width, height, img, sf, padding):
    x1 = cX - width // 2
    y1 = cY - height // 2
    x2 = cX + width // 2
    y2 = cY + height // 2

    # Adjust if bounding box goes beyond image boundaries (assuming image dimensions are known)
    x1 = max(x1, 0) * sf
    y1 = max(y1, 0) * sf
    x2 = min(x2, img.shape[2]) * sf
    y2 = min(y2, img.shape[1]) * sf

    #create padding
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, img.shape[2])
    y2 = min(y2 + padding, img.shape[1])

    return x1, y1, x2, y2

def find_biggest_bound_box(slice_imgs):

    # d = {}

    # Compute bounding boxes and centroids for all slices
    bounding_box_slices = [compute_bounding_box(slice)[0] for slice in slice_imgs]
    centroids_slices = [compute_bounding_box(slice)[1] for slice in slice_imgs]

    # Ensure there are bounding boxes in the slices
    if not bounding_box_slices or not all(bounding_box_slices):
        return [], []

    # Find the maximum number of bounding boxes in any slice
    max_bounding_boxes = max(len(bboxes) for bboxes in bounding_box_slices)

    biggest_bounding_boxes = []
    # centroid_of_biggest = []
    biggest_mask = []

    # Iterate through each index of bounding boxes across slices
    for i in range(max_bounding_boxes):
        max_area = 0
        # biggest_bbox = None
        # biggest_centroid = None

        # Iterate through each slice
        for slice_index in range(len(slice_imgs)):
            if i < len(bounding_box_slices[slice_index]):
                x, y, w, h = bounding_box_slices[slice_index][i]
                area = w * h

                # Update if this bounding box is bigger
                if area > max_area:
                    max_area = area
                    _, _, w_best, h_best = bounding_box_slices[slice_index][i]

                    # best_mask = slice_imgs[slice_index][0][i]
                    # biggest_centroid = centroids_slices[slice_index][i]

                    #close blob
                    # best_mask = close_blob(best_mask)

        biggest_bounding_boxes.append((w_best, h_best))
        # centroid_of_biggest.append(biggest_centroid)
        # biggest_mask.append(best_mask)

        # d[i] = biggest_bbox
    

    return biggest_bounding_boxes, centroids_slices, bounding_box_slices


def assert_same_length(list_of_lists):
    assert all(len(inner_list) == len(list_of_lists[0]) for inner_list in list_of_lists), "Not all lists are of the same length."

def compute_bounding_box(imgs):

    bounding_box = []
    centroids = []

    #get the individual tissues
    for img in imgs[0]:
        # Find contours in the image
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming there is only one blob in the image
        if len(contours) > 0:
            # Get the first contour
            cnt = contours[0]

            # Compute the centroid of the blob
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0  # Set centroid to origin if the area is zero

            # Compute the bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Add padding to the bounding box
            # x_padded = max(x - padding, 0)
            # y_padded = max(y - padding, 0)
            # w_padded = min(w + 2 * padding, img.shape[1] - x_padded)
            # h_padded = min(h + 2 * padding, img.shape[0] - y_padded)
            
            #crop image
            # img = img[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
            # Draw the bounding box and centroid on the image (for visualization)
            # output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.circle(output_image, (cX, cY), 5, (0, 0, 255), -1)

            # Print the centroid coordinates (we'll use the center of the bounding box as the centroid)
            print("Centroid coordinates: ({}, {})".format(cX, cY))

            # bounding_box.append((x_padded, y_padded, w_padded, h_padded))
            bounding_box.append((x, y, w, h))
            centroids.append((cX, cY))

    return bounding_box, centroids

def detect_tissues(cc_img, num_tissue=8):

    iso_tissue = []

    num_labels = cc_img[0]  # The first cell is the number of labels
    labels = cc_img[1]  # The second cell is the label matrix itself
    stats = cc_img[2]  # The third cell is the stat matrix

    #remove island and small blobs
    # new_labels = remove_islands_and_small_blobs(labels)


    nunique = np.unique(labels)
    sort_stats = stats[:, cv2.CC_STAT_AREA].copy()
    sort_stats.sort()
    # Filter based on size
    min_size = sort_stats[-num_tissue - 1]  # Set your minimum size threshold

    tissue_values = nunique[stats[:, cv2.CC_STAT_AREA] >= min_size]


    # Create a new image that will be the filtered result
    new_labels = filter_array(labels, tissue_values)

    #renumber labels to be sequential
    new_values = np.arange(0, len(tissue_values))

    for new, old in zip(new_values, tissue_values):
        new_labels[new_labels == old] = new

    for i in new_values[1:]:  # Skip the background label 0
        large_blobs = np.zeros_like(labels, dtype=np.uint8)
        large_blobs[new_labels == i] = 255
        iso_tissue.append(large_blobs)
    
    return iso_tissue, new_labels

    # return new_labels

def filter_array(arr, values):
    """
    Set elements in 'arr' to 0 if they are not in 'values'.

    :param arr: Input 2D numpy array.
    :param values: List of values to keep.
    :return: Filtered 2D numpy array.
    """
    # Convert the list to a numpy array for efficient processing
    values_arr = np.array(values)

    # Create a boolean mask where True indicates the value should be kept
    mask = np.isin(arr, values_arr)

    # Apply the mask and set non-matching elements to 0
    filtered_arr = np.where(mask, arr, 0)

    return filtered_arr

def remove_islands_and_small_blobs(image, noise_reduction_size=3, blob_size=3):
    """
    Remove small isolated pixels and blobs from the image.

    :param image: Input binary image (should be a binary image with values 0 and 255).
    :param noise_reduction_size: Kernel size for median filter for noise reduction.
    :param blob_size: Kernel size for morphological opening to remove small blobs.
    :return: Processed image.
    """

    # Ensure the image is in the correct format (8-bit)
    if image.dtype != np.uint8:
        # Normalize and convert to 8-bit format
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Check if image is single-channel; if not, convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median filter to remove noise
    denoised = cv2.medianBlur(image, noise_reduction_size)

    # Define the structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blob_size, blob_size))

    # Apply morphological opening to remove small objects
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    return opened


def morphological_operation(img, kernel_size, operation):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilation':
        return cv2.dilate(img, kernel, iterations=1)
    elif operation == 'erosion':
        return cv2.erode(img, kernel, iterations=1)
    else:
        return img

def sum_channels(img_arr):
    
    return [np.sum(img, axis=0, dtype=np.uint16) for img in img_arr]

def plot_img_from_list(img_list):
    fig, ax = plt.subplots(1, len(img_list))
    for i, img in enumerate(img_list):
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def save_arrays_as_images(arrays, use_colormap=False, output_folder='figures', file_prefix="image", file_extension=".png"):
    """
    Save each 2D NumPy array in a list as an image file. Optionally use a colormap to assign unique colors to each value,
    while keeping 0 values as black. The same color is used for the same value across all images.

    :param arrays: List of 2D NumPy arrays.
    :param use_colormap: Boolean to turn on/off the colormap.
    :param output_folder: Folder where images will be saved.
    :param file_prefix: Prefix for the output filenames.
    :param file_extension: Extension for the output files.
    """

    # If the input is a single array, wrap it in a list
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    # Generate a consistent colormap for all arrays if needed
    if use_colormap:
        all_unique_values = set(np.unique(arrays[0]))
        color_map = {0: (0, 0, 0)}  # Assign black to 0
        for value in all_unique_values:
            if value != 0:
                color_map[value] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for i, array in enumerate(arrays):
        if use_colormap:
            # Create a colored image based on the color map
            colored_image = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
            for value, color in color_map.items():
                colored_image[array == value] = color
            image_to_save = colored_image
        else:
            # Normalize and convert to 8-bit grayscale image
            image_to_save = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
            if image_to_save.dtype != np.uint8:
                image_to_save = image_to_save.astype(np.uint8)

        # Construct the filename
        filename = f"{output_folder}/{file_prefix}{i}{file_extension}"

        # Save the image
        cv2.imwrite(filename, image_to_save)
        print(f"Saved {filename}")

if __name__ == "__main__":

    # optional parameteres - otsu and threshold of max (percentage)
    # downsample parameter 

    # each piece as seperate files and upsampled and downsampled versions

    # make debug of steps - save images of each step.
    p = ArgumentParser()
    p.add_argument('--level', type=int, default=0, help='Pyrmaid level of the image, default is 0 which is the original image size')
    p.add_argument('--thresh', type=int, default=None, help='Threshold value for binarization, default is done by otsu')
    p.add_argument('--kernel_size', type=int, default=0, help='Size of the structuring element used for closing, default is 0')
    p.add_argument('--holes_thresh', type=int, default=300, help='Area threshold for removing small holes, default is 300')
    p.add_argument('--scale_factor', type=int, default=10, help='Scale factor for downsample, default is 10')
    p.add_argument('--padding', type=int, default=20, help='Padding for bounding box, default is 20')
    p.add_argument('--connect', type=int, default=2, help='Connectivity for connected components, default is 2')
    
    args = p.parse_args()

    main(
        level = args.level,
        thresh = args.thresh,
        kernel_size = args.kernel_size,
        holes_thresh = args.holes_thresh,
        scale_factor = args.scale_factor,
        padding = args.padding,
        connect = args.connect
    )