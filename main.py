import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import morphology, transform
import cv2
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import types
import random
from argparse import ArgumentParser
# from skimage.filters import threshold_otsu
# from aicsimageio import AICSImage
# import pandas as pd
import optuna
import time
# import torch
from skimage.filters import threshold_multiotsu
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from numpy.fft import fft
from scipy import stats
import json
import os

# CONST_PIXEL_SIZE_FOR_OPERATIONS = 0.5073519424785282 * 10 #microns
CONST_PIXEL_SIZE_FOR_OPERATIONS = 4.058815539828226

def main(
    num_tissue: int,
    level: int, 
    thresh: int,
    kernel_size: int,
    holes_thresh: int,
    scale_factor: int,
    align_upsample_factor: int,
    padding: int,
    connect: int, 
    pixel_size: list,
    output_folder: str,
    input_path: str,
    file_basename: str,
    optimize: bool,
    crop_only: bool
):

    start_begin = time.time()
    input_path = Path(input_path)
    img_list = []

    if os.path.isdir(input_path):
        img_dir = input_path

        # Load images
        img_list_sorted =  sorted(img_dir.glob('*.qptiff'), key=lambda x: x.name[:3].lower())
        for img_path in img_list_sorted:
            img_list.append(tiff.TiffFile(img_path))
    elif os.path.isfile(input_path):
        img_list.append(tiff.TiffFile(input_path))
    else:
        print(f"The path {input_path} does not exist.")
        exit()
        

    # in hive
    # img_dir = Path('raw_data')
    # local
    # img_dir = Path('raw_data/IMGS')


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
        
    # pixel_size = [0.5073519424785282, 0.5073519424785282] #microns
    if not pixel_size[1] == 4.058815539828226 and not pixel_size[0] == 4.058815539828226:
        scale_factor_x = int(CONST_PIXEL_SIZE_FOR_OPERATIONS / pixel_size[0])
        scale_factor_y = int(CONST_PIXEL_SIZE_FOR_OPERATIONS / pixel_size[1])
    else:
        scale_factor_x = scale_factor
        scale_factor_y = scale_factor

    pps = types.PhysicalPixelSizes(X=pixel_size[0], Y=pixel_size[1], Z=2.0)
    ###########################

    print('Starting...Read images')
    #time the process
    start = time.time()
    # Load images
    # img_list = []
    # img_list_sorted =  sorted(img_dir.glob('*.qptiff'), key=lambda x: x.name[:3].lower())
    # for img_path in img_list_sorted:
    #     img_list.append(tiff.TiffFile(img_path))

    img_arr = []
    for img in img_list:
        img_arr.append(img.series[0].levels[level].asarray())

    #downsample images - can replace orgining with this if we want to conserve memory
    img_arr_downsample = [transform.downscale_local_mean(img, (1, scale_factor_x,scale_factor_y)) for img in img_arr]

    img_2D = sum_channels(img_arr_downsample)
        
    # img_2D = sum_channels(img_arr)
    print('Time to read images + Downsampling + Summing all channels:', time.time() - start)

    # plot_img_from_list(img_2D)

    print('Preprocessing images...Thresholding, Downsample, Closing, Filling Holes, Erosion, Dilation, Connected Components')
    # time the process
    start = time.time()
    # otsu for threshold for automation
    if thresh == None:
        # thresh = [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] for img in img_2D]
        thresh = [threshold_multiotsu(img, classes=3)[0] / 2 for img in img_2D]
    else:
        thresh = [thresh for _ in img_2D]

    #downsample images
    # img_2D_downsample = [transform.downscale_local_mean(img, (scale_factor,scale_factor)) for img in img_2D]




    # convert to binary image by thresholding > 0 
    binary_imgs = [img > thresh[i] for i, img in enumerate(img_2D)]
    # binary_imgs = [img > thresh[i] for i, img in enumerate(img_2D_downsample)]
    binary_imgs = [(img * 255).astype(np.uint8) for img in binary_imgs]

    # plot_img_from_list(binary_imgs)

    # close images
    closed_imgs = [morphological_operation(img, kernel_size, 'closing') for img in binary_imgs]

    #erosion to prevent connected components from merging
    eroded_imgs = [morphological_operation(img, kernel_size, 'erosion') for img in closed_imgs]

    # Dilate the image
    dilated_imgs = [morphological_operation(img, int(kernel_size/10), 'dilation', its=2) for img in eroded_imgs]

    # Erode the dilated image
    eroded_imgs_2 = [morphological_operation(img, int(kernel_size/10), 'erode', its=2) for img in dilated_imgs]

    # filled_imgs = [morphology.remove_small_holes(img, area_threshold=holes_thresh) for img in eroded_imgs] 
    # filled_imgs = [(img * 255).astype(np.uint8) for img in filled_imgs]
    # filled_imgs = [morphology.remove_small_holes(img, area_threshold=holes_thresh*2) for img in closed_imgs] 
    # filled_imgs = [(img * 255).astype(np.uint8) for img in filled_imgs]

    # dilate image then close
    # dilated_imgs_2 = [morphological_operation(img, kernel_size, 'dilation') for img in eroded_imgs]
    # closed_imgs_2 = [morphological_operation(img, kernel_size, 'closing') for img in dilated_imgs_2]

    # image opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(kernel_size * 0.3), int(kernel_size * 0.3)))
    opened_imgs = [cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) for img in eroded_imgs_2]

    # dilate image then close
    # dilated_imgs_2 = [morphological_operation(img, 0, 'dilation') for img in opened_imgs]
    # eroded_imgs_3 = [morphological_operation(img, 0, 'erosion') for img in dilated_imgs_2]
    # dilated_imgs_3 = [morphological_operation(img, 0, 'dilation') for img in eroded_imgs_3]
    closed_imgs_2 = [morphological_operation(img, int(kernel_size * 2), 'closing') for img in opened_imgs]

    # remove holes again
    # filled_imgs_2 = [morphology.remove_small_holes(img, area_threshold=holes_thresh) for img in closed_imgs_2]
    # filled_imgs_2 = [(img * 255).astype(np.uint8) for img in filled_imgs_2]

    # # erode image to remove connected tissues that may have formed
    # eroded_imgs_3 = [morphological_operation(img, int(kernel_size / 10), 'erosion', its=2) for img in closed_imgs_2]

    # # # dilate image back
    processed_imgs = [morphological_operation(img, kernel_size, 'dilation') for img in closed_imgs_2]
    # dilated_imgs_3 = [morphological_operation(img, int(kernel_size / 10), 'dilation', its=2) for img in eroded_imgs_3]

    ##################################################################################################################
    connected_comp_imgs = [cv2.connectedComponentsWithStats(img, connect, cv2.CV_32S) for img in processed_imgs]

    print('Time to Preprocess images:', time.time() - start)

    print('Detecting tissues...')
    # time the process
    start = time.time()
    # filtered_imgs = [detect_tissues(img, num_tissue) for img in connected_comp_imgs]
    iso_imgs, full_labels  = detect_tissues(connected_comp_imgs, num_tissue)
    print('Time to Detect tissues:', time.time() - start)

    ### COLOR LABEL IMAGES ###
    save_arrays_as_images(full_labels, use_colormap=True, output_folder=output_folder, file_prefix="labels", file_extension=".png")
    ##########################

    #Match overlap of filtered images so that the same tissues are matching
    # time the process
    # start = time.time()
    # filtered_imgs = match_overlap(filtered_imgs)
    # print('Time to Match overlap:', time.time() - start)

    assert_same_length(iso_imgs)

    print('Cropping images and stack...')
    # time the process
    start = time.time()
    # find biggest bounding box and mask
    tissue_bbox, centroid_slices, bbox_slices, ref_slices = find_biggest_bound_box(iso_imgs)


    # crop images
    cropped_imgs = crop_imgs(img_arr, tissue_bbox, centroid_slices, padding, iso_imgs, align_upsample_factor, scale_factor_x, scale_factor_y, thresh, kernel_size, scale=True)
    # cropped_imgs = crop_imgs([img_arr[4]], tissue_bbox, [centroid_slices[4]], scale_factor, padding, [filtered_imgs[4]])
    if crop_only:
        for i, img in enumerate(img_list):
            metadata = img.ome_metadata
            tiff.write(cropped_imgs[0][i], metadata=metadata)
        
        exit()

    # stack images
    stacked_imgs = stack_images(cropped_imgs)
    print('Time to Crop images and stack:', time.time() - start)


    print('Aligning images...')
    # time the process
    start = time.time()

    aligned_tissue_list = []
    for i, img in enumerate(stacked_imgs):

        # save before alignment
        summed_channels = np.sum(img, axis=1).astype(np.uint8)
        OmeTiffWriter.save(
            summed_channels,
            f"{output_folder}/{file_basename}_{i}_sumChannels_beforeAlignment.ome.tif",
            dim_order="ZYX",
            physical_pixel_sizes=pps,
        )

        # OPTUNA - hyperparameter tuning
        if optimize:
            optimize_sift_parameters(summed_channels, img, output_folder, ref_slices[i], i)
            continue
    


        # Normal alignment using dapi
        # aligned_tissue, average_dice = align_z_slices(img, ref_slices[i])

        # alignment using summed channels
        aligned_tissue, average_dice = align_z_slices(summed_channels, img, ref_slices[i])


        dice_list, d_avg, area_consist = calculate_metrics(aligned_tissue, ref_slices[i])
        print(f"Average Dice coefficient: {d_avg}")
        print(f"Average Area consistency: {area_consist}")

        # plot_stack(aligned_tissue)

        #save aligned tissue
        summed_channels = np.sum(aligned_tissue, axis=1).astype(np.uint8)
        OmeTiffWriter.save(
            summed_channels,
            f"{output_folder}/{file_basename}_{i}_sumChannels.ome.tif",
            dim_order="ZYX",
            physical_pixel_sizes=pps,
        )


        aligned_tissue_list.append(aligned_tissue)

    if optimize:
        exit()

    
    print('Time to Align images:', time.time() - start)


    print('Saving images...')
    # time the process
    start = time.time()
    # save stack images
    for i, img in enumerate(aligned_tissue_list):
        OmeTiffWriter.save(
            img,
            f"{output_folder}/{file_basename}_{i}.ome.tif",
            dim_order="ZCYX",
            channel_names=channelnames,
            physical_pixel_sizes=pps,
        )

    print('Time to Save images:', time.time() - start)
    print('Total time:', time.time() - start_begin)



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



def align_z_slices(summed_channel, image_4d, reference_z=0, align_channel=0, params=None):
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
        flann_check = params['flann_check']
        flann_trees = params['flann_trees']

    else:
        n_features = 0  # 0 means no limit
        n_octave_layers = 4  # Default value
        contrast_threshold = 0.03  # Lowering this might result in more features being detected
        edge_threshold = 5  # Lowering this might result in more features being detected 
        sigma = 2  
        ratio_threshold = 0.75  # Lowe's ratio test
        flann_check = 200
        flann_trees = 20

    sift = cv2.SIFT_create(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma)

    # sift = cv2.SIFT_create()
    # Convert the reference z-slice to grayscale for feature detection
    # ref_slice = cv2.cvtColor(image_4d[reference_z].transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)

    #average or total sum of all channels
    # ref_slice = convert_to_grayscale_sum(image_4d[reference_z])
    # ref_slice = np.sum(image_4d[reference_z].transpose(1, 2, 0), axis=2)
    # ref_slice = np.mean(image_4d[reference_z], axis=0).astype(np.uint8)

    

    #one channel - dapi
    # ref_slice = image_4d[reference_z, align_channel].astype(np.uint8)
    # binary_ref = generate_binary_mask(ref_slice)

    #otsu filter for threshold
    # _, ref_slice = cv2.threshold(ref_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #summed channels method
    ref_slice = summed_channel[reference_z]
    binary_ref = generate_binary_mask(ref_slice)

    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_slice, None)

    

    # Initialize FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)
    search_params = dict(checks=flann_check)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Initialize an empty array for the aligned z-slices
    aligned_image_4d = np.zeros_like(image_4d)

    #get average dice
    average_dice = []

    try:

        # Align each z-slice to the reference
        for z in range(image_4d.shape[0]):
            if z != reference_z:
                # Convert z-slice to grayscale
                # current_slice = convert_to_grayscale_sum(image_4d[z])
                # current_slice = cv2.cvtColor(image_4d[z].transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
                # current_slice = np.mean(image_4d[z], axis=0).astype(np.uint8)
                # current_slice = np.sum(image_4d[z].transpose(1, 2, 0), axis=2)

                #one channel
                # current_slice = image_4d[z, align_channel].astype(np.uint8)

                #all channels
                # current_slice = convert_to_grayscale_sum(image_4d[z])
                # current_slice = image_4d[z]
                current_slice = summed_channel[z]

                #otsu filter for threshold
                # _, current_slice = cv2.threshold(current_slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                keypoints, descriptors = sift.detectAndCompute(current_slice, None)

                #print keypoints and descriptors
                # print(f"Keypoints: {len(keypoints)}")
                # print(f"Descriptors: {descriptors.shape}")

                # Match descriptors
                matches = flann.knnMatch(descriptors_ref, descriptors, k=2)

                # Filter matches using Lowe's ratio test
                # good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

                #print good matches
                print(f"Good matches: {len(good_matches)}")

                # Find homography matrix
                src_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                try: 
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    aligned_slice = cv2.warpPerspective(image_4d[z].transpose(1, 2, 0), M, (image_4d.shape[3], image_4d.shape[2]))

                    #print M
                    # print(f"Homography matrix for slice {z}: {M}")
                except Exception as e:
                    print(e)
                    return None, 0
                    

                # Apply the transformation to align the z-slice
                aligned_image_4d[z] = aligned_slice.transpose(2, 0, 1)

                # Update the keypoints and descriptors for the reference slice as the currrent aligned slice
                # keypoints_ref, descriptors_ref = sift.detectAndCompute(aligned_image_4d[z, align_channel], None)
                # keypoints_ref = keypoints
                # descriptors_ref = descriptors

                #check dice coeff against ref   
                # binary_slice = generate_binary_mask(aligned_image_4d[z, align_channel])
                binary_slice = generate_binary_mask(summed_channel[z])
                intersection = np.logical_and(binary_ref, binary_slice)
                dice = 2. * intersection.sum() / (binary_ref.sum() + binary_slice.sum())
                print(f"For reference slice {reference_z} and current slice {z} the dice coeff: {dice}")
                average_dice.append(dice)

            else:
                # Copy the reference slice as-is
                aligned_image_4d[z] = image_4d[z]
    except Exception as e:
        print(e)
        return None, 0

    return aligned_image_4d, np.mean(average_dice)

def generate_binary_mask(image, threshold=30):
    """
    Convert an image to a binary mask based on a threshold.
    
    :param image: Input image (2D array).
    :param threshold: Threshold value for binarization.
    :return: Binary mask of the image.
    """
    # _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresholds = threshold_multiotsu(image, classes=3)
    print(f"Multi Otsu: {thresholds}")
    # binary_mask = np.zeros_like(image, dtype=np.uint8)
    binary_mask = image > thresholds[0]
    binary_mask = (image > 0).astype(np.uint8)  # Convert to binary mask

    return binary_mask

def objective(trial, summed_channels, image_4d, ref_slices):
    # Define the range of values for each parameter you want to optimize
    n_features = trial.suggest_int('n_features', 0, 1000)
    contrast_threshold = trial.suggest_float('contrast_threshold', 0.01, 0.1)
    edge_threshold = trial.suggest_float('edge_threshold', 2, 15)
    sigma = trial.suggest_float('sigma', 0, 2.0)
    n_octave_layers = trial.suggest_int('n_octave_layers', 1, 10)
    ratio_threshold = trial.suggest_float('ratio_threshold', 0.5, 0.9)
    flann_check = trial.suggest_int('flann_check', 100, 500)
    flann_trees = trial.suggest_int('flann_trees', 5, 50)

    #put parameters in a dictionary
    params = {
        'n_features': n_features,
        'n_octave_layers': n_octave_layers,
        'contrast_threshold': contrast_threshold,
        'edge_threshold': edge_threshold,
        'sigma': sigma,
        'ratio_threshold': ratio_threshold,
        'flann_check': flann_check,
        'flann_trees': flann_trees

    }

    # Initialize SIFT with suggested parameters
    # sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)

    # Perform alignment using the current SIFT configuration
    # Note: You'll need to modify your alignment function to accept the `sift` parameter
    _, average_dice = align_z_slices(summed_channels, image_4d, ref_slices, params=params)
    
    # Evaluate alignment quality
    # _, average_dice, _ = calculate_metrics(aligned_image_4d)
    
    # Since Optuna minimizes the objective, return a negative value of the metric if higher is better
    return average_dice


# Example usage
def optimize_sift_parameters(summed_channels, image_4d, output_dir, ref_slices, image_index):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, summed_channels, image_4d, ref_slices), n_trials=25)  # Adjust n_trials to your preference

    
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    trials_data = []
    # Detailed logging of all trials
    for trial in study.trials:
        print(f"Trial {trial.number}, Value: {trial.value}")
        print(f" Parameters: {trial.params}")
        # If you have user attributes or other specific data to log, you can access them like this:
        # print(f" User Attributes: {trial.user_attrs}")
        print("-------------")

        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            # Include any other trial information you need
            # 'user_attrs': trial.user_attrs,
            # 'system_attrs': trial.system_attrs,
        }
        trials_data.append(trial_data)
    
    # Serialize to JSON and save to file
    with open(f'{output_dir}/optuna_trials_image_{image_index}.json', 'w') as f:
        json.dump(trials_data, f, indent=4)

    print('All trials have been saved to optuna_trials.json.')


def calculate_metrics(aligned_image_4d, ref_slice, align_channel=0, threshold=30):
    """
    Calculate the Dice coefficient and area consistency between consecutive slices in a 4D image array
    after converting each slice into a binary image.
    
    :param aligned_image_4d: 4D image array with shape (z-slice, channels, height, width).
    :param align_channel: The channel index to use for creating binary images.
    :param threshold: Threshold value for binarization.
    :return: List of Dice coefficients between consecutive slices.
    """

    if aligned_image_4d is None:
        return [], 0, 0

    dice_coefficients = []
    area_consistency = []

    #get or of all channels
    all_sum_img_per_z = [sum_channels(img) for img in aligned_image_4d]
    all_sum_img_per_z = [convert_to_grayscale_sum(img) for img in all_sum_img_per_z]

    #convert to binary mask
    all_binary_img_per_z = [generate_binary_mask(img) for img in all_sum_img_per_z]

    # sum all channels
    # prev_sum_img = all_sum_img_per_z[0]

    
    # Convert the first slice to a binary mask and store as the previous slice's mask
    # prev_slice_mask = generate_binary_mask(aligned_image_4d[0, align_channel])
    # prev_slice_mask = generate_binary_mask(prev_sum_img)
    prev_slice_mask = all_binary_img_per_z[0]

    #or all images in the list 
    image_coverage = np.logical_or.reduce(all_binary_img_per_z)
    
    for z in range(1, aligned_image_4d.shape[0]):
        # Convert the current slice to a binary mask
        # current_sum_img = convert_to_grayscale_sum(aligned_image_4d[z])

        # current_slice_mask = generate_binary_mask(aligned_image_4d[z, align_channel])
        # current_slice_mask = generate_binary_mask(current_sum_img)
        current_slice_mask = all_binary_img_per_z[z]
        
        # Calculate Dice coefficient
        intersection = np.logical_and(prev_slice_mask, current_slice_mask)
        dice = 2. * intersection.sum() / (prev_slice_mask.sum() + current_slice_mask.sum())

        # Calculate area consistency
        compare = (current_slice_mask == image_coverage)
        overlap = compare.sum() / image_coverage.sum()

        print(f"Slice {z} - Dice coefficient (compared to previous slice): {dice}, Area consistency (compared to overall coverage): {overlap}")


        area_consistency.append(compare.sum() / image_coverage.sum())
        dice_coefficients.append(dice)
        
        # Update the previous slice mask
        prev_slice_mask = current_slice_mask

    average_dice = np.mean(dice_coefficients) if dice_coefficients else 0
    average_area_consistency = np.mean(area_consistency) if area_consistency else 0
    
    return dice_coefficients, average_dice, average_area_consistency

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
    stacked_images = [np.stack(tissue_imgs[i], axis=0) for i in range(num_images)]

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


def close_blob(binary_image, kernel_size=10):
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

def upsample_image(image, sfx, sfy):
    """
    Upsample the image by the specified factor.

    :param image: Input image to upsample.
    :param upsample_factor: Factor to upsample the image by.
    :return: Upsampled image.
    """
    # Calculate the new dimensions
    new_height = int(image.shape[0] * sfy)
    new_width = int(image.shape[1] * sfx)

    # Upsample the image using cubic interpolation
    upsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return upsampled_image

def crop_imgs(imgs, bbox, centroids, padding, filtered_imgs, upsample_factor, sfx, sfy, thresh, kernel_size=10, scale=True):
    
    cropped_slices = []

    for i, img in enumerate(imgs):
        cropped_imgs = []
        for j in range(len(bbox)):
            w, h = bbox[j]
            cX, cY = centroids[i][j]
            # x, y, _, _ = bbox_slices[i][j]
            x1, y1, x2, y2 = create_bounding_box(cX, cY, w, h, img, padding, sfx, sfy, scale=True)


            mask = filtered_imgs[i][j]

            #erode image a bit to not include noise around the tissue
            # mask = morphological_operation(mask, kernel_size=kernel_size, operation='erosion')

            #remove holes
            # mask = cv2.bitwise_not(mask)
            # kernel_2 = np.ones((kernel_size * 3, kernel_size * 3), np.uint8)
            # opened_image = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_2)
            # mask = cv2.bitwise_not(opened_image)

            #erode image a bit to not include noise around the tissue
            mask = morphological_operation(mask, kernel_size=int(kernel_size * 2), operation='erosion', its=1)

            #print number of pixels before and after these morphological operations
            # print(f"Number of pixels before morphological operations: {np.sum(filtered_imgs[i][j])}")
            # print(f"Number of pixels after morphological operations: {np.sum(mask)}")


            # if scale is True
            #use upsample mask on original image
            upsampled_mask = upsample_image(mask, sfx, sfy)
            upsampled_mask = (upsampled_mask > 0).astype(np.uint8) #convert to [0, 1]

            #crop image and mask first to save memory
            new_img = img[:, y1:y2, x1:x2]
            upsampled_mask = upsampled_mask[y1:y2, x1:x2]

            # new_cX = abs(x1 - x2) / 2
            # new_cY = abs(y1 - y2) / 2

            #create new mask
            mask = process_tissue(new_img, thresh[i], kernel_size, upsampled_mask, connect=2)


            #apply mask to original image
            # mask_img = img * upsampled_mask
            # mask_img = img * mask
            # new_img = new_img * upsampled_mask

            new_img = new_img * mask

            ############################################
            #crop image
            # new_img = mask_img[:, y1:y2, x1:x2]
            # new_img = img[:, y1:y2, x1:x2]
            # binary_img = new_img.copy()
            # binary_img = np.sum(binary_img, axis=0)

            # #normalize image
            # binary_img = cv2.normalize(binary_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # #threshold image
            # _, binary_img = cv2.threshold(binary_img, 1, 255, cv2.THRESH_BINARY)

            # #close blob
            # binary_img = close_blob(binary_img, kernel_size=50)

            # #find connected components
            # cc_img = cv2.connectedComponentsWithStats(binary_img, 2, cv2.CV_32S)

            #get bounding box of largest connected component




            


            cropped_imgs.append(new_img)
            # cropped_imgs.append(img[y:y+h, x:x+w])

        cropped_slices.append(cropped_imgs) 

    
    return cropped_slices

def process_tissue(cropped_img, thresh, kernel_size, up_mask, connect=2):

    #set new kernel_size
    kernel_size = int(kernel_size / 5)

    img_2D = np.sum(cropped_img, axis=0)
    img_2D_thresh = ((img_2D > thresh) * 255).astype(np.uint8)

    #first round
    closed_img = morphological_operation(img_2D_thresh, kernel_size, 'closing')
    eroded_img = morphological_operation(closed_img, kernel_size, 'erosion')
    dilated_img = morphological_operation(eroded_img, int(kernel_size), 'dilation')

    #second round
    open_img = morphological_operation(dilated_img, kernel_size, 'opening')
    closed_img_2 = morphological_operation(open_img, kernel_size, 'closing')

    processed_img = morphological_operation(closed_img_2, int(kernel_size * 2), 'dilation')

    #remove small holes
    hole_thresh = int(processed_img.shape[0] * processed_img.shape[1] / 100) 
    processed_img_final = morphology.remove_small_holes(processed_img, area_threshold=hole_thresh)
    processed_img_final = (processed_img_final * 255).astype(np.uint8)

    #connected components
    cc_img = cv2.connectedComponentsWithStats(processed_img_final, connect, cv2.CV_32S)

    # num_labels = cc_img[0]  # The first cell is the number of labels
    labels = cc_img[1]  # The second cell is the label matrix itself
    # stats = cc_img[2]  # The third cell is the stat matrix

    # nunique = np.unique(labels)
    # sort_stats = stats[:, cv2.CC_STAT_AREA].copy()
    # sort_stats.sort()

    #get biggest piece - not correct as there is a bug in which the noise is a bigger piece
    # tissue_values = nunique[stats[:, cv2.CC_STAT_AREA] == sort_stats[-1]]
    # tissue_value = labels[cY, cX]
    masked_area = labels[up_mask > 0].flatten()
    # foreground_values = masked_area[masked_area != 0].flatten()
    tissue_value, _ = stats.mode(masked_area)

    mask = labels == tissue_value[0]
    mask = mask.astype(np.uint8)

    #process the mask more for irregularities
    mask_closed = morphological_operation(mask, kernel_size * 8, 'closing')
    mask_final = morphology.remove_small_holes(mask_closed, area_threshold=hole_thresh)

    return mask_final


def create_bounding_box(cX, cY, width, height, img, padding, sfx, sfy, scale=True):
    x1 = cX - width // 2
    y1 = cY - height // 2
    x2 = cX + width // 2
    y2 = cY + height // 2

    if scale:
        # Adjust if bounding box goes beyond image boundaries (assuming image dimensions are known)
        x1 = max(x1, 0) * sfx
        y1 = max(y1, 0) * sfy
        x2 = min(x2, img.shape[2]) * sfx
        y2 = min(y2, img.shape[1]) * sfy
    else: 
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img.shape[2])
        y2 = min(y2, img.shape[1])

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
    slice_list = []

    # Iterate through each index of bounding boxes across slices
    for i in range(max_bounding_boxes):
        max_area = 0
        # biggest_bbox = None
        # biggest_centroid = None
        best_slice = 0

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
                    best_slice = slice_index

                    #close blob
                    # best_mask = close_blob(best_mask)

        biggest_bounding_boxes.append((w_best, h_best))
        # centroid_of_biggest.append(biggest_centroid)
        # biggest_mask.append(best_mask)
        slice_list.append(best_slice)

        # d[i] = biggest_bbox
    

    return biggest_bounding_boxes, centroids_slices, bounding_box_slices, slice_list


def assert_same_length(list_of_lists):
    assert all(len(inner_list) == len(list_of_lists[0]) for inner_list in list_of_lists), "Not all lists are of the same length."

def compute_bounding_box(imgs):

    bounding_box = []
    centroids = []

    #get the individual tissues
    for img in imgs:
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
            # print("Centroid coordinates: ({}, {})".format(cX, cY))

            # bounding_box.append((x_padded, y_padded, w_padded, h_padded))
            bounding_box.append((x, y, w, h))
            centroids.append((cX, cY))

    return bounding_box, centroids

def detect_tissues(cc_img_list, num_tissue=8):

    filtered_imgs = []
    new_values = np.arange(0, num_tissue + 1)  # New label values for the filtered images

    for cc_img in cc_img_list:
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
        
        for new, old in zip(new_values, tissue_values):
            new_labels[new_labels == old] = new

        filtered_imgs.append(new_labels)

    # Relative position matching by centroid
    all_centroids = [calculate_centroids(img) for img in filtered_imgs]
    sorted_labels_per_image = [sort_split_combine(centroids) for centroids in all_centroids]


    # MATCHING BASED ON FEATURES
    # matches = match_all_images(filtered_imgs)
    # global_mappings = propagate_labels(matches)
    # reindexed_images = apply_new_labels(filtered_imgs, global_mappings)
        
    # MATCHING BASED ON OVERLAP
    # max_height = max(img.shape[0] for img in filtered_imgs)
    # max_width = max(img.shape[1] for img in filtered_imgs)

    # # Pad images to the maximum width and height
    # padded_images = [np.pad(img, ((0, max_height - img.shape[0]), (0, max_width - img.shape[1])), 'constant', constant_values=0) for img in filtered_imgs]

    # all_matches = []  # To store match results for each image pair

    # for i in range(len(padded_images) - 1):
    #     source_image = padded_images[i]
    #     target_image = padded_images[i + 1]
    #     matches = {}
    #     overlaps = {}
    #     matched_list = []

    #     for label in range(1, num_tissue+1):  # Assuming labels 1 to 8
    #         matching_label, overlap = calculate_overlap(source_image, target_image, label, matched_list)
    #         matched_list.append(matching_label)
    #         if matching_label is not None:
    #             matches[label] = matching_label
    #             overlaps[(label, matching_label)] = overlap

    #     all_matches.append(matches)

    # global_mappings = propagate_labels(all_matches)
    # reindexed_images = apply_new_labels(filtered_imgs, global_mappings)
    
    slices_tissue = []
    remapped_images = []
    for i, img in enumerate(filtered_imgs):
        iso_tissue = []

        remapped_image = np.zeros_like(img)
        for label, index in sorted_labels_per_image[i].items():  # Skip the background label 0
            large_blobs = np.zeros_like(img, dtype=np.uint8)
            large_blobs[img == label] = 255
            iso_tissue.append(large_blobs)

            #renumber 
            remapped_image[img == label] = index
        slices_tissue.append(iso_tissue)
        remapped_images.append(remapped_image)

    return slices_tissue, remapped_images

def sort_split_combine(centroids):
    # Convert the centroids dictionary to a list of tuples (label, (x, y))
    centroids_list = list(centroids.items())

    # Step 2: Sort by x-coordinate
    sorted_by_x = sorted(centroids_list, key=lambda item: item[1][0])
    
    # Step 3: Split into two halves
    mid_point = len(sorted_by_x) // 2
    first_half = sorted_by_x[:mid_point]
    second_half = sorted_by_x[mid_point:]
    
    # Step 4: Sort each half by y-coordinate
    sorted_first_half = sorted(first_half, key=lambda item: item[1][1], reverse=True)  # Reverse for higher indexing
    sorted_second_half = sorted(second_half, key=lambda item: item[1][1])

    # Combine the halves, with the first half first
    combined_sorted = sorted_first_half + sorted_second_half
    
    # Assign new indices based on this combined order
    new_order = {item[0]: index+1 for index, item in enumerate(combined_sorted)}  # +1 if indexing should start from 1

    return new_order

def calculate_centroids(image, num_tissue=8):
    centroids = {}
    for label in range(1, num_tissue + 1):  # Assuming labels from 1 to 8
        positions = np.argwhere(image == label)
        if positions.size == 0:  # If the label is not found in the image
            continue
        centroid = positions.mean(axis=0)
        centroids[label] = (centroid[1], centroid[0])  # (x, y) format
    return centroids

def calculate_overlap(source_image, target_image, label, matched_list):
    """
    Calculate the overlap of a blob labeled 'label' in 'source_image' with
    blobs in 'target_image'. Returns the label in 'target_image' with the highest overlap.
    """

    # matched_list = []

    # Create a binary mask for the blob in the source image
    source_mask = source_image == label
    
    # Initialize variables to track the maximum overlap
    max_overlap = 0
    max_label = None
    
    # Check overlap with each label in the target image
    for target_label in np.unique(target_image):
        if target_label == 0:  # Skip background
            continue
        
        # Create a binary mask for the current blob in the target image
        target_mask = target_image == target_label
        
        # Calculate overlap by counting the pixels where both masks are True
        overlap = np.sum(source_mask & target_mask)
        print(f"Overlap between source label {label} and target label {target_label}: {overlap}")
        
        # Update max overlap and corresponding label
        if overlap > max_overlap and target_label not in matched_list:
            max_overlap = overlap
            max_label = target_label
    
    return max_label, max_overlap

def find_matching_blob(centroid, target_image):
    """
    Find the blob in 'target_image' that overlaps with the given 'centroid'.
    """
    x, y = int(centroid[0]), int(centroid[1])
    # Check if the centroid falls within a labeled blob in the target image
    return target_image[y, x] if target_image[y, x] != 0 else None


def propagate_labels(all_matches):
    """
    Propagates label changes across all images based on match mappings.
    
    :param all_matches: List of dictionaries with match mappings between consecutive image pairs.
    :return: List of dictionaries representing the global label mapping for each image.
    """
    global_mappings = [{} for _ in range(len(all_matches) + 1)]  # Initialize global mappings

    # Start with the first image, assuming its labels are correct
    for label in range(1, 9):  # Assuming labels 1 to 8
        global_mappings[0][label] = label
    
    # Propagate matches
    for i, matches in enumerate(all_matches):
        for label1, label2 in matches.items():
            global_mappings[i + 1][label2] = global_mappings[i][label1]
    
    return global_mappings

def apply_new_labels(images, global_mappings):
    """
    Applies new labels to all images based on the global label mappings.
    
    :param images: List of 2D arrays, each representing an image with labeled blobs.
    :param global_mappings: List of dictionaries representing the global label mapping for each image.
    :return: List of 2D arrays with reindexed blobs.
    """
    reindexed_images = []
    for img, mapping in zip(images, global_mappings):
        new_img = np.zeros_like(img)
        for old_label, new_label in mapping.items():
            new_img[img == old_label] = new_label
        reindexed_images.append(new_img)
    return reindexed_images

def extract_features(image):
    """
    Extracts a comprehensive set of features for each labeled blob in the image,
    including area, centroid, Hu Moments, eccentricity, solidity, perimeter,
    compactness, and Fourier Descriptors.
    Returns a dictionary with labels as keys and feature vectors as values.
    """
    features = {}
    for label in np.unique(image):
        if label == 0:  # Skip background
            continue
        blob = (image == label).astype(np.uint8)
        
        # Find contours for the current blob
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic Shape Descriptors
        area = cv2.contourArea(largest_contour)
        M = cv2.moments(largest_contour)
        centroid_x = M['m10'] / M['m00']
        centroid_y = M['m01'] / M['m00']
        huMoments = cv2.HuMoments(M).flatten()
        huMoments = np.sign(huMoments) * np.log10(np.abs(huMoments))
        
        # Eccentricity and Solidity
        (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
        eccentricity = np.sqrt(1 - (MA/ma)**2)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        
        # Boundary Features
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (perimeter**2) / (4 * np.pi * area)
        
        # Fourier Descriptors
        contour_complex = largest_contour[:, 0, 0] + 1j * largest_contour[:, 0, 1]
        fourier_result = fft(contour_complex)
        # Normalize and keep a fixed number of descriptors
        fourier_descriptors = np.abs(fourier_result[:10]) / np.abs(fourier_result[0])

        # Combine all features into a single vector
        features[label] = np.hstack([huMoments, solidity, eccentricity, area, centroid_x, centroid_y, perimeter, compactness])
        
    return features

def standardize_features(features):
    """
    Standardizes the feature vectors for all blobs across all images.
    """
    all_features = np.vstack(list(features.values()))
    scaler = StandardScaler().fit(all_features)
    for label in features:
        features[label] = scaler.transform(features[label].reshape(1, -1))
    return features

def compare_blobs(features1, features2):
    """
    Compares blobs between two images based on their standardized features.
    Returns a dictionary mapping labels in the first image to labels in the second image.
    """
    labels1, features1 = zip(*features1.items())
    labels2, features2 = zip(*features2.items())
    features1 = np.array(features1).squeeze()
    features2 = np.array(features2).squeeze()
    
    # Calculate pairwise distances between standardized feature vectors
    distances = cdist(features1, features2, 'euclidean')
    
    matches = {}
    for i, label1 in enumerate(labels1):
        match_index = np.argmin(distances[i])
        match_label = labels2[match_index]
        matches[label1] = match_label
    return matches

def match_all_images(images):
    """
    Matches blobs across multiple images after standardizing features.
    """
    all_matches = []
    all_features = [extract_features(img) for img in images]

    # Standardize features across all blobs in all images
    standardized_features = [standardize_features(features) for features in all_features]
    
    for i in range(len(standardized_features) - 1):
        matches = compare_blobs(standardized_features[i], standardized_features[i + 1])
        all_matches.append(matches)
    
    return all_matches

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


def morphological_operation(img, kernel_size, operation, its=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilation':
        return cv2.dilate(img, kernel, iterations=its)
    elif operation == 'erosion':
        return cv2.erode(img, kernel, iterations=its)
    else:
        return img

def sum_channels(img_arr):
    
    return [np.sum(img, axis=0, dtype=np.uint16) for img in img_arr]

def plot_img_from_list(img_list):
    fig, ax = plt.subplots(1, len(img_list))
    for i, img in enumerate(img_list):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_img(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
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

    p = ArgumentParser()
    p.add_argument('--num_tissue', type=int, default=1, help='Number of tissues to detect, default is 8')
    p.add_argument('--level', type=int, default=0, help='Pyrmaid level of the image, default is 0 which is the original image size')
    p.add_argument('--thresh', type=int, default=None, help='Threshold value for binarization, default is done by otsu')
    p.add_argument('--kernel_size', type=int, default=100, help='Size of the structuring element used for closing, default is 0')
    p.add_argument('--holes_thresh', type=int, default=5000, help='Area threshold for removing small holes, default is 300')
    p.add_argument('--scale_factor', type=int, default=8, help='Scale factor for downsample, default is 10')
    p.add_argument('--padding', type=int, default=50, help='Padding for bounding box, default is 20')
    p.add_argument('--connect', type=int, default=2, help='Connectivity for connected components, default is 2')
    p.add_argument('--pixel_size', type=list, default=[0.5073519424785282, 0.5073519424785282], help='Physical pixel size of the image in microns, default is [0.5073519424785282, 0.5073519424785282]')
    # p.add_argument('--pixel_size', type=list, default=[4.058815539828226, 4.058815539828226], help='Physical pixel size of the image in microns, default is [0.5073519424785282, 0.5073519424785282]')
    # p.add_argument('--pixel_size', type=list, default=[0.5082855933597976, 0.5082855933597976])
    p.add_argument('--output_dir', type=str, default='./outputs', help='Output folder for saving images, default is outputs')
    p.add_argument('--input_path', type=str, default='/hive/hubmap/data/CMU_Tools_Testing_Group/phenocycler/20c4aa0d79c0b8af37f27d436c1b42c4/QPTIFF-test/3D_image_stack.ome.tiff', help='Input folder for reading images, default is inputs')
    # p.add_argument('--input_path', type=str, default='raw_data', help='Input folder for reading images, default is inputs')
    p.add_argument('--output_file_basename', type=str, default='aligned_tissue', help='Output file basename, default is aligned_tissue')
    p.add_argument('--align_upsample_factor', type=int, default=2, help='Upsample factor for aligning images, default is 2')
    p.add_argument('--optimize', type=bool, default=True, help="optimize alignment parameters using optuna")
    p.add_argument('--crop_only', type=bool, default=True, help="only identify tissues and crop, no alignment")

    args = p.parse_args()

    main(
        num_tissue = args.num_tissue,
        level = args.level,
        thresh = args.thresh,
        kernel_size = args.kernel_size,
        holes_thresh = args.holes_thresh,
        scale_factor = args.scale_factor,
        align_upsample_factor = args.align_upsample_factor,
        padding = args.padding,
        connect = args.connect,
        pixel_size = args.pixel_size,
        output_folder = args.output_dir,
        input_path = args.input_path,
        file_basename = args.output_file_basename,
        optimize= args.optimize,
        crop_only = args.crop_only

    )