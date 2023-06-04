#!/usr/bin/env python3

# Library imports
import numpy as np
import pandas as pd
import os
import glob
import skimage
import pickle
import zarr
from PIL import Image
import scipy
from dotenv import load_dotenv
import dask.array as da
from dask import delayed
from napari.settings import get_settings
from itertools import groupby, count
import mahotas as mh
import math

# Load environment variables.
load_dotenv()

# Flag to determine if sample/mask mappings exist.
HAVE_MAPPINGS = os.environ["HAVE_MAPPINGS"]

# Set up directory paths.
BASE_DIR = os.environ["BASE_DIR"]
SAMPLES_DIR = os.environ["SAMPLES_DIR"]
UTILITY_DATA_DIR = os.environ["UTILITY_DATA_DIR"]
LUMEN_DIR = os.path.join(UTILITY_DATA_DIR, "lumen")
AREA_LOWER_BOUND = os.environ["AREA_LOWER_BOUND"]

sample_files = glob.glob(SAMPLES_DIR+"*.svs")
masks_files = glob.glob(SAMPLES_DIR+"*.tif")

# Dataframe containing sample/mask coordinate mappings.
df = pd.read_csv(os.path.join(BASE_DIR, "updated_coords.csv"))
df.set_index("Hashed Accession #", inplace=True)

def gray_image_maker(rgb_image):
    '''Converts RGB image to grayscale color space'''
    
    def gray_scale(rgb):
        result = ((rgb[..., 0] * 0.2125) +
                  (rgb[..., 1] * 0.7154) +
                  (rgb[..., 2] * 0.0721))
        return result
    # Uses Dask takes longer but does not crash on big images.
    single_image_result = gray_scale(da.from_array(rgb_image, chunks="auto"))
    return np.asarray(single_image_result.astype("uint8"))

def reduce_blank_dimensions(image, mask):
    '''Takes sample image with corresponding mask and removes blank space,
        returns cropped sample and image masks'''
    matches = image.shape[:2] == mask.shape
    if not matches:
        raise ValueError("Bad dimensions! image: ", image.shape[:2], "mask: ", mask.shape)
    # Get max value from each image row to know where biopsy samples 
    # are and where blank space is.
    rows = np.max(mask, axis=1)
    not_blank_rows = np.squeeze(np.argwhere(rows == 1), axis=-1)
    lst = []

    # Accumulate row slices if more than 100 rows are blank.
    groups = groupby(not_blank_rows, key=lambda item, c=count(): item - next(c))
    for index, group in groups:
        not_blank_region = list(group)
        if len(not_blank_region) > 100:
            slicing = [not_blank_region[0], not_blank_region[-1]]
            lst.append(slice(*slicing))

    # Stitch sample and mask images before returning
    reduced_image = np.vstack([image[slc] for slc in lst])
    reduced_mask = np.vstack([mask[slc] for slc in lst])
    return reduced_image, reduced_mask

def save_file_mapping(filename, dict_to_save):
    with open(os.path.join(UTILITY_DATA_DIR, "misc", filename), "wb") as fp:
        print("Saving file mapping file")
        pickle.dump(dict_to_save, fp)
        print("Done.")

def load_file_mapping(filename):
    with open(os.path.join(UTILITY_DATA_DIR, "misc", filename), "rb") as fp:
        print("Loading file mappings")
        files_mapping = pickle.load(fp)
        print("Done.")
    return files_mapping

def update_file_mapping(samples_lst, mask_lst, df):
    files_mapping = dict()
    for f_name, jobid in df[df.columns[:2]].to_dict("tight")["data"]:
        for sample_name in samples_lst:
            if f_name in sample_name:
                for mask_name in mask_lst:
                    if f"_{jobid}_" in mask_name:
                        files_mapping[f_name] = [sample_name, mask_name]
    return files_mapping


# Create accession and mask/sample path mapping
if HAVE_MAPPINGS:
    files_mapping = load_file_mapping("files_mapping.pkl")
else:
    files_mapping = update_file_mapping(sample_files, masks_files, df)
    save_file_mapping("files_mapping.pkl", files_mapping)


# Classic feature definition
feature_prop = ['label', 'area', 'eccentricity', 'extent', 'min_intensity',
                'mean_intensity', 'max_intensity', 'solidity', 'axis_major_length',
                'axis_minor_length', 'feret_diameter_max', 'perimeter', 'moments_hu']

def centiles(regionmask, intensity):
    # Returns percentiles of color intensities in 5% intervals of region.
    return np.percentile(intensity[regionmask], q=tuple(range(5, 100, 5)))

def std_intensity(regionmask, intensity):
    # Returns color intensity standard deviation of region.
    return np.std(intensity[regionmask])

def return_shape(slice_obj):
    return [slice_obj[0].stop - slice_obj[0].start, slice_obj[1].stop - slice_obj[1].start]

def heralic_feats(gray_image):
    # Return Haralick feature vector for region, 
    # ignores background in calculations.
    return mh.features.haralick(gray_image, ignore_zeros=True).ravel()


for k, v in files_mapping.items():
    spl, msk = v
    # open sample and mask images.
    sample = skimage.io.imread(spl, plugin="tifffile")
    mask = skimage.io.imread(msk, plugin="tifffile")
    # Fix mask to binary image
    mask[(mask > 0) & (mask != 1)] = 0
    # Fill holes in mask in case theres holes.
    filled_holes = scipy.ndimage.binary_fill_holes(mask)
    # Classifier coordinates for mapping
    roa_top, roa_left, roa_height, roa_width, roa_zoom = df.loc[k, df.columns[1:6]].to_list()
    # Resample points in mask to fit biopsy sample resolution.
    downsampled_mask = scipy.ndimage.zoom(mask, prefilter=False, zoom=1/roa_zoom)
    
    # Crop sample to mask dimensions
    sub_sample = sample[roa_top:roa_top + roa_height, roa_left:roa_left + roa_width]
    
    # Remove blank space, checking vertically and horizontally.
    nu_image, nu_mask = reduce_blank_dimensions(sub_sample, downsampled_mask)
    transposed_image, transposed_mask = reduce_blank_dimensions(np.transpose(sub_sample, (1, 0, 2)), downsampled_mask.T)
    rowing_dim = nu_mask.shape[0] * nu_mask.shape[1]
    transp_dim = transposed_mask.shape[0] * transposed_mask.shape[1]

    # Keeping lower dimension images.
    if rowing_dim > transp_dim:
        nu_image, nu_mask = transposed_image, transposed_mask
    
    # Remove from RAM to free up memory
    del transposed_image
    del transposed_mask
    del sub_sample
    del downsampled_mask


    # Saving reduced masks
    mask_downsampled = Image.fromarray(np.asarray(nu_mask))
    mask_downsampled.save(os.path.join(UTILITY_DATA_DIR, "masks", f"{k}.tif"), compression='tiff_lzw')
    # Saving reduced biopsy samples
    sub_to_save = Image.fromarray(np.asarray(nu_image))
    sub_to_save.save(os.path.join(UTILITY_DATA_DIR, "samples", f"{k}.tif"), compression='tiff_lzw')

    # Labeling mask connected regions to ROI's
    label_mask, num_obj = scipy.ndimage.label(nu_mask)

    # Extracts feature properties as table, using label image and sample image.
    prop_features = skimage.measure.regionprops_table(label_mask, 
        intensity_image=nu_image, properties=feature_prop, extra_properties=(centiles, std_intensity))
    prop_haralick = skimage.measure.regionprops_table(label_mask, properties=['label', 'image', 'bbox'])

    # Dataframes for each feature sets, label for identification purposes.
    data_props = pd.DataFrame(prop_features)
    data_props.set_index("label", inplace=True)

    data_har = pd.DataFrame(prop_haralick)
    data_har.set_index('label', inplace=True)

    # Filtering dataframe in tubule area only lower bound is concidered for now.
    filt = (data_props.area > AREA_LOWER_BOUND)
    data_props = data_props[filt]

    #============================
    # RGB > Grayscale for Haralick feature calculations.
    tmp_gray = gray_image_maker(nu_image)
    # Thresholding gray image to get lumen area data.
    threshed_white = tmp_gray > 200
    nu_masks_labels = label_mask * threshed_white
    lumen_data = skimage.measure.regionprops_table(nu_masks_labels, properties=['label', 'area'])
    lumen_props = pd.DataFrame(lumen_data)
    lumen_props.set_index("label", inplace=True)
    lumen_props = lumen_props.loc[filt]

    # Calculating some features prematurely.
    data_props["lumen_area"] = lumen_props["area"]
    # Getting ratio of lumen area and tubule area
    data_props["l2tratio"] = data_props["lumen_area"] / data_props["area"]
    # Calculates circularity of tubule shape.
    data_props["circularity"] = (4 * math.pi * data_props["area"]) / (data_props["perimeter"] * data_props["perimeter"])
    data_props["aspect_ratio"] = data_props["axis_major_length"] / data_props["axis_minor_length"]

    #============================
    # Haralick features. 4 directions * 13 features = 52 features. 
    column_index = ['label', *range(1,53)]
    har_df = pd.DataFrame(columns=column_index)
    # Extrating individual tubules and calculating haralick features for each one.
    for lb in data_props.loc[filt].index.values:
        rmin, cmin, rmax, cmax = data_har.loc[lb][['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].to_numpy()
        img_gray_extract = tmp_gray[int(rmin):int(rmax), int(cmin):int(cmax)]
        gray_mask = data_har.loc[lb]['image']
        har_image_to_calculate = img_gray_extract * gray_mask
        feature_vector = [lb, *heralic_feats(har_image_to_calculate)]
        har_df.loc[len(har_df)] = feature_vector

    har_df.label = har_df.label.astype(int)
    har_df.set_index("label", inplace=True)

    # Merge dataframes and save to directory.
    T1 = pd.merge(data_props, har_df, on=har_df.index, how='outer')
    T1.rename(columns={'key_0': 'label'}, inplace=True)
    T1.to_csv(os.path.join(UTILITY_DATA_DIR, "dataframes", f"{k}.csv"), index=False)

    # Tubular diameter calculations same as with lumen but different mask.
    distance_from_labels = scipy.ndimage.distance_transform_edt(nu_masks_labels)
    propz = skimage.measure.regionprops_table(label_image=nu_masks_labels, intensity_image=distance_from_labels, properties=['label', 'intensity_max', 'area'])
    data_set = pd.DataFrame(propz)
    data_set = data_set[data_set.area > AREA_LOWER_BOUND]
    data_set = data_set.rename(columns={"intensity_max": "max_circle_diameter"})
    data_set.to_csv(os.path.join(LUMEN_DIR, f"{k}.csv"), index=False)
