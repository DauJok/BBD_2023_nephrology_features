#!/usr/bin/env python3

import os
import skimage
import scipy
import pandas as pd
from dotenv import load_dotenv

# Load environment variables.
load_dotenv()

# Directory path setup
UTILITY_DATA_DIR = os.environ["UTILITY_DATA_DIR"]
DATAFRAMES_DIR = os.path.join(UTILITY_DATA_DIR, "dataframes")
MASK_DIR = os.path.join(UTILITY_DATA_DIR, "masks")
NEI_DATAFRAMES = os.path.join(UTILITY_DATA_DIR, "nei_dataframes")

NEIGHBOR_COUNT = 5 + 1 # plus 1 because first point is sample point to itself.
AREA_LOWER_BOUND = os.environ["AREA_LOWER_BOUND"]

for file_name in os.listdir(DATAFRAMES_DIR):
    msk_name = file_name.partition(".")[0]
    mask = skimage.io.imread(os.path.join(MASK_DIR, f"{msk_name}.tif"), plugin="tifffile")
    lbl, num_obs = scipy.ndimage.label(mask)
    # Euclidian distance transform finds diameter of biggest circle in ROI
    distance_from_labels = scipy.ndimage.distance_transform_edt(lbl)
    propz = skimage.measure.regionprops_table(label_image=lbl, intensity_image=distance_from_labels, properties=['label', 'intensity_max', 'area'])
    data_set = pd.DataFrame(propz)
    data_set = data_set[data_set.area > AREA_LOWER_BOUND]
    data_set = data_set.rename(columns={"intensity_max": "max_circle_diameter"})
    # Get centroids of tubules to calculate neighborhood distances.
    propx = skimage.measure.regionprops_table(label_image=lbl, properties=['label', 'centroid', 'area'])
    data_cords = pd.DataFrame(propx)
    data_cords = data_cords[data_cords.area > AREA_LOWER_BOUND]
    # Find n-th nearest neighbors and their distances
    tree = scipy.spatial.cKDTree(data_cords[["centroid-0", "centroid-1"]].values)
    # Query n-th neighbor tubules in KDtree.
    dd, ii = tree.query(data_cords[["centroid-0", "centroid-1"]].values, k=NEIGHBOR_COUNT)
    neighbors_df = pd.DataFrame(dd, columns=["label", "neighbour_1", "neighbour_2", "neighbour_3", "neighbour_4", "neighbour_5"])
    neighbors_df["label"] = data_set["label"].to_list()
    # Merge dataframes and save the dataframe also keeping ids for later analysis.
    final_df = pd.merge(neighbors_df, data_set, on="label", how="left")
    final_df = final_df.rename(columns={"key_0": "label", "area": "area_ch"})
    final_df.to_csv(os.path.join(NEI_DATAFRAMES, f"{msk_name}.csv"), index=False)
    print(f"Saving dataframe {msk_name}")
