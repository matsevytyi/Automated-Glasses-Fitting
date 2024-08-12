
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


# Function to calculate the centroid of a mask
def calculate_centroid(mask):
    # Get the coordinates of all the pixels in the mask
    y_indices, x_indices = np.nonzero(mask)
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    return (centroid_x, centroid_y)

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_non_transparent_edges(image):
    # Split the image into its color channels and the alpha channel
    b, g, r, alpha = cv2.split(image)
    
    # Find the indices where alpha is greater than 0 (i.e., non-transparent pixels)
    non_transparent_coords = np.argwhere(alpha > 0)
    
    # If no non-transparent pixels found, return None
    if non_transparent_coords.size == 0:
        return None, None
    
    # Get the leftmost and rightmost x-coordinates
    leftmost_x = np.min(non_transparent_coords[:, 1])
    rightmost_x = np.max(non_transparent_coords[:, 1])
    
    return leftmost_x, rightmost_x

def load_from_xls(filename, columns, name):
    df = pd.read_excel(filename)
    df['EAN'] = df['EAN'].astype(str)
    name = str(name)
    df = df[df['EAN'] == name]
    df = df[columns].values[0]
    a, b = df[0], df[1]
    df = (int(a[:-2]), int(b[:-2]))
    return df