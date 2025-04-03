import os
import glob
from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define directories
input_dir = "C:/hd-vton-dataset/train"
output_dir = "C:/hd-vton-dataset/train"

# Create output directories if they don't exist
cloth_front_dir = os.path.join(output_dir, "cloth-front")
depth_map_dir = os.path.join(output_dir, "cloth-depth-map")
combined_mask_dir = os.path.join(output_dir, "cloth-front-mask")

for d in [cloth_front_dir, depth_map_dir, combined_mask_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# Load the depth estimation pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Process each image in the input directory
for image_path in glob.glob(os.path.join(input_dir, "cloth" , "*.jpg")):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Perform depth estimation
    depth_output = pipe(image)
    depth_map = depth_output["depth"]

    # Convert the depth map to a numpy array and normalize
    depth_map_np = np.array(depth_map)
    depth_min = depth_map_np.min()
    depth_max = depth_map_np.max()
    normalized_depth = (depth_map_np - depth_min) / (depth_max - depth_min)

    # Create a binary mask based on the depth values
    depth_threshold = 0.5  # Adjust this value based on your needs
    binary_mask = (normalized_depth > depth_threshold).astype(np.uint8)

    # Load the original binary mask of the dress (assumed to be in the same directory with "_mask" suffix)
    dress_mask_path = image_path.replace("cloth", "cloth-mask")
    if not os.path.exists(dress_mask_path):
        continue  # Skip if mask does not exist
    dress_mask_image = Image.open(dress_mask_path).convert("L")
    dress_mask = np.array(dress_mask_image).astype(np.uint8)

    # Ensure that masks are of the same size
    if dress_mask.shape != binary_mask.shape:
        raise ValueError("Masks must be of the same size.")

    # Apply depth mask to the dress mask
    combined_mask = np.where(binary_mask == 0, 0, dress_mask)

    # Apply the mask to the original image
    image_np = np.array(image)
    masked_image = np.where(combined_mask[:, :, np.newaxis] == 0, [255, 255, 255], image_np).astype(np.uint8)

    # Save the processed images
    base_name = os.path.basename(image_path).replace(".jpg", "")
    Image.fromarray((normalized_depth * 255).astype(np.uint8)).save(
        os.path.join(depth_map_dir, f"{base_name}.png"))
    Image.fromarray(combined_mask).save(os.path.join(combined_mask_dir, f"{base_name}.png"))
    Image.fromarray(masked_image).save(os.path.join(cloth_front_dir, f"{base_name}.png"))

print("Processing complete.")
