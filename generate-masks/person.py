import os
import glob
from transformers import pipeline
from PIL import Image
import numpy as np

# Define directories
input_dir = "C:/hd-vton-dataset/test/image"
output_dir = "C:/hd-vton-dataset/test"

# Create output directory if it doesn't exist
depth_masks_dir = os.path.join(output_dir, "img-depth-v2")
if not os.path.exists(depth_masks_dir):
    os.makedirs(depth_masks_dir)

# Load the depth estimation pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Process each image in the input directory
for image_path in glob.glob(os.path.join(input_dir, "*.jpg")):
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

    # Convert normalized depth map to an 8-bit grayscale image
    depth_image = (normalized_depth * 255).astype(np.uint8)

    # Save the depth mask image
    base_name = os.path.basename(image_path).replace(".jpg", "")
    Image.fromarray(depth_image).save(os.path.join(depth_masks_dir, f"{base_name}.jpg"))

print("Depth masks extraction complete.")
