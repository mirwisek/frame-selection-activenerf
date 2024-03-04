import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def get_focal_length(image_width, camera_angle_x):
  focal_length = image_width / (2 * np.tan(camera_angle_x / 2))
  return np.array(focal_length)


def load_tiny_nerf_data(path, device):
  # Load input images, poses, and intrinsics
  data = np.load(path)

  # Images
  images = data["images"]
  # Camera extrinsics (poses)
  
  tform_cam2world = data["poses"]
  
  # Focal length (intrinsics)
  focal_length = data["focal"]
  focal_length = torch.from_numpy(focal_length).to(device)

  # Hold one image out (for test).
  testimgs, testposes = images[100:], tform_cam2world[100:]
  testimgs = torch.from_numpy(testimgs).to(device)
  testposes = torch.from_numpy(testposes).to(device)

  # Map images to device
  images = torch.from_numpy(images[:100, ..., :3]).to(device)
  tform_cam2world = torch.from_numpy(tform_cam2world[:100,]).to(device)

  return images, tform_cam2world, focal_length, testimgs, testposes


def load_nerf_data(base_dir, subject, json_file, skip_images=0):
    """
    Load NeRF data from real NeRF dataset

    Parameters:
    - base_dir (str): The base directory where the dataset is located.
    - json_file (str): The name of the JSON config file containing frame data.
    - skip_images (int, optional): The number of images to skip after processing an image.
      This is useful for reducing the dataset size. For example, in test dataset we want to
      reduce the size of 200 images to get 10 most diversified angle images
      Default is 0, which means no images are skipped.

    Returns:
    - images (numpy.ndarray): Array of loaded and processed images.
    - poses (numpy.ndarray): Array of camera poses corresponding to each image.
    - camera_angle_x (float): Horizontal field of view of the camera in radians.
    """

    data_dir = os.path.join(base_dir, subject)

    with open(os.path.join(data_dir, json_file), 'r') as f:
        data = json.load(f)

    camera_angle_x = data['camera_angle_x']
    images = []
    poses = []

    skip_counter = 0  # Counter for skipping images

    for frame in data['frames']:
        # Skip images with 'depth' or 'normal' in their file path
        if 'depth' in frame['file_path'] or 'normal' in frame['file_path']:
            continue

        # Process the image if skip_counter is 0, otherwise skip it
        if skip_counter == 0:
            # Load and process the image
            image_path = os.path.join(data_dir, frame['file_path'] + '.png')
            image = Image.open(image_path).convert("RGB")
            resized_image = image.resize((100, 100))
            images.append(TF.to_tensor(resized_image))

            # Extract camera pose
            pose = np.array(frame['transform_matrix'], dtype=np.float32)
            poses.append(pose)

            # Reset the skip_counter if an image was processed
            skip_counter = skip_images
        else:
            # Decrease the skip_counter since we're skipping this image
            skip_counter -= 1

    # Convert lists to numpy arrays
    images = torch.stack(images)
    poses = np.array(poses)
    images = images.permute(0, 2, 3, 1)

    return images, poses, camera_angle_x