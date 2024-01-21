import torch
import numpy as np
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from nerf_utils import get_ray_bundle
from image_encoder import ImageEncoder
from pytorch3d.ops import box3d_overlap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from skimage.metrics import structural_similarity as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# Identify optimal PCA dimension for all the images
def converge_to_pca_dimensions(embeddings_flattened, min_variance=0.75):
    # Perform PCA for dimensionality reduction
    pca = PCA()
    pca.fit(embeddings_flattened)

    # Calculate explained variance ratio and cumulative explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var_ratio)

    # Find the number of components that explain the specified variance
    n_components = np.argmax(cum_explained_var >= min_variance) + 1

    if n_components > 100:
      print('Max number of components reached..\n')
      n_components = 100

    # Retrain PCA with the selected number of components
    pca = PCA(n_components=n_components)
    dim_reduction = pca.fit_transform(embeddings_flattened)

    print(f'Number of components: {n_components}\n')

    return dim_reduction, pca.explained_variance_ratio_, n_components

def camera_position_from_extrinsic_matrix(extrinsic_matrix):
    """
    Calculate the camera position from the extrinsic matrix using PyTorch.

    :param extrinsic_matrix: A 3x4 or 4x4 extrinsic matrix of the camera
    :return: The position of the camera in world coordinates
    """
    # Extract the rotation matrix (R) and the translation vector (t)
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]

    # Calculate the camera position: C = -R^-1 * t
    camera_position = -torch.inverse(R) @ t
    return camera_position

def furthest_view_sampling_k(train_tform_c2w, holdout_tform_c2w, k, device):
    # Calculate camera positions for the training set and holdout
    training_positions = [camera_position_from_extrinsic_matrix(transform) for transform in train_tform_c2w]
    holdout_position = [camera_position_from_extrinsic_matrix(transform) for transform in holdout_tform_c2w]
    training_positions = torch.stack(training_positions)
    holdout_position_orig = torch.stack(holdout_position)
    holdout_position = torch.stack(holdout_position)

    # Initialize a list to hold the furthest candidates
    furthest_candidates = []

    # Calculate the distance of each candidate to all camera positions in the training set   
    while len(furthest_candidates) < k-1:
      
      # convert list of tensor to tensor
      avg_train_pos = torch.mean(training_positions, dim=0)

      euclidean_distance = torch.sqrt(torch.sum((holdout_position - avg_train_pos) ** 2, dim=1))

      # get index with largest distance
      max_index = torch.argmax(euclidean_distance).item()

      # get element with largest distance
      max_element = holdout_position[max_index]

      # remove element from holdout_position
      holdout_position = torch.cat((holdout_position[:max_index], holdout_position[max_index+1:]))
      
      # get index of element from holdout_position_orig      
      
      max_element_index = torch.where(torch.all(torch.eq(holdout_position_orig, max_element), dim=1))[0].item()

      furthest_candidates.append(max_element_index)

      # Append the training_positions with maximum distance
      training_positions = torch.cat((training_positions, torch.unsqueeze(max_element, 0)))
      
    return furthest_candidates

def extract_features(image_array, device):
  # Instantiate the ImageEncoder
  image_encoder = ImageEncoder(device)

  # Get the embeddings
  embeddings = image_encoder(image_array)

  # Perform PCA for dimensionality reduction
  #  pca = PCA(n_components=10)

  image_embd = embeddings.squeeze().contiguous().view(embeddings.size(0), -1).cpu().detach().numpy()

  pca = PCA(n_components=5)
  #umap_reducer = umap.UMAP(n_neighbors=5, n_components=5, min_dist = 0.5)

  reduced_img_embd = pca.fit_transform(image_embd)
  #reduced_img_embd = umap_reducer.fit_transform(image_embd)

  scaler = StandardScaler()
  scaled_red_img_emb = scaler.fit_transform(reduced_img_embd)

  return scaled_red_img_emb

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2, ord=2)

def get_fvs_cluster(cluster_camera_pos):
    pairwise_distances = squareform(pdist(cluster_camera_pos))
    indices = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
    return indices

def select_diverse_images_per_cluster(tform_cam2world, labels):
    """
    Select the most diverse image from each cluster based on cosine similarity.
    """
    unique_labels = set(labels)
    selected_indices = []

    for label in unique_labels:

        if label == -1:
            continue  # Skip noise

        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        cluster_camera_pos = [camera_position_from_extrinsic_matrix(transform).detach().cpu().numpy() for transform in tform_cam2world[cluster_indices]]

        if len(cluster_indices) > 1:
            
            pairwise_distances = squareform(pdist(cluster_camera_pos))
            indices = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
            selected_indices.append(cluster_indices[indices[0]])
            selected_indices.append(cluster_indices[indices[1]])

        else:
            selected_indices.append(cluster_indices[0])

    return selected_indices

def get_box_vertices(pose: torch.Tensor, focal_length: float, height=100, width=100, near_thresh=2.0, far_thresh=6.0):
    """Compute the vertices of the bounding box (in 3D) defined by the near_thresh and far_thresh values."""
	 
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length, pose)
    near_points = ray_origins + ray_directions*near_thresh
    far_points = ray_origins + ray_directions*far_thresh
    box_vertices = torch.stack([near_points[0,0,:], near_points[-1,0,:], near_points[0,-1,:], near_points[-1,-1,:], 
    far_points[0,0,:], far_points[-1,0,:], far_points[0,-1,:], far_points[-1,-1,:]], dim=0)
	
    return box_vertices

def select_frames_based_on_strategy(images, focal_length, tform_cam2world, device, strategy = "random", k = 10):
  
  if strategy == "full":
    training_images = images
    training_tforms = tform_cam2world

  elif strategy == "random":

    indices = list(range(images.shape[0]))
    np.random.shuffle(indices)
    training_indices = indices[:k]
    training_images = images[training_indices]
    training_tforms = tform_cam2world[training_indices]

  elif strategy == "fvs":

    indices = list(range(images.shape[0]))
    np.random.shuffle(indices)
    training_indices = indices[:1]
    holdout_indices = indices[1:]
    # defin holdout here
    training_images = images[training_indices]
    training_tforms = tform_cam2world[training_indices]
    
    holdout_images = images[holdout_indices]
    holdout_tforms = tform_cam2world[holdout_indices]

    furthest_view_indices = furthest_view_sampling_k(training_tforms, holdout_tforms, k, device)
    training_images = torch.cat((training_images, holdout_images[furthest_view_indices]))
    training_tforms = torch.cat((training_tforms, holdout_tforms[furthest_view_indices]))

  elif strategy == "min_iou_3d":
    boxes = []
    for pose in tform_cam2world:
        boxes.append(get_box_vertices(pose, focal_length))
    boxes = torch.stack(boxes, dim=0)
    _, iou_3d = box3d_overlap(boxes, boxes)
    iou_3d_upper = torch.triu(iou_3d, diagonal=1)
    iou_3d_upper[iou_3d_upper==0] = 2.0
    _, top_indices = torch.topk(iou_3d_upper.flatten(), k//2, largest=False)
    min_iou_3d_indices = torch.cat((top_indices % 100, top_indices // 100), dim=0)
    training_images = images[min_iou_3d_indices]
    training_tforms = tform_cam2world[min_iou_3d_indices]

  elif strategy == "embedding":

    #best_dim_red_params = find_best_dim_red_params(images, embedding = "simple")
    embeddings = extract_features(images, device)
    #labels = DBSCAN(eps=0.7, min_samples=3).fit_predict(embeddings)
    #parameters = find_best_hdbscan_params(embeddings)
    labels = HDBSCAN(min_cluster_size=6, min_samples=2).fit_predict(embeddings)
    diverse_indices = select_diverse_images_per_cluster(tform_cam2world, labels)
    training_images = images[diverse_indices]
    training_tforms = tform_cam2world[diverse_indices]
  
  else:
    raise ValueError(f"Invalid strategy i.e. strategy = {strategy}. Valid strategies are: full, random, fvs, min_iou_3d, embedding")

  return training_images, training_tforms

def calculate_lpips(img1, img2, device):
  # Function to calculate LPIPS
  lpips = LPIPS(net_type='alex').to(device) # The model used is Alex (lightweight)
  img1 = img1.permute(2, 0, 1).unsqueeze(0).to(device)
  img2 = img2.permute(2, 0, 1).unsqueeze(0).to(device)
  return lpips(img1, img2).item()

# Function to calculate SSIM using torchvision
def calculate_ssim(img1, img2):
    return SSIM(img1, img2, channel_axis=-1).item()
