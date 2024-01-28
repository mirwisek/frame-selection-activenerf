import umap
import torch
import numpy as np
from umap import umap_
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

def furthest_view_sampling_k(tform_cam2world, k = 10, seed = 9458):

    # shuffle indices of tform_cam2world multiple times
    indices = np.arange(tform_cam2world.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)

    # take 1 random sample for trainnig and rest for holdout
    training_indices = indices[:1]
    holdout_indices = indices[1:]

    train_tform_c2w = tform_cam2world[training_indices]
    holdout_tform_c2w = tform_cam2world[holdout_indices]

    # Calculate camera positions for the training set and holdout
    training_positions = [camera_position_from_extrinsic_matrix(transform) for transform in train_tform_c2w]
    holdout_position = [camera_position_from_extrinsic_matrix(transform) for transform in holdout_tform_c2w]
    training_positions = torch.stack(training_positions)
    holdout_position_orig = torch.stack(holdout_position)
    holdout_position = torch.stack(holdout_position)

    # Initialize a list to hold the furthest candidates
    furthest_candidates = []

    # Calculate the distance of each candidate to all camera positions in the training set   
    while len(furthest_candidates) < k:
      
      avg_train_pos = torch.mean(training_positions, dim=0)
      euclidean_distance = torch.sqrt(torch.sum((holdout_position - avg_train_pos) ** 2, dim=1))
      max_index = torch.argmax(euclidean_distance).item()
      max_element = holdout_position[max_index]
      holdout_position = torch.cat((holdout_position[:max_index], holdout_position[max_index+1:]))
      max_element_index = torch.where(torch.all(torch.eq(holdout_position_orig, max_element), dim=1))[0].item()
      furthest_candidates.append(max_element_index)
      training_positions = torch.cat((training_positions, torch.unsqueeze(max_element, 0)))
      
    return furthest_candidates

def select_frames_using_min_3d_iou_greedy(tform_cam2world, focal_length, k = 10, seed = 9458):

    # shuffle indices of tform_cam2world multiple times
    indices = np.arange(tform_cam2world.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)

    # take 1 random sample for trainnig and rest for holdout
    training_indices = indices[:1]
    holdout_indices = indices[1:]

    # get box vertices for training and holdout
    training_tforms = tform_cam2world[training_indices].squeeze()
    holdout_tforms = tform_cam2world[holdout_indices].squeeze()
    training_box = get_box_vertices(training_tforms.squeeze(), focal_length)
    holdout_boxes = torch.stack([get_box_vertices(tform.squeeze(), focal_length) for tform in holdout_tforms])

    # calculate 3d iou between training and holdout 
    _, iou_3d = box3d_overlap(training_box.unsqueeze(0), holdout_boxes)
    min_iou_index = torch.argmin(iou_3d.squeeze()).item()

    # add new index with minimum 3d iou to training indices
    training_indices = np.append(training_indices, holdout_indices[min_iou_index])
    holdout_indices = np.delete(holdout_indices, min_iou_index)
        
    for i in range(k-2):

        # get new poses based on minimum 3d iou
        training_tforms = tform_cam2world[training_indices].squeeze()
        holdout_tforms = tform_cam2world[holdout_indices].squeeze()
        
        # get box vertices for training and holdout
        training_boxes = torch.stack([get_box_vertices(tform.squeeze(), focal_length) for tform in training_tforms])
        holdout_boxes = torch.stack([get_box_vertices(tform.squeeze(), focal_length) for tform in holdout_tforms])

        # calculate 3d iou between training and holdout
        _, iou_3d = box3d_overlap(training_boxes, holdout_boxes)
        
        # get holdout index with minimum 3d iou value for each training box
        min_iou_values, min_iou_indices = torch.min(iou_3d, dim=1) # torch.Size([training_boxes])

        # get next holdout index with minimum 3d iou value
        min_iou_value_idx = torch.argmin(min_iou_values).item()
        min_iou_indices_index = min_iou_indices[min_iou_value_idx].item() # torch.Size([1])
        
        # add new index with minimum 3d iou to training indices
        training_indices = np.append(training_indices, holdout_indices[min_iou_indices_index])
        holdout_indices = np.delete(holdout_indices, min_iou_indices_index)
    
    return training_indices

def extract_features(embeddings, dim_red_method = "pca", pca_dims = 5, umap_params=[]):

  image_embd = embeddings.squeeze().contiguous().view(embeddings.size(0), -1).cpu().detach().numpy()

  # Perform PCA for dimensionality reduction
  if dim_red_method == "pca":
    pca_reducer = PCA(n_components=pca_dims)
    reduced_img_embd = pca_reducer.fit_transform(image_embd)
  elif dim_red_method == "umap":
    umap_reducer = umap_.UMAP(n_neighbors=umap_params[0], min_dist=umap_params[1], n_components=umap_params[2])
    reduced_img_embd = umap_reducer.fit_transform(image_embd)
  else:
    raise ValueError(f"Invalid dimension reduction method. Valid methods are: pca, umap")

  return reduced_img_embd

def select_diverse_images_per_cluster(tform_cam2world, labels, strategy, focal_length, k):
    """
    Select the most diverse image from each cluster based on cosine similarity.
    """
    unique_labels = set(labels)
    selected_indices = []

    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_clusters = 1 if n_clusters == 0 else n_clusters

    items_per_cluster = k // n_clusters
    items_per_cluster = 1 if items_per_cluster == 0 else items_per_cluster

    for label in unique_labels:

        if label == -1:
            continue  # Skip noise

        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]

        if len(cluster_indices) > 1:       
            if strategy == "fvs_distance":
                cluster_camera_pos = [camera_position_from_extrinsic_matrix(transform).detach().cpu().numpy() for transform in tform_cam2world[cluster_indices]] 
                torch_cc_pos = torch.tensor(cluster_camera_pos)
                pairwise_distances = torch.cdist(torch_cc_pos, torch_cc_pos, p=2.0)
                pairwise_distances_upper = torch.triu(pairwise_distances, diagonal=1)
                pairwise_distances_upper[pairwise_distances_upper==0] = -torch.inf
                _, top_indices = torch.topk(pairwise_distances_upper.flatten(), items_per_cluster, largest=True)
                max_pairwise_distances_indices = torch.cat((top_indices // len(cluster_indices), top_indices % len(cluster_indices)), dim=0)
                unique_indices = torch.unique(max_pairwise_distances_indices)
                selected_indices.extend([cluster_indices[idx] for idx in unique_indices])
            
            elif strategy == "fvs_iou_3d":
                boxes = []
                for idx in cluster_indices:
                    boxes.append(get_box_vertices(tform_cam2world[idx], focal_length))
                boxes = torch.stack(boxes, dim=0)
                _, iou_3d = box3d_overlap(boxes, boxes)
                iou_3d_upper = torch.triu(iou_3d, diagonal=1)
                iou_3d_upper[iou_3d_upper==0] = 2.0
                _, top_indices = torch.topk(iou_3d_upper.flatten(), items_per_cluster, largest=False)
                min_iou_3d_indices = torch.cat((top_indices // len(cluster_indices), top_indices % len(cluster_indices)), dim=0)
                unique_indices = torch.unique(min_iou_3d_indices)
                selected_indices.extend([cluster_indices[idx] for idx in unique_indices])
                
        else:
            selected_indices.append(cluster_indices[0])
        
    return selected_indices[:k]

def get_box_vertices(pose: torch.Tensor, focal_length: float, height=100, width=100, near_thresh=2.0, far_thresh=6.0):
    """Compute the vertices of the bounding box (in 3D) defined by the near_thresh and far_thresh values."""
	 
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length, pose)
    near_points = ray_origins + ray_directions*near_thresh
    far_points = ray_origins + ray_directions*far_thresh
    box_vertices = torch.stack([near_points[0,0,:], near_points[-1,0,:], near_points[0,-1,:], near_points[-1,-1,:], 
    far_points[0,0,:], far_points[-1,0,:], far_points[0,-1,:], far_points[-1,-1,:]], dim=0)
	
    return box_vertices

def select_frames_from_baseline(images, focal_length, tform_cam2world, baseline = "random", k = 10, seed = 9458):
  
  if baseline == "full":
    training_images = images
    training_tforms = tform_cam2world

  elif baseline == "random":

    indices = list(range(images.shape[0]))
    np.random.seed(seed)
    np.random.shuffle(indices)
    training_indices = indices[:k]
    training_images = images[training_indices]
    training_tforms = tform_cam2world[training_indices]

  elif baseline == "fvs":

    furthest_view_indices = furthest_view_sampling_k(tform_cam2world, k, seed)
    training_images = images[furthest_view_indices]
    training_tforms = tform_cam2world[furthest_view_indices]

  elif baseline == "min_iou_3d":
      
      min_iou_3d_indices = select_frames_using_min_3d_iou_greedy(tform_cam2world, focal_length, k, seed)
      training_images = images[min_iou_3d_indices]
      training_tforms = tform_cam2world[min_iou_3d_indices]

  else:
    raise ValueError(f"Invalid strategy i.e. baseline = {baseline}. Valid strategies are: full, random, fvs, min_iou_3d, embedding")

  return training_images, training_tforms

def select_frames_from_clustering(embeddings, tform_cam2world, focal_length, **kwargs):
  # Get the embeddings

  embeddings = extract_features(embeddings,
                                dim_red_method = kwargs.get("dim_red_method","pca"), 
                                umap_params=kwargs.get("umap_params", [5, 0.1, 5]))

  # Get the labels
  labels = HDBSCAN(min_cluster_size=kwargs.get("min_cluster_size",6),
                   min_samples=kwargs.get("min_samples",2)).fit_predict(embeddings)

  # Select the frames
  diverse_indices = select_diverse_images_per_cluster(tform_cam2world, labels, 
                                                      kwargs.get("strategy","fvs_distance"), 
                                                      focal_length, 
                                                      kwargs.get("k", 10))

  return diverse_indices

def calculate_lpips(img1, img2, device):
  # Function to calculate LPIPS
  lpips = LPIPS(net_type='alex').to(device) # The model used is Alex (lightweight)
  img1 = img1.permute(2, 0, 1).unsqueeze(0).to(device)
  img2 = img2.permute(2, 0, 1).unsqueeze(0).to(device)
  return lpips(img1, img2).item()

# Function to calculate SSIM using torchvision
def calculate_ssim(img1, img2):
    return SSIM(img1, img2, channel_axis=-1).item()
