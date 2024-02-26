
# %%
# !pip3 -q install umap-learn
# !pip3 -q install hdbscan
# !pip3 -q install lpips
# !pip3 -q install torchmetrics
# !pip3 -q install optuna

# %%
# Import all the good stuff
import os
import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## A few utility functions

# %%
from model import VeryTinyNerfModel
from image_encoder import ImageEncoder
from data_utils import load_tiny_nerf_data, load_nerf_data, get_focal_length
from nerf_utils import positional_encoding, get_minibatches, run_one_iter_of_tinynerf
from utils import select_frames_from_baseline, select_frames_from_clustering, calculate_lpips, calculate_ssim

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--num_clusters", type=int, default=10, help="Number of clusters")
args = parser.parse_args()

k = args.num_clusters

# %%
experiment_data_type = "tiny_nerf"
#experiment_data_type = "nerf_synthetic"
nerf_test_subject = "ship" if experiment_data_type == "nerf_synthetic" else "lego" 
nerf_subjects_list = ['ship', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic']

tiny_data_path = "data/tiny_nerf_data.npz"
nerf_synthetic_path = "data/nerf_synthetic"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if experiment_data_type == "tiny_nerf":
    logs_path = f"logs/{experiment_data_type}_k{k}"
else:
    logs_path = f"logs/{experiment_data_type}_k{k}/{nerf_test_subject}"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

# %% [markdown]
# ## Get data

# %%
# Download sample data used in the official tiny_nerf example
if not os.path.exists(tiny_data_path):
    wget -P data/ http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz

# %% [markdown]
# ## Load up input images, poses, intrinsics, etc.

# %%
# Near and far clipping thresholds for depth values.
near_thresh = 2.
far_thresh = 6.

# Height and width of each image
height, width = 100, 100

if experiment_data_type == "tiny_nerf":
    images, tform_cam2world, focal_length, testimgs, testposes = load_tiny_nerf_data(tiny_data_path, device)
else:
    images, train_poses, camera_angle_x = load_nerf_data(nerf_synthetic_path, nerf_test_subject, 'transforms_train.json')
    tform_cam2world = train_poses

    focal_length = get_focal_length(width, camera_angle_x)
    focal_length = torch.from_numpy(focal_length).to(device)

    # Hold one image out (for test).
    testimgs, testposes, _ = load_nerf_data(nerf_synthetic_path, nerf_test_subject, 'transforms_test.json', skip_images=20)

    testimgs = testimgs.to(device)
    testposes = torch.from_numpy(testposes).to(device)

    # # Map images to device
    images = images.to(device)
    tform_cam2world = torch.from_numpy(tform_cam2world[:100,]).to(device)

# %% [markdown]
# #### Display the image used for testing

# %%
if experiment_data_type == "tiny_nerf":
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
else:
    fig, axs = plt.subplots(2, 5, figsize=(10, 6))

# Loop through all axes and images
for ax, image in zip(axs.ravel(), testimgs):
    ax.imshow(image.detach().cpu().numpy(), cmap='viridis')  # You can change the colormap if needed
    ax.axis('off')  # Turn off axis

plt.tight_layout()
if experiment_data_type == "tiny_nerf":
    plt.savefig(f"{logs_path}/test_images.png")
else:
    #create folder for nerf test subject
    if not os.path.exists(f"{logs_path}"):
        os.makedirs(f"{logs_path}")
    plt.savefig(f"{logs_path}/test_images_nerf.png")
plt.show()

# %%
psnr_vals_dict = {}
lpips_vals_dict = {}
ssim_vals_dict = {}



# %%
def objective(trial, images, embeddings, tform_cam2world):
    
    # tiny nerf parameters - not being changed
    lr = 5e-3
    num_iters = 200
    chunksize = 16384
    num_encoding_functions = 6
    depth_samples_per_ray = 32
    encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)

    # parameters fixed at the moment
    strategy = "fvs_distance"

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # dimensionality reduction params to be tuned
    n_neighbors = trial.suggest_int("n_neighbors", 2, 50)
    min_dist = trial.suggest_float("min_dist", 0.01, 0.99)
    n_components = trial.suggest_int("n_components", 5, 50)

    # hdbscan clustering algorithm params to be tuned
    min_cluster_size = trial.suggest_int("min_cluster_size", 2, 10)
    min_samples = trial.suggest_int("min_samples", 2, 10)

    # initialize model and optimizer
    model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # make a string for the trial name based on the parameters
    trial_name = f"n_neighbors_{n_neighbors}_min_dist_{min_dist}_n_components_{n_components}_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}"
    
    try:
        diverse_indices = select_frames_from_clustering(embeddings, tform_cam2world, focal_length, 
                                                    device=device, strategy=strategy, k=k, 
                                                    dim_red_method="umap", umap_params=[n_neighbors, min_dist, n_components],
                                                    min_cluster_size=min_cluster_size, min_samples=min_samples)
    except Exception as e:
        print('Skipping trial due to error: ', e)
        return 0
    
    training_images = images[diverse_indices]
    training_tforms = tform_cam2world[diverse_indices]
    psnr_vals, lpips_vals, ssim_vals = [], [], []

    for j in range(num_iters+1):

        # Randomly pick an image as the target.
        for training_img, training_pose in zip(training_images, training_tforms):
            
            # Run one iteration of TinyNeRF and get the rendered RGB image.
            rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                                    training_pose, near_thresh,
                                                    far_thresh, depth_samples_per_ray,
                                                    encode, get_minibatches, chunksize, model)

            # Compute mean-squared error between the predicted and target images. Backprop!
            loss = torch.nn.functional.mse_loss(rgb_predicted, training_img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_psnr, avg_lpips, avg_ssim = [], [], []
        with torch.no_grad():
            for testimg, testpose in zip(testimgs, testposes):
                avg_psnr = []
                rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                                        testpose, near_thresh,
                                                        far_thresh, depth_samples_per_ray,
                                                        encode, get_minibatches, chunksize, model)
                loss = torch.nn.functional.mse_loss(rgb_predicted, testimg)
                psnr = -10. * torch.log10(loss)
                avg_psnr.append(psnr.item())
                lpips_loss = calculate_lpips(rgb_predicted, testimg, device)
                avg_lpips.append(lpips_loss)
                rgb_predicted_cpu = rgb_predicted.cpu().detach().numpy()
                testimg_cpu = testimg.cpu().detach().numpy()
                ssim_loss = calculate_ssim(rgb_predicted_cpu, testimg_cpu)
                avg_ssim.append(ssim_loss)
        
        psnr_avg = np.average(avg_psnr)
        trial.report(psnr_avg, j)
        psnr_vals.append(psnr_avg)
        lpips_vals.append(np.average(avg_lpips))
        ssim_vals.append(np.average(avg_ssim))

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    psnr_vals_dict[trial_name] = psnr_vals
    lpips_vals_dict[trial_name] = lpips_vals
    ssim_vals_dict[trial_name] = ssim_vals

    if not os.path.exists(f"{logs_path}/models"):
        os.makedirs(f"{logs_path}/models")
    torch.save(model.state_dict(), f"{logs_path}/models/{trial_name}.pth")

    if not os.path.exists(f"{logs_path}/results"):
        os.makedirs(f"{logs_path}/results")
    np.save(f"{logs_path}/results/psnr_vals_dict_clustering.npy", psnr_vals_dict)
    np.save(f"{logs_path}/results/lpips_vals_dict_clustering.npy", lpips_vals_dict)
    np.save(f"{logs_path}/results/ssim_vals_dict_clustering.npy", ssim_vals_dict)
    
    return psnr_avg

# %%
image_encoder = ImageEncoder(device)
embeddings = image_encoder(images)

# %%
study = optuna.create_study(study_name=f'{experiment_data_type}-{nerf_test_subject}', direction="maximize")
study.optimize(lambda trial: objective(trial, images, embeddings, tform_cam2world), n_trials=100, timeout=18000)

pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%


# %%


# %%


# %%


def calculate_iou_3d(tform1, tform2, focal_length):
    boxes1 = get_box_vertices(tform1, focal_length)
    boxes2 = get_box_vertices(tform2, focal_length)
    vol, _ = box3d_overlap(boxes1.unsqueeze(0), boxes2.unsqueeze(0))
    return vol[0].item()


def maximal_coverage(tform_cam2world, focal_length, k):
    n = tform_cam2world.shape[0]
    
    # Initialize DP table
    DP = torch.zeros((n + 1, k + 1))
    
    # Build DP table
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            DP[i][j] = float('inf')
            for l in range(i):
                DP[i][j] = min(DP[i][j], DP[l][j - 1] + calculate_iou(tform_cam2world, focal_length, i - 1, l))

    # Find the maximum coverage for k images
    # max_coverage = max(DP[n][k])

    # Traceback to find the selected indices
    selected_indices = []
    i, j = n, k
    while i > 0 and j > 0:
        if DP[i][j] != DP[i - 1][j]:
            selected_indices.append(i - 1)
            j -= 1
        i -= 1

    return DP, selected_indices

# %%


# %%


# %%


# %%



