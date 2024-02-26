# !pip3 -q install umap-learn
# !pip3 -q install hdbscan
# !pip3 -q install lpips
# !pip3 -q install torchmetrics
# !pip3 -q install optuna

# Import all the good stuff
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ## A few utility functions

from model import VeryTinyNerfModel
from data_utils import load_tiny_nerf_data, load_nerf_data, get_focal_length
from nerf_utils import positional_encoding, get_minibatches, run_one_iter_of_tinynerf
from utils import select_frames_from_baseline, calculate_lpips, calculate_ssim, cluster_poses_by_kmeans, plot_images_as_grid
import argparse
import time


# Parse command line arguments
parser = argparse.ArgumentParser(description='k=10 Clusters, num_iters=200 iterations')
parser.add_argument('k', type=int, default=10, help='Number of clusters')
parser.add_argument('num_iters', type=int, default=200, help='iterations for training the model')
parser.add_argument('dataset', type=str, default='lego', help='Dataset to use')

args = parser.parse_args()
k, num_iters = args.k, args.num_iters
nerf_test_subject = args.dataset

print(f'Running baseline with k={k}, num_iters={num_iters}, dataset={nerf_test_subject}')

experiment_data_type = "baseline"
nerf_subjects_list = ['ship', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'chair']

tiny_data_path = "data/tiny_nerf_data.npz"
nerf_synthetic_path = "data/nerf_synthetic"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parent_dir = 'logs_test'
if experiment_data_type == "tiny_nerf":
    logs_path = f"{parent_dir}/{experiment_data_type}"
else:
    logs_path = f"{parent_dir}/{experiment_data_type}/{nerf_test_subject}_k{k}"

execution_time_file_path = f'logss/{experiment_data_type}/execution_time.csv'

paths = [f'{logs_path}/models', f'{logs_path}/results']

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)


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

    ###
    # K-means clustering
    ###

    # Drop the rotation matrix R from the extrinsic matrix [R | t]
    camera_poses_intrinsic = train_poses[:, :3, 3]
    # It leaves us with (100, 0, 3) shape
    camera_poses_intrinsic = camera_poses_intrinsic.reshape((camera_poses_intrinsic.shape[0], 3))

    
    title = f'Clustered Intrinsic Positions ({nerf_test_subject} k={k})'
    save_path = f'clustered_intrinsic_{nerf_test_subject}_k{k}.png'
    selected_images_indices = cluster_poses_by_kmeans(camera_poses_intrinsic, title, logs_path, n_clusters=k, save_path=save_path)
    # plot_images_as_grid(images[selected_images_indices], logs_path, f'output_intrinsic_{nerf_test_subject}_k{k}.png')

    ### K-means clustering end

    focal_length = get_focal_length(width, camera_angle_x)
    focal_length = torch.from_numpy(focal_length)

    # Hold one image out (for test).
    testimgs, testposes, _ = load_nerf_data(nerf_synthetic_path, nerf_test_subject, 'transforms_test.json', skip_images=20)

    testimgs = testimgs.to(device)
    testposes = torch.from_numpy(testposes).to(device)

    # # Map images to device
    images = images.to(device)
    tform_cam2world = torch.from_numpy(tform_cam2world[:100,]).to(device)

# #### Display the image used for testing


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

# ## **Tiny NeRF Parameters**

# Number of functions used in the positional encoding (Be sure to update the
# model if this number changes).
num_encoding_functions = 6
# Specify encoding function.
encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
# Number of depth samples along each ray.
depth_samples_per_ray = 32

# Chunksize (Note: this isn't batchsize in the conventional sense. This only
# specifies the number of rays to be queried in one go. Backprop still happens
# only after all rays from the current "bundle" are queried and rendered).
chunksize = 16384  # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.

# Optimizer parameters
lr = 5e-3

# Misc parameters
display_every = 1  # Number of iters after which stats are displayed

model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Seed RNG, for repeatability
seed = 9458
torch.manual_seed(seed)
np.random.seed(seed)

# Lists to log metrics etc.
psnr_vals_dict = {}
lpips_vals_dict = {}
ssim_vals_dict = {}

strategies = ["camera_intrinsic", "random", "fvs"]
# strategies = ["camera_intrinsic", "random", "fvs", "min_iou_3d"]

for strategy in strategies:
    
    start_time = time.time()

    psnr_vals, lpips_vals, ssim_vals = [], [], []
    training_images, training_tforms = select_frames_from_baseline(images, focal_length, tform_cam2world, strategy ,k, indices = selected_images_indices)
    
    model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if strategy != "full":
        print(f'Strategy: {strategy} - Training images utilized')
        fig, axs = plt.subplots(2, 5, figsize=(12, 6))
        # Loop through all axes and images
        for ax, image in zip(axs.ravel(), training_images):
            ax.imshow(image.detach().cpu().numpy(), cmap='viridis')  # You can change the colormap if needed
            ax.axis('off')  # Turn off axis

        plt.tight_layout()
        if experiment_data_type == "tiny_nerf":
            plt.savefig(f"{logs_path}/chosen_frames_using_{strategy}_strategy.png")
        else:
            #create folder for nerf test subject
            if not os.path.exists(f"{logs_path}"):
                os.makedirs(f"{logs_path}")
            plt.savefig(f"{logs_path}/chosen_frames_using_{strategy}_strategy.png")
        # plt.show()

    for j in range(num_iters):

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

        psnr_vals.append(np.average(avg_psnr))
        lpips_vals.append(np.average(avg_lpips))
        ssim_vals.append(np.average(avg_ssim))

    psnr_vals_dict[strategy] = psnr_vals
    lpips_vals_dict[strategy] = lpips_vals
    ssim_vals_dict[strategy] = ssim_vals
    torch.save(model.state_dict(), f"{logs_path}/models/model_{nerf_test_subject}_k{k}_{strategy}.pth")
    
    # Calcualte the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Save execution time to a CSV file
    if not os.path.exists(execution_time_file_path):
        data = {'subject': [nerf_test_subject], 'strategy': [strategy], 'num_iters': [num_iters], 'k': [k], 'execution_time': [execution_time]}
        df = pd.DataFrame(data)
        df.to_csv(execution_time_file_path, index=False)
    else:
        data = {'subject': nerf_test_subject, 'strategy': strategy, 'num_iters': num_iters, 'k': k, 'execution_time': [execution_time]}
        df = pd.DataFrame(data)
        df.to_csv(execution_time_file_path, mode='a', header=False, index=False)

np.save(f'{logs_path}/results/psnr_vals_dict_baseline_{nerf_test_subject}_k{k}.npy', psnr_vals_dict)
np.save(f'{logs_path}/results/lpips_vals_dict_baseline_{nerf_test_subject}_k{k}.npy', lpips_vals_dict)
np.save(f'{logs_path}/results/ssim_vals_dict_baseline_{nerf_test_subject}_k{k}.npy', ssim_vals_dict)


