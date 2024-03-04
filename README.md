# Frame Selection: 3D Scene Reconstruction Project

The objective of the project is to select frames that maximize pose **DIVERSITY** and help NeRF generalize effectively under **constrained** input budget.

## Dataset

In our experiments we utlize `nerf_synthetic` [dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1?usp=sharing) from NeRF that comprises of 8 subjects.

## How to work with this code

### Installing dependencies
While most of the dependencies can be installed with pip `requirements.txt` file but installing `pytorch3d` can be problematic on some systems which require that the version of cuda, python and pytorch itself be compatible. Our setup consisted of the following versions:

- Python 3.9
- Cuda 11.8
- Pytorch 2.1.2

Once the above versions are setup, run the `utils > pytorch3d_version.py` to extract part of the url and substitute in the url
`pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/<replace-str-here>/download.html`

### Running on Ruche server
The shell scripts under `jobs_scripts` directory can be utilized to schedule jobs on Ruche server. `script_baseline.sh` runs the baseline models while `script_clustering.sh` runs the HDBScan experiments with Optuna on whole NeRF dataset.

### Project structure
The two main important files are:
- `baseline_models.py` runs baseline models on the NeRF dataset configured with $k = [5, 10, 15, 20, 25]$
- `clustering_script.py` runs HDBScan on the NeRF dataset configured with $k = [5, 10, 15, 20, 25]$
Upon running, these file store results in `logs` directory mostly in `.npz` files. The results including the metrics and graphs need to be explicitly plotted using `plot_logs.ipynb`.



