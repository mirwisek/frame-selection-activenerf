#
# Installing pytorch3d can be tricky on some OS
# To install we could run:
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
#
# But to get the actual version we need to run the current script to extract part of the url
# "py38_cu113_pyt1110" which is printing after running this python file
#

import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
print(version_str)