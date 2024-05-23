# migYOLO v1.0.0
migYOLO is a package containing tools for using YOLOv8 to process and perform object detection analyses on CMOS camera data from the MIGDAL experiment (**link to arXiv paper when up and add a gif of bounding boxes**). The package has GPU support through PyTorch.
### 1. Installation

a. Clone the repository
```sh
git clone https://github.com/jschuel/migYOLO.git
```
or
```sh
git@github.com:jschuel/migYOLO.git
```

b. Set up a virtual environment with `python>=3.8`. If you use Anaconda, you can do something like
```sh
conda create -n migYOLO python=3.10
```
```sh
conda activate migYOLO
```

c. Navigate to `migYOLO` where the `setup.py` file is and run
```sh
pip install -e .
```

This will install migYOLO and all necessary packages **except** PyTorch and Ultralytics' YOLOv8.

d. Follow the instructions on the installation guide of the front page of the [PyTorch website](https://pytorch.org/) to install pytorch. The "compute platform" row allows you to select the relevant platform for your GPU or "CPU only" if you do not have a compatible GPU

e. Install Ultralytics' YOLOv8 package with
```sh
pip install ultralytics
```

f. Download `zipped_files.zip` here (give link) and move it to `migYOLO/migYOLO` (once we have a non Google Drive server to store our data, this step will be automated with `wget` statements in `setup_environment.sh`):
```sh
mv /path/to/zipped_files.zip /path/to/migYOLO/migYOLO/
```
make sure you replace both instances of `/path/to/` to the correct paths on your filesystem. `zipped_files.zip` contains the example image data used to get started with this package as well as two pretrained YOLO models called `base.pt` and `augment.pt`. These are the base and augment models described in the paper (**provide link when on arXiv**)

g. In `migYOLO/migYOLO` run `setup_environment.sh` with
```sh
source setup_environment.sh
```
Running this command will (1) unzip zipped_files.zip, (2) move all of its contents to their appropriate directories, and (3) remove zipped_files.zip

h. Navigate to the `tests/` directory and first run `downsample.py` and then `runYOLO.py`. If both of these run, then everything was installed correctly! If you run into issues please report them.

# 2. Usage
