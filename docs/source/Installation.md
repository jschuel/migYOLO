# Installation

We have successully tested migYOLO with python 3.8, 3.9, 3.10, 3.11, and 3.12. As of June 12th, 2024, data is not yet available

## Instructions

1. Clone the migYOLO repository
```sh
git clone https://github.com/jschuel/migYOLO.git
```
or

```sh
git@github.com:jschuel/migYOLO.git
```

2. Set up a virtual environment with `python>=3.8`. If you use Anaconda, you can do something like

```sh
conda create -n migYOLO python=3.10
```
```sh
conda activate migYOLO
```

3. Navigate to the `migYOLO` parent directory where the `setup.py` file is and run

```sh
pip install -e .
```

This will install migYOLO and all necessary packages **except** PyTorch and Ultralytics' YOLOv8.

4. Follow the instructions on the installation guide of the front page of the [PyTorch website](https://pytorch.org/) to install pytorch. The "compute platform" row allows you to select the relevant platform for your GPU or "CPU only" if you do not have a compatible GPU

5. Install Ultralytics' YOLOv8 package with

```sh
pip install ultralytics
```

**Update (June 12th): Data is currently unavailable so these steps will not work. I anticipate data to be available within a week.**

6. Download `data_and_models.zip` (*link coming soon*) and move it to `migYOLO/migYOLO` (once we have a non Google Drive server to store our data, this step will be automated with `wget` statements in `setup_environment.sh`):
```sh
mv /path/to/data_and_models.zip /path/to/migYOLO/migYOLO/
```
make sure you replace both instances of `/path/to/` to the correct paths on your filesystem. `data_and_models.zip` contains the example image data used to get started with this package as well as two pretrained YOLO models called `base.pt` and `augment.pt`. These are the base and augment models described in the paper (**provide link when on arXiv**)

7. In `migYOLO/migYOLO` run `setup_environment.sh` with
```sh
source setup_environment.sh
```
Running this command will (1) unzip data_and_models.zip, (2) move all of its contents to their appropriate directories, and (3) remove data_and_models.zip

8. Navigate to the `tests/` directory and first run `python3 downsample.py` and then `python3 runYOLO.py`. If both of these run, then everything was installed correctly! If you run into issues please report them.