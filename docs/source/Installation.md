# Installation

We have successully tested migYOLO with python 3.8, 3.9, 3.10, 3.11, and 3.12. *Recommended:* Use python 3.10 or newer.

## Instructions

1. Clone the migYOLO repository
```sh
git clone https://github.com/jschuel/migYOLO.git
```
or

```sh
git clone git@github.com:jschuel/migYOLO.git
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

6. In `migYOLO/migYOLO` run `setup_environment.sh` with (*this step requires close to 10GB of disk space*)
```sh
source setup_environment.sh
```
Running this command will (1) fetch test data zip files associated with this package, (2) move all contents to their appropriate directories, and (3) remove the downloaded zip files

7. Navigate to the `tests/` directory and first run `python3 downsample.py` and then `python3 runYOLO.py`. If both of these run, then everything was installed correctly! If you run into issues please report them.