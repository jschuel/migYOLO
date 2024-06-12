.. migYOLO documentation master file, created by
   sphinx-quickstart on Wed Jun 12 12:12:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to migYOLO's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction
   Installation
   Usage

migYOLO v1.0.0
===============

migYOLO is a package containing tools for using the YOLOv8-based processing and rare event search analysis pipeline for CMOS camera data from the MIGDAL experiment (`paper preprint here <https://arxiv.org/abs/2406.07538>`__). This package has GPU support through PyTorch.

.. image:: ../../migYOLO/figures/object_detect.gif
   :alt: Object detection
   :align: center

Getting started (successfully tested with Python 3.8, 3.9, 3.10, 3.11, and 3.12)
-----------------------------------------------------------------------------

a. Clone the repository::

   git clone https://github.com/jschuel/migYOLO.git

   or

   git@github.com:jschuel/migYOLO.git

b. Set up a virtual environment with `python>=3.8`. If you use Anaconda, you can do something like::

   conda create -n migYOLO python=3.10
   conda activate migYOLO

c. Navigate to `migYOLO` where the `setup.py` file is and run::

   pip install -e .

   This will install migYOLO and all necessary packages **except** PyTorch and Ultralytics' YOLOv8.

d. Follow the instructions on the installation guide of the front page of the `PyTorch website <https://pytorch.org/>`__ to install PyTorch. The "compute platform" row allows you to select the relevant platform for your GPU or "CPU only" if you do not have a compatible GPU.

e. Install Ultralytics' YOLOv8 package with::

   pip install ultralytics

f. Download `data_and_models.zip` (link coming soon) and move it to `migYOLO/migYOLO` (once we have a non-Google Drive server to store our data, this step will be automated with `wget` statements in `setup_environment.sh`):::

   mv /path/to/data_and_models.zip /path/to/migYOLO/migYOLO/

Make sure you replace both instances of `/path/to/` with the correct paths on your filesystem. `data_and_models.zip` contains the example image data used to get started with this package as well as two pretrained YOLO models called `base.pt` and `augment.pt`. These are the base and augment models described in the `paper <https://arxiv.org/abs/2406.07538>`__.

g. In `migYOLO/migYOLO` run `setup_environment.sh` with::

   source setup_environment.sh

   Running this command will (1) unzip `data_and_models.zip`, (2) move all of its contents to their appropriate directories, and (3) remove `data_and_models.zip`.

h. Navigate to the `tests/` directory and first run `python3 downsample.py` and then `python3 runYOLO.py`. If both of these run, then everything was installed correctly! If you run into issues, please report them.

Usage (Coming soon)
----------------------

Usage instructions will be available soon.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
