# migYOLO v1.0.0
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12628437.svg)](https://doi.org/10.5281/zenodo.12628437)

**Update (June 24th, 2024): Data is available (see Installation instructions in the [official documentation](https://migyolo.readthedocs.io/en/latest/))**

migYOLO is a package containing tools for using the YOLOv8-based processing and rare event search analysis pipeline for CMOS camera data from the MIGDAL experiment ([paper preprint here](https://arxiv.org/abs/2406.07538)). This package has GPU support through PyTorch. For installation and usage instructions, please consult the [official documentation](https://migyolo.readthedocs.io/en/latest/).

![Object detection](migYOLO/figures/object_detect.gif)

## How to Cite

If you have found this software useful for your work, please consider citing both our paper and this software:

### Paper Citation
```bibtex
@article{MIGDAL:2024alc,
    author = "Schueler, J. and others",
    collaboration = "MIGDAL",
    title = "{Transforming a rare event search into a not-so-rare event search in real-time with deep learning-based object detection}",
    eprint = "2406.07538",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    month = "6",
    year = "2024"
}
```

### Software Citation
```bibtex
@software{schueler_2024_12628437,
  author       = {Schueler, Jeffrey},
  title        = {{migYOLO - An end-to-end YOLOv8-based object 
                   detection pipeline for CMOS camera images recorded
                   by the MIGDAL experiment}},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.12628437},
  url          = {https://doi.org/10.5281/zenodo.12628437}
}
```
