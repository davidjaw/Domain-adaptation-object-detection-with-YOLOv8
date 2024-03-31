# A simplified implementation of adversarial-based domain adaptation on object detection

## Introduction
This repo is a simplified, conceptual implementation of adversarial-based domain adaptation on object detection.
The core idea is similar with [Multidomain Object Detection Framework Using Feature Domain Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/10243073) that uses RoI align and multiscale feature fusion Discriminator, but the way of implementation is different.
Specifically, this repo uses Gradient Reverse Layer (GRL), therefore the training process is more straightforward without training policy; and the Teacher-student distillation is not implemented.

## Dataset
To run this, you have to download the [ACDC dataset](https://acdc.vision.ee.ethz.ch/), and run `scripts/dataset_preprocess.py` to generate the dataset in the format that this repo can use.
You can use `scripts/vis.py` to visualize the dataset and see if the conversion is correct.

## Dependencies
- Python 3.10
- PyTorch 2.0.1
- Torchvision 0.15.2

* Be aware of [this bug](https://github.com/facebookresearch/d2go/issues/650) if your PyTorch version is greater than mine.

## Result summary
Please do note that the result is not stable, and the result may vary even with the same random seed.

|   Name   | Trained Dataset | FOG (mAP50) | RAIN (mAP50) |
|:--------:|:---------------:|:-----------:|:------------:|
|   Fog    |       Fog       |    47.3     |     21.8     |
|   Rain   |      Rain       |    39.3     |     24.8     |
| FogRain  |   Fog + Rain    |    50.3     |     28.0     |
| Proposed |   Fog + Rain    |    54.1     |     30.5     |

## BibTeX
If you find this repo helpful, please consider citing the original paper:
```
@ARTICLE{10243073,
  author={Jaw, Da-Wei and Huang, Shih-Chia and Lu, Zhi-Hui and Fung, Benjamin C. M. and Kuo, Sy-Yen},
  journal={IEEE Transactions on Cybernetics}, 
  title={Multidomain Object Detection Framework Using Feature Domain Knowledge Distillation}, 
  year={2023},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/TCYB.2023.3300963}
}
```

Note that due to patent constraints and collaboration agreements, the original code and dataset described in the paper will not be publicly available. This repository offers a conceptual approximation using PyTorch, simplified for educational and research purposes. It does not implement the core training policies detailed in the paper. This approach ensures compliance with existing agreements while facilitating understanding and exploration of the concepts discussed.

## Acknowledgement
Special thanks to Ultralytics for their exceptional open-source contributions, which have significantly enhanced this project.
Visit [Ultralytics on GitHub](https://github.com/ultralytics/ultralytics) for more on their impactful work.

