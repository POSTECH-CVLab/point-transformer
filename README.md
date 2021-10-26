# Point-Transformer PyTorch

* This is an unofficial implemention of [Point Transformer](https://arxiv.org/abs/2012.09164).
* We use [PAConv repo](https://github.com/CVMI-Lab/PAConv) which the first author of [Point Transformer](https://arxiv.org/abs/2012.09164) has participated in for the codebase.
* For k-nearest neighbor search with heap sort, please refer [PAConv repo](https://github.com/CVMI-Lab/PAConv).


## Updates

* [2021.10.26] Add semantic segmentation on S3DIS Area 5 with the new codebase, [PAConv repo](https://github.com/CVMI-Lab/PAConv).


## Setup

* The code is tested on Ubuntu 20.04 and CUDA 11.1.
* Install dependencies

  `conda env create --file environment.yaml`


## Training & Evaluation

Please check directories for classification, part segmentation, and scene segmentation.


## Experiment Results

- Classification on ModelNet40

|Model | mAcc | OA |
|-------| ------| -------|
|Paper| 90.6 | 93.7 |
|Our implemention |  |  |

- Part Segmentation on ShapeNet

|Model | cat. mIoU | ins. mIoU |
|-------| ------| -------|
|Paper| 83.7 | 86.6 |
|Our implemention |  |  |


- Semantic Segmentation on S3DIS Area5

|Model | mAcc | OA | mIoU |
|-------| ------| ----| -------|
|Paper| 76.5 | 90.8 | 70.4 |
|Our implemention |  |  |  |


## Acknowledment
Our code is heavily based on [PAConv repo](https://github.com/CVMI-Lab/PAConv).


## Citations

If you use this code, please cite [Point Transformer](https://arxiv.org/abs/2012.09164) and [PAConv](https://arxiv.org/abs/2103.14635):
```
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```
```
@inproceedings{xu2021paconv,
  title={PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds},
  author={Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan},
  booktitle={CVPR},
  year={2021}
}
```