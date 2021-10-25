# 3D Semantic Segmentation

## Installation

### Requirements
   - Hardware: 1 GPU
   - Software: 
      PyTorch>=1.5.0, Python>=3, CUDA>=10.2, tensorboardX, tqdm, h5py, pyYaml

### Dataset
- Download S3DIS [dataset](https://drive.google.com/drive/folders/12wLblskNVBUeryt1xaJTQlIoJac2WehV) and symlink the paths to them as follows (you can alternatively modify the relevant paths specified in folder `config`):
    ```
     mkdir -p dataset
     ln -s /path_to_s3dis_dataset dataset/s3dis
     ```

## Usage

1. Requirement:

   - Hardware: 2 GPUs to hold 10000MB
   - Software: 
      PyTorch>=1.5.0, Python3.7, CUDA>=10.2, tensorboardX, tqdm, h5py, pyYaml

2. Train:

   - Specify the gpu used in config and then do training:

     ```shell
     sh tool/train.sh s3dis point_transformer
     ```

3. Test:

   - For full testing (get listed performance):

     ```shell
     CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis point_transformer
     ```
    
   - For 6-fold validation (calculating the metrics with results from different folds merged): 
     1) Change the [test_area index](https://github.com/POSTECH-CVLab/point-transformer/blob/main/scene_seg/config/s3dis/s3dis_point_transformer.yaml#L7) in the config file to 1;
     2) Finish full train and test, the test result files of Area-1 will be saved in corresponding paths after the test;
     3) Repeat a,b by changing the [test_area index](https://github.com/POSTECH-CVLab/point-transformer/blob/main/scene_seg/config/s3dis/s3dis_point_transformer.yaml#L7) to 2,3,4,5,6 respectively;
     4) Collect all the test result files of all areas to one directory and state the path to this directory [here](https://github.com/POSTECH-CVLab/point-transformer/blob/main/scene_seg/tool/test_s3dis_6fold.py#L52);
     5) Run the code for 6-fold validation to get the final 6-fold results:
        ```shell
        python test_s3dis_6fold.py
        ```
        
    
   
[comment]: <> (5. Visualization: [tensorboardX]&#40;https://github.com/lanpa/tensorboardX&#41; incorporated for better visualization.)

[comment]: <> (   ```shell)

[comment]: <> (   tensorboard --logdir=run1:$EXP1,run2:$EXP2 --port=6789)

[comment]: <> (   ```)


[comment]: <> (6. Other:)

[comment]: <> (   - Video predictions: Youtube [LINK]&#40;&#41;.)


## Citation

If you use this code, please consider citing PAConv:

```
@inproceedings{xu2021paconv,
  title={PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds},
  author={Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgement
The code is partially borrowed from [PointWeb](https://github.com/hszhao/PointWeb).
The code is heavily based on [PAConv](https://github.com/CVMI-Lab/PAConv).