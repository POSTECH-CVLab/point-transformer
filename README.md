# Point Transformer
This repository reproduces [Point Transformer](https://arxiv.org/abs/2012.09164). \
The codebase is provided by the first author of [Point Transformer](https://arxiv.org/abs/2012.09164).

## Notes
- For shape classification and part segmentation, please use paconv-codebase branch. After some testing, we will merge it into the master branch.

---
## Dependencies
- Ubuntu: 18.04 or higher
- PyTorch: 1.9.0 
- CUDA: 11.1 
- Hardware: 4GPUs (TITAN RTX) to reproduce [Point Transformer](https://arxiv.org/abs/2012.09164) 
- To create conda environment, command as follows:

  ```
  bash env_setup.sh pt
  ```

## Dataset preparation
- Download S3DIS [dataset](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) and symlink the paths to them as follows:

     ```
     mkdir -p dataset
     ln -s /path_to_s3dis_dataset dataset/s3dis
     ```

## Usage
- Shape classification on ModelNet40
  - For now, please use paconv-codebase branch.
- Part segmentation on ShapeNetPart
  - For now, please use paconv-codebase branch.
- Semantic segmantation on S3DIS Area 5
  - Train

    - Specify the gpu used in config and then do training:

      ```
      sh tool/train.sh s3dis pointtransformer_repro
      ```

  - Test

    - Afer training, you can test the checkpoint as follows:

      ```
      CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis pointtransformer_repro
      ```
  ---
## Experimental Results

- Semanctic Segmentation on S3DIS Area 5

  |Model | mAcc | OA | mIoU |
  |-------| ------| ----| -------|
  |Paper| 76.5 | 90.8 | 70.4 |
  |Hengshuang's code | 76.8 | 90.4 | 70.0 |
---
## References

If you use this code, please cite [Point Transformer](https://arxiv.org/abs/2012.09164):
```
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```

## Acknowledgement
The code is from the first author of [Point Transformer](https://arxiv.org/abs/2012.09164).
We also refer [PAConv repository](https://github.com/CVMI-Lab/PAConv).
