Point-Transformer PyTorch
============================

* This is an unofficial implemention of `Point Transformer <https://arxiv.org/abs/2012.09164>`_.
* We use `PAConv Pytorch repository <https://github.com/CVMI-Lab/PAConv>`_ which the first author of `Point Transformer <https://arxiv.org/abs/2012.09164>`_ has participated in for the codebase.
* For k-nearest neighbor search with heap sort, please refer `PAConv Pytorch repository <https://github.com/CVMI-Lab/PAConv>`_.


Setup
-----
* The code is tested on Ubuntu 20.04 and CUDA 11.1.
* Install dependencies

  ::

    conda env create --file environment.yaml


Training & Evaluation
----------------

Check out sub-directories for classification, part segmentation, and scene segmentation.


Experiment Results
----------------------------------

- Classification on ModelNet40

================  ========  ======
Model             mAcc      OA
================  ========  ======
Paper             90.6      93.7
Our Implemention            
================  ========  ======

- Part Segmentation on ShapeNet

================  =========  =========
Model             cat. mIoU  ins. mIoU
================  =========  =========
Paper             83.7       86.6
Our Implemention             
================  =========  =========

- Semantic Segmentation on S3DIS Area5

================  ========  ======  ======
Model             mAcc      OA      mIoU
================  ========  ======  ======
Paper             76.5      90.8    70.4
Our Implemention               
================  ========  ======  ======


Acknowledment
-----

Our code is heavily based on `PAConv Pytorch repository <https://github.com/CVMI-Lab/PAConv>`_.