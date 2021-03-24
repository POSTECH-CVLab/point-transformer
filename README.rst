Point-Transformer PyTorch
============================

* Implemention of `Point Transformer <https://arxiv.org/abs/2012.09164>`_.

* Code based on `PointNet2 Pytorch repository <https://github.com/erikwijmans/Pointnet2_PyTorch>`_.


Setup
-----

* Install ``python`` -- This repo is tested with ``{3.6, 3.7}``

* Install ``pytorch`` with CUDA -- This repo is tested with ``{1.4, 1.5}``.
  It may work with versions newer than ``1.5``, but this is not guaranteed.


* Install dependencies

  ::

    pip install -r requirements.txt


Training
----------------

Install with: ``pip install -e .``

The example training script can be found in ``pointnet2/train.py``.  The training examples are built
using `PyTorch Lightning <https://github.com/williamFalcon/pytorch-lightning>`_ and `Hydra <https://hydra.cc/>`_.


You can train a Point Transformer model on various tasks as,

::

  # train Point Transformer for classification task on ModelNet40
  python -m pointnet2.train task=cls model=point_transformer 
  
  # train Point Transformer for part segmentation task on ShapeNet
  python -m pointnet2.train task=partseg model=point_transformer
  
  # train Point Transformer for semantic segmentation task on S3DIS
  python -m pointnet2.train task=semseg model=point_transformer

If you want to override the default config, you can pass the command line arguments, 

:: 

  # Change the batch size to 32
  python -m pointnet2.train task=cls model=point_transformer batch_size=32



Building only the CUDA kernels
----------------------------------


::

  pip install pointnet2_ops_lib/.

  # Or if you would like to install them directly (this can also be used in a requirements.txt)

  pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"



Experiment Results
----------------------------------

- Classification on ModelNet40

========  ========  ======
Model     mAcc      OA
========  ========  ======
Paper     90.6      93.7
Ours                87.2
========  ========  ======

- Part Segmentation on ShapeNet


- Semantic Segmentation on S3DIS



Contributing
------------

This repository uses `black <https://github.com/ambv/black>`_ for linting and style enforcement on python code.
For c++/cuda code,
`clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ is used for style.  The simplest way to
comply with style is via `pre-commit <https://pre-commit.com/>`_

::

  pip install pre-commit
  pre-commit install
