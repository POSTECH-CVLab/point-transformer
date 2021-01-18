Point-Transformer PyTorch
============================

* Implemention of `Point Transformer <https://arxiv.org/abs/2012.09164>`_.

* Code based on PointNet2 Pytorch repository `https://github.com/erikwijmans/Pointnet2_PyTorch`_.


Setup
-----

* Install ``python`` -- This repo is tested with ``{3.6, 3.7}``

* Install ``pytorch`` with CUDA -- This repo is tested with ``{1.4, 1.5}``.
  It may work with versions newer than ``1.5``, but this is not guaranteed.


* Install dependencies

  ::

    pip install -r requirements.txt







Example training
----------------

Install with: ``pip install -e .``

There example training script can be found in ``pointnet2/train.py``.  The training examples are built
using `PyTorch Lightning <https://github.com/williamFalcon/pytorch-lightning>`_ and `Hydra <https://hydra.cc/>`_.


A classifion pointnet can be trained as

::

  python pointnet2/train.py task=cls

  # Or with model=msg for multi-scale grouping

  python pointnet2/train.py task=cls model=msg


Similarly, semantic segmentation can be trained by changing the task to ``semseg``

::

  python pointnet2/train.py task=semseg



Multi-GPU training can be enabled by passing a list of GPU ids to use, for instance

::

  python pointnet2/train.py task=cls gpus=[0,1,2,3]


Building only the CUDA kernels
----------------------------------


::

  pip install pointnet2_ops_lib/.

  # Or if you would like to install them directly (this can also be used in a requirements.txt)

  pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"






Contributing
------------

This repository uses `black <https://github.com/ambv/black>`_ for linting and style enforcement on python code.
For c++/cuda code,
`clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ is used for style.  The simplest way to
comply with style is via `pre-commit <https://pre-commit.com/>`_

::

  pip install pre-commit
  pre-commit install
