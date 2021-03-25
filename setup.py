import os.path as osp

from setuptools import find_packages, setup

requirements = ["hydra-core==0.11.3", "pytorch-lightning==0.7.1"]


exec(open(osp.join("point_transformer", "_version.py")).read())

setup(
    name="point_transformer",
    version=__version__,
    author="POSTECH Computer Vision Lab",
    packages=find_packages(),
    install_requires=requirements,
)
