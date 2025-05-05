from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("rough_vis_deformation.py")
)