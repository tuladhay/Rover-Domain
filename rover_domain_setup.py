from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "rover_domain",
        ["rover_domain.pyx"],
        extra_compile_args=['-std=c++11'])]

setup(ext_modules=cythonize(ext_modules))
