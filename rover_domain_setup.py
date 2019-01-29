# Run setup
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

#Options.annotate = True

setup(
    ext_modules = cythonize("rover_domain.pyx")
)