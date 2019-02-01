# Run setup
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from Cython.Distutils import build_ext

#Options.annotate = True

setup(
    name="rover_domain",
    ext_modules=[
        Extension('rover_domain',
            sources=['rover_domain.pyx'],
            extra_compile_args=['-std=c++11'],
            language='c++')
        ],
    cmdclass = {'build_ext': build_ext}
)

