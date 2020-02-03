from distutils.core import setup, Extension
import numpy
try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

if USE_CYTHON:
    ext = Extension(name='c_funcs',
                    sources=["c_funcs.pyx"],
                    language="c",
                    include_dirs=[numpy.get_include()])
    ext = cythonize(ext,
                    annotate=False,
                    language_level="3",
                    compiler_directives={'cdivision': True,
                                         'boundscheck': False,
                                         'wraparound': False,
                                         'binding': True})
else:
    ext = Extension(name="c_funcs",
                    sources=["c_funcs.c"],
                    include_dirs=[numpy.get_include()])

setup(ext_modules=ext)