from distutils.core import setup, Extension
import numpy
try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

if USE_CYTHON:
    extensions = [Extension(name='dp_utils', sources=["dp_utils.pyx"], language="c", include_dirs=[numpy.get_include()]),
                  Extension(name='index_utils', sources=["index_utils.pyx"], language="c", include_dirs=[numpy.get_include()])]
    extensions = cythonize(extensions, annotate=False, language_level="3",
                           compiler_directives={'cdivision': True,
                                                'boundscheck': False,
                                                'wraparound': False,
                                                'binding': True})
else:
    extensions = [Extension(name="*", sources=["*.c"], include_dirs=[numpy.get_include()])]

setup(ext_modules=extensions)