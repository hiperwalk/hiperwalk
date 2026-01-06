from setuptools import setup, Extension
import numpy as np
import os

here = os.path.abspath(os.path.dirname(__file__))


HIPERBLAS_PREFIX = os.environ.get(
    "HIPERBLAS_PREFIX",
    os.path.expanduser("~/hiperblas")
)

hiperblas_module = Extension(
    "hiperblas",
    define_macros=[
        ("MAJOR_VERSION", "0"),
        ("MINOR_VERSION", "4"),
    ],
    libraries=["hiperblas-core", "hiperblas-cpu-bridge"],
    runtime_library_dirs=[os.path.join(HIPERBLAS_PREFIX, "lib")],
    library_dirs=[os.path.join(HIPERBLAS_PREFIX, "lib")],
    include_dirs=[
        os.path.join(HIPERBLAS_PREFIX, "include"),
        np.get_include(),   # <<< ESSENCIAL
    ],
    sources=["hiperblas_wrapper.c"]
)


    #runtime_library_dirs=[ os.path.join(here, "lib"), ],
    #library_dirs=[ os.path.join(here, "~/hiperblas/lib"), ],
    #include_dirs=[ os.path.join(here, "include"),   np.get_include(), ],
setup(
    name="HiperblasExtension",
    version="0.4",
    description="HiperBLAS math package",
    ext_modules=[hiperblas_module],
)

