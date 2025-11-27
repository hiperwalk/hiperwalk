from distutils.core import setup, Extension
import numpy as np
import os

home = os.path.expanduser("~")

hiperblas_module = Extension(
    "hiperblas",
    define_macros=[("MAJOR_VERSION", "0"),
                   ("MINOR_VERSION", "4")],
    libraries=["hiperblas-core", "hiperblas-cpu-bridge"],
    include_dirs=[
        os.path.join(home, "hiperblas/include"),
        np.get_include()
    ],
    library_dirs=[os.path.join(home, "hiperblas/lib")],
    runtime_library_dirs=[os.path.join(home, "hiperblas/lib")],
    sources=["hiperblas_wrapper.c"],
)

setup(
    name="HiperblasExtension",
    version="0.4",
    description="HiperBLAS math package",
    ext_modules=[hiperblas_module],
)

