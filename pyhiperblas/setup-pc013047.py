from setuptools import setup, Extension
import numpy as np
import os

PREFIX = os.environ.get("HIPERBLAS_PREFIX", "/home/bidu/hiperblas")

hiperblas_module = Extension(
    "hiperblas",
    sources=["hiperblas_wrapper.c"],
    include_dirs=[
        np.get_include(),
        os.path.join(PREFIX, "include"),
    ],
    libraries=[
        "hiperblas-core",
        "hiperblas-cpu-bridge",
    ],
    library_dirs=[
        os.path.join(PREFIX, "lib"),
    ],
    runtime_library_dirs=[
        os.path.join(PREFIX, "lib"),
    ],
)

setup(
    name="hiperblas",
    version="0.4",
    description="HiperBLAS math package",
    ext_modules=[hiperblas_module],
)

