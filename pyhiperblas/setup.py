from setuptools import setup, Extension
import numpy as np
import os

HIPERBLAS_PREFIX = os.path.expanduser(
    os.environ.get("HIPERBLAS_PREFIX", "~/hiperblas")
)

hiperblas_module = Extension(
    name="hiperblas",
    sources=["hiperblas_wrapper.c"],
    include_dirs=[
        np.get_include(),
        os.path.join(HIPERBLAS_PREFIX, "include"),
    ],
    libraries=[
        "hiperblas-core",
        "hiperblas-cpu-bridge",
    ],
    library_dirs=[
        os.path.join(HIPERBLAS_PREFIX, "lib"),
    ],
    extra_link_args=[
        "-Wl,-rpath,$ORIGIN",
        "-Wl,-rpath," + os.path.join(HIPERBLAS_PREFIX, "lib"),
    ],
)

setup(
    name="hiperblas",
    version="0.4",
    ext_modules=[hiperblas_module],
)

