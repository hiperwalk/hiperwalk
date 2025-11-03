from distutils.core import setup, Extension
import numpy
import os

home = os.path.expanduser("~")
home = os.environ.get("home", os.path.expanduser("~"))
print("home = ", home); #exit()
#home = "/mnt/c/Users/bidu/"
#home = "/home/bidu"

hiperblas_module = Extension(
    'hiperblas',
    define_macros=[('MAJOR_VERSION', '0'),
                   ('MINOR_VERSION', '4')],
    libraries=['hiperblas-core', 'hiperblas-cpu-bridge'],  # inclui ambas
    include_dirs=[os.path.join(home, 'hiperblas/include'), numpy.get_include()],
    library_dirs=[os.path.join(home, 'hiperblas/lib')],
    sources=['hiperblas_wrapper.c']
)
setup(
    name='HiperblasExtension',
    version='0.4',
    description='This is the hiperblas math package',
    ext_modules=[hiperblas_module],
    include_dirs=[numpy.get_include()]
)

