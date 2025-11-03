from distutils.core import setup, Extension
import numpy

hiperblas_module = Extension('hiperblas',
                        define_macros = [('MAJOR_VERSION', '0'),
                                         ('MINOR_VERSION', '4')],
                        libraries = ['hiperblas-core'],

                        #include_dirs = ['../hiperblas-core/include',numpy.get_include()],
                        #library_dirs = ['/usr/local/lib64'],
                        #include_dirs = ['/home/bidu/libs/include',numpy.get_include()],
                        #library_dirs = ['/home/bidu/libs/lib'],
                        #include_dirs = ['/home/bidu/hiperblas/include',numpy.get_include()],
                        #library_dirs = ['/home/bidu/hiperblas/lib'],
                        include_dirs = ['$HOME/hiperblas/include',numpy.get_include()],
                        library_dirs = ['$HOME/hiperblas/lib'],
                        sources = ['hiperblas_wrapper.c'])
setup(name = 'HiperblasExtension',version='0.4',description = 'This is the hiperblas math package',ext_modules = [hiperblas_module],
      include_dirs=[numpy.get_include()])
