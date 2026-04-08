try:
    import hiperblas as hpb
except ModuleNotFoundError:
    pass
from .._constants import *

############################################
# used for automatically stopping the engine
import atexit

__engine_initiated = False
__hpc_type = None

def get_hpc():
    global __hpc_type

    if __hpc_type == 0:
        return 'cpu'
    if __hpc_type == 1:
        return 'gpu'

    return None

def set_hpc(hpc):
    r"""
    Indicate which HPC platform is going to be used.

    After executing the ``set_hpc`` command,
    all subsequent hiperwalk commands will
    use the designated HPC platform.

    Parameters
    ----------
    hpc : {None, 'cpu', 'gpu'}
        Indicates whether to utilize HPC
        for matrix multiplication using CPU or GPU.
        If ``hpc=None``, it will use standalone Python.
    """
    new_hpc = hpc

    if hpc is not None:
        hpc = hpc.lower()
        hpc = hpc.strip()

        if hpc == 'cpu':
            new_hpc = 0
        elif hpc == 'gpu':
            new_hpc = 1
        else:
            raise ValueError(
                    'Unexpected value of `hpc`: '
                    + new_hpc + '. Expected a value in '
                    + "[None, 'cpu', 'gpu'].")

    global __hpc_type
    global __engine_initiated

    if __hpc_type != new_hpc:

        __hpc_type = new_hpc

        if __hpc_type == 0 and not __engine_initiated:
            hiperblas_imported = True
            try:
                hpb.init_engine(__hpc_type, 0)
            except NameError:
                hiperblas_imported = False
            if not hiperblas_imported:
                raise ModuleNotFoundError(
                    "Module hiperblas was not imported. "
                    + "Was it installed properly?"
                )
            __engine_initiated = True

def exit_handler():
    global __engine_initiated

    if __engine_initiated:
        hpb.stop_engine()
        __engine_initiated = False

atexit.register(exit_handler)
