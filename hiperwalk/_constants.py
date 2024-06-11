__version__ = '2.0b13'

from sys import modules as sys_modules
from enum import Enum

__USING_PDB__ = 'pdb' in sys_modules
__GENERATING_DOCS__ = 'sphinx' in sys_modules
# ignores debug print messages when generating docs
# __DEBUG__ = __debug__ and __USING_PDB__ and not __GENERATING_DOCS__
