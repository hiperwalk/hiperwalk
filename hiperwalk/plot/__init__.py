# Always use Gtk 3.0
from gi import require_version
require_version('Gtk', '3.0')

# importing everything under the hood of the
# hiperwalk.plot package
from ._plot import *
from ._animation import *
