..
    [comment] Adapted from Sphinx-Autosummary-Recursion.
    https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/tree/master/docs/_templates

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

