.. comment
    adapted from networkx

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::

   {% for item in methods %}
       {% if item != "__init__" %}
          ~{{ name }}.{{ item }}
       {% endif %}
   {%- endfor %}

   {% for item in methods %}
   {% if item != "__init__" %}
   .. automethod:: {{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. if the class page gets too large,
..    change the layout as follows
..    (removing second if and changing the first if)
..   {% if methods %}
..   .. rubric:: {{ _('Methods') }}
..   .. autosummary::
..      :toctree: generated
..   {% for item in methods %}
..       {% if item != "__init__" %}
..          ~{{ name }}.{{ item }}
..       {% endif %}
..   {%- endfor %}
..   {% endif %}
