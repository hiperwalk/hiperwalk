# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'Hiperwalk'
copyright = '2024'
author = 'Gustavo Bezerra'

# The full version, including alpha/beta/rc tags
release = 'stable'
version = 'stable'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo', #'sphinx_autodoc_typehints',
    'numpydoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.doctest',
]

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'networkx' : ('https://networkx.org/documentation/stable/', None),
}

autosummary_generate = True
autoclass_content = "both"
set_type_checking_flag = True
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Removes numpy auto documentation.
# Documentation is generated recursively according to _templates files
numpydoc_show_class_members = False
#numpydoc_class_members_toctree = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo" : {
        "text": "Hiperwalk",
        "alt_text": "Hiperwalk",
        "image_dark": "https://hiperwalk.org/images/logo_small_dark.png",
        "image_light": "https://hiperwalk.org/images/logo_small_light.png",
    },
    "switcher": {
        "json_url": "https://hiperwalk.org/switcher.json",
        "version_match": version,
    },
   "navbar_start": ["navbar-logo", "version-switcher"]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

master_doc = 'index'
