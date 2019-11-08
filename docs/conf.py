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
sys.path.insert(0, os.path.abspath('../firelight/'))

master_doc = 'index'

# -- Project information -----------------------------------------------------

project = 'firelight'
copyright = '2019, Roman Remme'
author = 'Roman Remme'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx.ext.intersphinx',
        'sphinx.ext.doctest',
        'sphinx.ext.viewcode',
        'sphinx.ext.graphviz',
        'sphinx.ext.inheritance_diagram',
        #'sphinx.ext.autosummary',
        'sphinx_gallery.gen_gallery',
        'sphinx_paramlinks',
        'autodocsumm',
        'sphinx_automodapi.automodapi',
]

# autodoc_default_options = {
#     'autosummary': True,
# }

napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
#autosummary_generate = True

# interphinx configuration
intersphinx_mapping = {
        'numpy': ('http://docs.scipy.org/doc/numpy/', None),
        'python': ('https://docs.python.org/', None),
        'torch': ('https://pytorch.org/docs/master/', None),
        'sklearn': ('http://scikit-learn.org/stable',
                    (None, './_intersphinx/sklearn-objects.inv'))
}

# paths for sphinx gallery
sphinx_gallery_conf = {
        'examples_dir': '../examples',
        'gallery_dirs': 'auto_examples',
        'filename_pattern': '/*.py',
        'reference_url': {
            # The module you locally document uses None
            'sphinx_gallery': None,
        },
        # binder will does not work with readthedocs, see https://github.com/sphinx-gallery/sphinx-gallery/pull/505.
        # 'binder': {
        #      # Required keys
        #      'org': 'https://github.com',
        #      'repo': 'firelight',
        #      'branch': 'docs',  # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
        #      'binderhub_url': 'https://mybinder.org',  # Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
        #      'dependencies': 'requirements.txt',
        #      # Optional keys
        #      # 'filepath_prefix': 'docs/',  # A prefix to prepend to any filepaths in Binder links.
        #      'notebooks_dir': 'binder',  # Jupyter notebooks for Binder will be copied to this directory (relative to built documentation root).
        #      'use_jupyter_lab': False  # Whether Binder links should start Jupyter Lab instead of the Jupyter Notebook interface.
        # }
}

doctest_global_setup = """
import torch
from firelight.utils.dim_utils import *
from firelight.config_parsing import *

from firelight.visualizers.base import *
from firelight.visualizers.visualizers import *
from firelight.visualizers.container_visualizers import *
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
        'inferno.trainers.callbacks.base',
        'inferno.trainers.callbacks.logging.tensorboard',
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
