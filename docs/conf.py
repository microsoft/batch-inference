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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "Batch Inference Toolkit"
copyright = "2023, AI Platform Team, STCA, Microsoft"
author = "Yong Huang, Xi Chen, Lu Ye, Ze Tao"

# The full version, including alpha/beta/rc tags
release = "1.0rc0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_theme_options = {}

autoclass_content = "both"


def process_signature(app, what, name, obj, options, signature, return_annotation):
    print(f">>>> {return_annotation} {name}{signature}")

    if isinstance(signature, str):
        signature = signature.replace("pyis.python.lib.pyis_python.ops", "ops")
        signature = signature.replace("pyis.python.ops", "ops")

    if isinstance(return_annotation, str):
        return_annotation = return_annotation.replace(
            "pyis.python.lib.pyis_python.ops",
            "ops",
        )
        return_annotation = return_annotation.replace("pyis.python.ops", "ops")

    return (signature, return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", process_signature)
