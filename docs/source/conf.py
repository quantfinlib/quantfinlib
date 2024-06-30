import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "QuantFinLib"
author = "Thijs van den Berg, Andrejs Fedjajevs, Mohammadjavad Vakili, Nathan de Vries"
release = "0.0.4"

extensions = [
    # "autoapi.extension",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Insert a table that contains links to documented items, and a short summary blurb (the first sentence of the docstring) for each of them.
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",
    "myst_parser",  # Support for Markdown documents
    "nbsphinx",  # Include Jupyter notebooks (examples)
    "nbsphinx_link",  # Link to Jupyternotebook that are outside the /docs tree
    "matplotlib.sphinxext.plot_directive",  # A directive for including a matplotlib plot in a Sphinx document.
    "sphinx_plotly_directive",  # A directive for including plotly plots in a Sphinx document.
    "sphinx_exec_code",  # executing python code snippets in the docs and showing result
]

# Extension Settings
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autodoc_inherit_docstrings = False  # If no docstring, inherit from base class
# set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# nbsphinx_allow_errors = True  # Continue through Jupyter errors
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints

# -------------------------------------------------------------------------------
# Autodoc
# extension: sphinx.ext.autodoc
#
# Include documentation from docstrings

# This extension can import the modules you are documenting, and pull in 
# documentation from docstrings in a semi-automatic way.
#
# For this to work, the docstrings must of course be written in correct 
# reStructuredText.
# If you prefer NumPy or Google style docstrings over reStructuredText, you can
#  also enable the napoleon extension.
# -------------------------------------------------------------------------------
autodoc_class_signature = "mixed" # "mixed" "separated"
autodoc_default_options = {
    'exclude-members': '__weakref__',
    'members': '', #'var1, var2', 
    'member-order': 'alphabetical',
    'undoc-members': False,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': False, 
    'show-inheritance': False, 
    'ignore-module-all': True, 
    'imported-members': False, 
    # 'exclude-members', 
    'class-doc-from': 'class',  # AKA autoclass_content
    # 'no-value'
    'add_module_names': True, # Ensure module names are shown in the documentation
} 

# -------------------------------------------------------------------------------
# Napoleon
# extension: sphinx.ext.napoleon
#
# NumPy and Google style docstrings instead of reStructuredText
#
# reStructuredText is great, but it creates visually dense, hard to read docstrings.
# Napoleon is a extension that enables Sphinx to parse both NumPy and Google style 
# docstrings.
# -------------------------------------------------------------------------------

# Configure napoleon to use NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Optional: Additional napoleon settings
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


# -------------------------------------------------------------------------------
# HTML
# -------------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_copyright = False
html_show_sphinx = False
html_show_sourcelink = False  # Remove 'view source code' from top of page
html_sourcelink_suffix = ""
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}
html_css_files = [('quantfinlib.css', {'priority': 999})]

# -------------------------------------------------------------------------------
# Source code snippets code highlighting in docstrings
# .. code-block:: python
# .. exec_code::
# 
# For galery of styles see: https://pygments.org/styles/
# high contrast dark: monokai, lightbulb, github-dark, rrt
# less contrast dark: zenburn, nord, material, one-dark, dracula, nord-darker
#    gruvbox-dark, stata-dark, paraiso-dark, coffee, solarized-dark, native,
#    inkpot, fruity, vim
# -------------------------------------------------------------------------------
pygments_style = 'material'


# -------------------------------------------------------------------------------
# Execute code snippets in docstrings
# -------------------------------------------------------------------------------
exec_code_working_dir = "."
exec_code_source_folders = ["../.."]



# -------------------------------------------------------------------------------
# General
# -------------------------------------------------------------------------------

# The suffix of source filenames.
master_doc = "index"
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    "_autoapi_templates",
    "_templates",
    "___*",
]


suppress_warnings = [
    # "toc",
    # "nbsphinx.localfile",
    # "nbsphinx.gallery",
    # "nbsphinx.thumbnail",
    # "nbsphinx.notebooktitle",
    # "nbsphinx.ipywidgets",
    # "config.cache",
]


# -------------------------------------------------------------------------------
# Jupyter notebooks
# -------------------------------------------------------------------------------

# for matplotlib support
nbsphinx_execute_arguments = [
    "--InLineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
nbsphinx_input_prompt = "In [%s]:"
nbsphinx_output_prompt = "Out[%s]:"


myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]


# -----------------------------------------------------
# Auto API
# -----------------------------------------------------
autoapi_dirs = ["../../quantfinlib"]
autoapi_options = [
    "members",
    "show-module-summary",
    #"undoc-members",  # https://github.com/readthedocs/sphinx-autoapi/issues/448#issuecomment-2166095130
]
autoapi_root = "_autoapi_root"
autoapi_keep_files = True
autoapi_template_dir = "_autoapi_templates"

# Only include class members, not inherited members
autoapi_python_class_content = 'both'
autoapi_own_page_level = "class"

"""
autoapi_member_order = "alphabetical"
autoapi_template_dir = "_autoapi_templates"
autoapi_add_toctree_entry = False
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    # "undoc-members",
    "show-module-summary",
    # "show-inheritance",
    # "imported-members",
    # "private-members",
    # "special-members",
]
autoapi_root = "_autoapi_root"
autoapi_keep_files = True
autoapi_ignore = ["*/___*"]
"""

# -----------------------------------------------------
# Math
# -----------------------------------------------------
mathjax3_config = {
    "jax": ["input/TeX", "output/SVG"],
}



suppress_warnings += [
    "config.cache",  # https://github.com/sphinx-doc/sphinx/issues/12300#issuecomment-2061022198
]
