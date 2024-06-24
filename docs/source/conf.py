import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "QuantFinLib"
author = "Thijs van den Berg, Andrejs Fedjajevs, Mohammadjavad Vakili, Nathan de Vries"
release = "0.0.4"

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",
    "myst_parser",  # Support for Markdown documents
    "nbsphinx",  # Include Jupyter notebooks (examples)
    "nbsphinx_link",  # Link to Jupyternotebook that are outside the /docs tree
    "matplotlib.sphinxext.plot_directive",  # A directive for including a matplotlib plot in a Sphinx document.
    "sphinx_exec_code",  # executing python code snippets in the docs and showing result
]

# Extension Settings
autosummary_generate = True  # Turn on sphinx.ext.autosummary
html_show_sourcelink = False  # Remove 'view source code' from top of page
autodoc_inherit_docstrings = False  # If no docstring, inherit from base class
# set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# nbsphinx_allow_errors = True  # Continue through Jupyter errors
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures
# napoleon_google_docstring = False
# napoleon_use_param = False
# napoleon_use_ivar = True


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

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_copyright = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}


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

autodoc_default_flags = ["members", "imported-members"]

automodule_default_flags = ["members", "imported-members"]


# Boolean indicating whether to scan all found documents for autosummary directives,
# and to generate stub pages for each. It is enabled by default.
# autosummary_generate = True

# -----------------------------------------------------
# Auto API
# -----------------------------------------------------
autoapi_dirs = ["../../quantfinlib"]
autoapi_options = [
    "members",
]

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

mathjax3_config = {
    "jax": ["input/TeX", "output/SVG"],
}


exec_code_working_dir = "."
exec_code_folders = [".."]
