# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pintax"
copyright = "2025, Xinyang Chen"
author = "Xinyang Chen"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx_toolbox.more_autodoc",
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.linkcode",
    # "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    # "sphinx_toolbox.more_autodoc.overloads",
    #
    # "sphinx_tabs.tabs",
    # "sphinx-prompt",
    # "matplotlib.sphinxext.plot_directive",
    # "myst_nb",
    # "sphinx_remove_toctrees",
    # "sphinx_copybutton",
    # "jax_extensions",
    # "sphinx_design",
    # "sphinxext.rediraffe",
    "sphinxcontrib.restbuilder",
    "sphinx_rtd_theme",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
}

autodoc_type_aliases = {
    "QuantityLike": "Quantity | Unit | ArrayLike",
    "ArrayLike": "ArrayLike",
}

autodoc_class_signature = "separated"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
