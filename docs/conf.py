"""Sphinx configuration for the project documentation."""

from datetime import datetime

project = "WiFi Activity Recognition"
author = "Diogo Ribeiro"
copyright = f"{datetime.now().year}, {author}"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
html_theme = "sphinx_rtd_theme"
