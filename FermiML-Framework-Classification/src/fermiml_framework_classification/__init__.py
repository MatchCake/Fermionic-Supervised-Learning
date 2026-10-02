"""
Project description.
"""

import importlib_metadata

__author__ = "Jeremie Gince"
__email__ = "Jeremie.Gicne@usherbrooke.ca"
__copyright__ = "Copyright 2026, Jeremie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/FermiML-Framework-Classification"
__package__ = "fermiml_framework_classification"
__version__ = importlib_metadata.version(__package__)

import warnings

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")
