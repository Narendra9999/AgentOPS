"""
Shared dependency installer for iterative improvement notebooks.

Searches for wheels in three locations (in priority order):
  1. Bundled wheels (shipped with the DAB bundle)
  2. Mastercard UC Volume
  3. PyPI (internet-connected workspaces)

Usage in notebooks:
    exec(open("/Workspace" + os.path.dirname(os.path.dirname(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    )) + "/install_deps.py").read())
    install_iterative_deps(dbutils)
"""

import subprocess
import os
import sys


def _find_wheels_path():
    """Find the bundled wheels directory relative to this file's location in workspace."""
    # When deployed via DAB, the structure is:
    #   .bundle/agentops/<target>/files/src/agent_development/agent_evaluation/wheels/
    # This file is at:
    #   .bundle/agentops/<target>/files/src/agent_development/agent_evaluation/install_deps.py

    # Try bundled wheels (same directory as this file)
    candidates = []

    # Path 1: relative to notebook location
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        nb_dir = os.path.dirname(nb_path)  # .../notebooks
        eval_dir = os.path.dirname(nb_dir)  # .../agent_evaluation
        bundled = "/Workspace" + eval_dir + "/wheels"
        candidates.append(bundled)
    except Exception:
        pass

    # Path 2: Mastercard UC Volume (Python 3.12)
    candidates.append("/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs")

    # Path 3: FEVM UC Volume
    candidates.append("/Volumes/classic_stable_cykcbe_catalog/agentops/app_wheels")

    for path in candidates:
        if os.path.exists(path):
            whl_count = len([f for f in os.listdir(path) if f.endswith(".whl")])
            if whl_count > 0:
                print(f"Found {whl_count} wheels at: {path}")
                return path

    return None


def install_iterative_deps(dbutils_ref=None, packages=None):
    """Install iterative improvement dependencies.

    Args:
        dbutils_ref: dbutils reference (for restartPython)
        packages: Override package list. Defaults to iterative improvement packages.
    """
    if packages is None:
        packages = ["databricks-agents", "mlflow[genai]", "dspy"]

    wheels_path = _find_wheels_path()

    if wheels_path:
        # Air-gapped: install from local wheels
        print(f"Installing from wheels: {wheels_path}")
        # Use package names without version pins — let the wheels resolve
        pkg_names = [p.split("[")[0].split(">")[0].split("=")[0] for p in packages]
        subprocess.check_call([
            "pip", "install", "-U",
            *pkg_names,
            "--find-links", wheels_path,
            "--no-index", "-q",
        ])
    else:
        # Internet: install from PyPI
        print("Installing from PyPI...")
        subprocess.check_call([
            "pip", "install", "-U",
            *packages,
            "-q",
        ])

    print("Dependencies installed.")

    if dbutils_ref:
        dbutils_ref.library.restartPython()
