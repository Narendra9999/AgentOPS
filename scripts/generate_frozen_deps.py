"""
generate_frozen_deps.py — Run on a Databricks cluster to generate the complete
frozen dependency list for the AgentOPS app.

This installs all packages in a temp venv, freezes the full list,
and saves it to a UC Volume. Then you can download the frozen list
locally and use sync_app_wheels.sh to fetch all wheels.

Usage (in a Databricks notebook cell):
    %run /path/to/generate_frozen_deps

Or paste this into a notebook cell directly.
"""

import subprocess
import sys
import os
import tempfile

# Top-level packages for the app
PACKAGES = [
    "databricks-langchain",
    "databricks-ai-bridge",
    "databricks-sdk",
    "mlflow==3.10.1",
    "fastapi",
    "uvicorn",
    "pyyaml",
    "requests",
]

print(f"Python version: {sys.version}")
print(f"Installing: {PACKAGES}")
print()

# Create a temp venv to isolate from cluster packages
venv_dir = tempfile.mkdtemp(prefix="agentops_deps_")
subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

venv_pip = os.path.join(venv_dir, "bin", "pip")
venv_python = os.path.join(venv_dir, "bin", "python")

# Install all packages in the venv
print("Installing packages in isolated venv...")
subprocess.check_call(
    [venv_pip, "install", "--quiet"] + PACKAGES,
    stdout=subprocess.DEVNULL,
)

# Freeze the complete list
result = subprocess.run(
    [venv_pip, "freeze"],
    capture_output=True, text=True,
)
frozen = result.stdout.strip()
packages = [line for line in frozen.split("\n") if line and not line.startswith("-")]

print(f"\nTotal packages (including all transitive deps): {len(packages)}")
print("=" * 60)
for p in sorted(packages):
    print(f"  {p}")
print("=" * 60)

# Save to a file
output_path = "/tmp/agentops_app_frozen_requirements.txt"
with open(output_path, "w") as f:
    f.write("\n".join(sorted(packages)) + "\n")
print(f"\nSaved to: {output_path}")

# Also save to UC Volume if available
try:
    volume_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/311/agentops_app_frozen_requirements.txt"
    with open(volume_path, "w") as f:
        f.write("\n".join(sorted(packages)) + "\n")
    print(f"Saved to volume: {volume_path}")
except Exception as e:
    print(f"Volume save skipped: {e}")

# Cleanup
import shutil
shutil.rmtree(venv_dir, ignore_errors=True)

print("\nNext steps:")
print("1. Copy the frozen list to your local machine")
print("2. Run: pip download -r agentops_app_frozen_requirements.txt --python-version 311 --platform manylinux2014_x86_64 --only-binary=:all: -d ./wheels/")
print("3. For any missing wheels, run: pip download <package> --no-deps -d ./wheels/")
print("4. Upload to volume: ./scripts/sync_app_wheels.sh ./wheels <volume_path>")
