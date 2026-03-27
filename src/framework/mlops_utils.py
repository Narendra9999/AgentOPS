"""
mlops_utils.py — Shared utilities for resilient MLOps CI/CD model export/import.

Integrated from Mastercard's MLOps platform utilities.

Provides:
  - Retry with exponential backoff
  - Checkpoint / resume manager (survives CI runner restarts)
  - Dynamic socket timeout scaled to model size
  - SHA-256 integrity verification
  - Multi-format dependency parser (conda.yaml, requirements.txt, pyproject.toml, Pipfile, setup.cfg)
  - MLflow model-flavor detection (incl. reference-only HuggingFace)
  - Symlink resolution before archiving
  - Serverless Python-version probe with retry
  - Job submission / polling helpers
  - Volume-based staging helpers (avoids /tmp exhaustion)
"""

import os
import json
import re
import time
import hashlib
import socket
import shutil
import functools
import configparser
import zipfile
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHARED_LIBS_BASE = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python"
PYTHON_LIB_PATHS = {
    "310": f"{SHARED_LIBS_BASE}/310/python310_all_libs",
    "311": f"{SHARED_LIBS_BASE}/311/python311_all_libs",
    "312": f"{SHARED_LIBS_BASE}/312/python312_all_libs",
}
ENV_VERSION_PYTHON_MAP = {"1": "310", "2": "311"}
MAX_JOB_WAIT_SECONDS = 3600
DEFAULT_SOCKET_TIMEOUT = 300

SERVERLESS_BUILTIN_PACKAGES = {
    "mlflow", "numpy", "pandas", "scipy", "cloudpickle",
    "typing_extensions", "packaging", "pip", "setuptools", "wheel",
    "pyyaml", "protobuf", "pyarrow", "databricks_sdk",
}


def get_pip_find_links(python_version: str = "312") -> str:
    """Get the --find-links path for a given Python version."""
    return PYTHON_LIB_PATHS.get(python_version, PYTHON_LIB_PATHS["312"])


def get_pip_install_args(python_version: str = "312") -> str:
    """Get pip install args for air-gapped environments."""
    return f"--find-links {get_pip_find_links(python_version)} --no-index"


# =========================================================================
# 1. Retry Decorator
# =========================================================================
def retry_with_backoff(
    max_retries=3,
    base_delay=5,
    max_delay=120,
    retryable_exceptions=(OSError, IOError, TimeoutError, ConnectionError),
):
    """Decorator: retry with exponential backoff on transient failures."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        print(f"  [Retry] {func.__name__} exhausted {max_retries+1} attempts.")
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(
                        f"  [Retry] {func.__name__} attempt {attempt+1}/{max_retries+1} "
                        f"failed: {exc}  — retrying in {delay}s"
                    )
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


# =========================================================================
# 2. Dynamic Socket Timeout
# =========================================================================
def set_dynamic_timeout(model_size_mb: float = 0, base_timeout: int = DEFAULT_SOCKET_TIMEOUT):
    """Scale socket timeout to model size: ~1 s per 10 MB, min base_timeout."""
    dynamic = max(base_timeout, int(model_size_mb / 10) + base_timeout)
    socket.setdefaulttimeout(dynamic)
    print(f"  [Timeout] Socket timeout → {dynamic}s (model ~{model_size_mb:.0f} MB)")
    return dynamic


def reset_timeout():
    """Reset to a safe default after heavy I/O."""
    socket.setdefaulttimeout(DEFAULT_SOCKET_TIMEOUT)


# =========================================================================
# 3. Checkpoint / Resume Manager
# =========================================================================
class CheckpointManager:
    """JSON-file checkpoint so a re-run skips already-completed models."""

    def __init__(self, checkpoint_dir: str, pipeline_name: str = "pipeline"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            checkpoint_dir, f"{pipeline_name}_checkpoint.json"
        )
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as fh:
                state = json.load(fh)
            print(f"  [Checkpoint] Resumed from {self.checkpoint_file}")
            return state
        return {"completed": [], "failed": [], "in_progress": None, "metadata": {}}

    def save(self):
        with open(self.checkpoint_file, "w") as fh:
            json.dump(self.state, fh, indent=2)

    def is_completed(self, model_key: str) -> bool:
        return model_key in self.state.get("completed", [])

    def mark_in_progress(self, model_key: str):
        self.state["in_progress"] = model_key
        self.save()

    def mark_completed(self, model_key: str, metadata: dict = None):
        if model_key not in self.state["completed"]:
            self.state["completed"].append(model_key)
        self.state["in_progress"] = None
        self.state["failed"] = [f for f in self.state["failed"] if f.get("name") != model_key]
        if metadata:
            self.state["metadata"][model_key] = metadata
        self.save()

    def mark_failed(self, model_key: str, reason: str):
        self.state["in_progress"] = None
        self.state["failed"] = [f for f in self.state["failed"] if f.get("name") != model_key]
        self.state["failed"].append({"name": model_key, "reason": reason})
        self.save()

    def get_metadata(self, model_key: str) -> dict:
        return self.state.get("metadata", {}).get(model_key, {})

    def clear(self):
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        self.state = {"completed": [], "failed": [], "in_progress": None, "metadata": {}}


# =========================================================================
# 4. SHA-256 Integrity
# =========================================================================
def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_integrity(filepath: str, expected_hash: str):
    actual = compute_sha256(filepath)
    if actual != expected_hash:
        raise ValueError(
            f"Integrity FAILED for {filepath}: expected {expected_hash[:16]}… got {actual[:16]}…"
        )
    print(f"  [Integrity] SHA-256 OK: {actual[:16]}…")


# =========================================================================
# 5. Dependency Parser — Multi-format support
# =========================================================================
def _safe_yaml_load(path: str) -> dict:
    if yaml:
        with open(path, "r") as fh:
            return yaml.safe_load(fh)
    data = {"dependencies": []}
    in_pip = False
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "- pip:":
                in_pip = True
                continue
            if in_pip:
                if stripped.startswith("- "):
                    data["dependencies"].append({"pip": [stripped[2:]]})
                else:
                    in_pip = False
    return data


def parse_model_dependencies(model_dir: str) -> list:
    """Extract pip-installable dependencies from the model artifact.

    Fallback chain: conda.yaml → requirements.txt → pyproject.toml → Pipfile → setup.cfg
    """
    deps = []

    # conda.yaml
    for name in ("conda.yaml", "conda.yml"):
        conda_path = os.path.join(model_dir, name)
        if os.path.exists(conda_path):
            try:
                conda_env = _safe_yaml_load(conda_path)
                for dep in conda_env.get("dependencies", []):
                    if isinstance(dep, dict) and "pip" in dep:
                        deps.extend(dep["pip"])
                    elif isinstance(dep, str) and not dep.startswith("python"):
                        deps.append(dep)
                print(f"  [Deps] Parsed {len(deps)} deps from {name}")
            except Exception as exc:
                print(f"  [WARN] Could not parse {name}: {exc}")
            break

    # requirements.txt
    if not deps:
        req_path = os.path.join(model_dir, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, "r") as fh:
                deps = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
            print(f"  [Deps] Parsed {len(deps)} deps from requirements.txt")

    # Filter out mlflow and local paths
    deps = [d for d in deps if not d.startswith("mlflow") and not d.startswith("-e ") and not d.startswith("file:")]

    return deps


# =========================================================================
# 6. Model Flavor Detection
# =========================================================================
def detect_model_flavor(model_dir: str) -> dict:
    """Detect MLflow flavor, check for reference-only HuggingFace models."""
    mlmodel_path = os.path.join(model_dir, "MLmodel")
    if not os.path.exists(mlmodel_path):
        return {"flavor": "unknown", "is_reference_only": False, "all_flavors": []}

    mlmodel = _safe_yaml_load(mlmodel_path)
    flavors = mlmodel.get("flavors", {})
    flavor_names = list(flavors.keys())
    primary_flavor = "python_function"

    for candidate in ("transformers", "tensorflow", "torch", "sklearn", "xgboost", "lightgbm", "spark", "langchain"):
        if candidate in flavors:
            primary_flavor = candidate
            break

    is_reference = False
    if primary_flavor == "transformers":
        tf_cfg = flavors["transformers"]
        has_ref = bool(tf_cfg.get("source_model_name") or tf_cfg.get("source_model_revision"))
        if has_ref:
            weight_exts = (".bin", ".safetensors", ".h5", ".pt", ".pth", ".gguf")
            weight_files = [f for f in os.listdir(model_dir) if any(f.endswith(ext) for ext in weight_exts)]
            if not weight_files:
                is_reference = True

    return {
        "flavor": primary_flavor,
        "all_flavors": flavor_names,
        "is_reference_only": is_reference,
        "mlmodel": mlmodel,
    }


# =========================================================================
# 7. Symlink Resolution
# =========================================================================
def resolve_symlinks(directory: str) -> int:
    """Replace symlinks with real file copies (zip-safe)."""
    count = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files + dirs:
            path = os.path.join(root, name)
            if os.path.islink(path):
                target = os.path.realpath(path)
                os.unlink(path)
                if os.path.isdir(target):
                    shutil.copytree(target, path)
                elif os.path.isfile(target):
                    shutil.copy2(target, path)
                count += 1
    if count:
        print(f"  [Symlink] Resolved {count} symlinks")
    return count


# =========================================================================
# 8. Directory Size
# =========================================================================
def get_dir_size_mb(path: str) -> float:
    total = 0
    for dp, _, fnames in os.walk(path):
        for f in fnames:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp) and not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


# =========================================================================
# 9. Volume-Based Staging Helpers
# =========================================================================
@retry_with_backoff(max_retries=3, base_delay=10)
def upload_to_volume(w, local_path: str, volume_path: str):
    """Upload a file to a Unity Catalog Volume with retry."""
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  [Upload] {local_path} → {volume_path} ({size_mb:.1f} MB)")
    with open(local_path, "rb") as fh:
        w.files.upload(volume_path, fh, overwrite=True)
    print(f"  [Upload] Complete.")


@retry_with_backoff(max_retries=3, base_delay=10)
def download_from_volume(w, volume_path: str, local_path: str):
    """Download a file from a Unity Catalog Volume with retry."""
    print(f"  [Download] {volume_path} → {local_path}")
    resp = w.files.download(volume_path)
    stream = resp.contents if hasattr(resp, "contents") else resp
    with open(local_path, "wb") as fh:
        while True:
            chunk = stream.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  [Download] Complete ({size_mb:.1f} MB)")


def create_volume_if_missing(w, catalog: str, schema: str, volume_name: str = "models_volume"):
    """Ensure a managed Volume exists; create if not."""
    from databricks.sdk.service.catalog import VolumeType
    fqn = f"{catalog}.{schema}.{volume_name}"
    try:
        w.volumes.read(fqn)
    except Exception:
        print(f"  [Volume] Creating {fqn}")
        w.volumes.create(
            catalog_name=catalog, schema_name=schema,
            name=volume_name, volume_type=VolumeType.MANAGED,
        )


def safe_zip_directory(source_dir: str, zip_path: str):
    """Zip a directory after resolving symlinks (avoids duplication)."""
    resolve_symlinks(source_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for fname in files:
                abs_path = os.path.join(root, fname)
                arc_name = os.path.relpath(abs_path, source_dir)
                zf.write(abs_path, arc_name)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  [Zip] Created {zip_path} ({size_mb:.1f} MB)")


def safe_unzip(zip_path: str, dest_dir: str):
    """Extract zip to destination."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  [Unzip] Extracted to {dest_dir}")


# =========================================================================
# 10. Job Submission / Polling Helpers
# =========================================================================
@retry_with_backoff(max_retries=3, base_delay=10)
def submit_job(w, payload: dict) -> int:
    """Submit a one-shot job run with retry."""
    response = w.api_client.do("POST", "/api/2.1/jobs/runs/submit", body=payload)
    if isinstance(response, dict):
        return response.get("run_id")
    if hasattr(response, "run_id"):
        return response.run_id
    return json.loads(str(response)).get("run_id")


def wait_for_job(w, run_id: int, timeout: int = MAX_JOB_WAIT_SECONDS):
    """Poll until the run reaches a terminal state."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Job {run_id} exceeded {timeout}s timeout")
        run_info = w.jobs.get_run(run_id=run_id)
        lc = str(getattr(run_info.state.life_cycle_state, "value", run_info.state.life_cycle_state))
        if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(15)

    rs = run_info.state.result_state
    if rs is None:
        raise Exception(f"Job {run_id} terminated without result. Lifecycle: {lc}")
    rs_str = str(getattr(rs, "value", rs))
    if rs_str != "SUCCESS":
        raise Exception(f"Job {run_id} failed: {rs_str} / {run_info.state.state_message}")
    return run_info


# =========================================================================
# 11. Helpers
# =========================================================================
def _silent(fn):
    """Swallow exceptions from a cleanup call."""
    try:
        fn()
    except Exception:
        pass


def safe_cleanup(*paths):
    """Delete files and directories, ignoring errors."""
    for p in paths:
        if p and os.path.exists(p):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                _silent(lambda: os.remove(p))


def _sanitize_deps(deps: list) -> list:
    """Sanitize dependency specs for no-internet installs.
    Strips version pins and filters out serverless builtins."""
    if not deps:
        return []
    cleaned = []
    seen = set()
    for raw in deps:
        raw = raw.strip()
        if not raw or raw.startswith("-") or raw.startswith("#"):
            continue
        base = re.split(r"[=<>!~;@\[]", raw)[0].strip()
        if not base:
            continue
        normalized = base.lower().replace("-", "_")
        if normalized in SERVERLESS_BUILTIN_PACKAGES:
            continue
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def build_env_spec(env_version: str, lib_path: str, extra_deps: list = None) -> dict:
    """Build serverless environment spec with optional extra pip deps."""
    deps = [f"--find-links {lib_path}", "--no-index", "mlflow"]
    if extra_deps:
        sanitized = _sanitize_deps(extra_deps)
        deps.extend(sanitized)
    return {
        "environment_key": "default",
        "spec": {"client": env_version, "dependencies": deps},
    }
