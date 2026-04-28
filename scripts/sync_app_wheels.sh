#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# sync_app_wheels.sh — Download all app dependencies and upload to volume
#
# Uses the frozen requirements file (resolved by uv for Python 3.11)
# to download the exact complete dependency tree. No guessing.
#
# Usage:
#   ./scripts/sync_app_wheels.sh <local_dir> --download-only
#   ./scripts/sync_app_wheels.sh <local_dir> <volume_path> [--profile <name>]
#
# To regenerate the frozen file (if you change top-level packages):
#   uv pip compile app-bundle/app/requirements.in \
#     --python-version 3.11 --python-platform linux \
#     --index-url https://pypi-proxy.dev.databricks.com/simple \
#     --output-file app-bundle/app/requirements-frozen.txt
# ─────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FROZEN_FILE="$PROJECT_DIR/app-bundle/app/requirements-frozen.txt"

DOWNLOAD_DIR=""
VOLUME_PATH=""
PROFILE=""
DOWNLOAD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile|-p) PROFILE="$2"; shift 2 ;;
        --download-only) DOWNLOAD_ONLY=true; shift ;;
        --frozen) FROZEN_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 <local_dir> [<volume_path>] [--download-only] [--profile <name>]"
            echo ""
            echo "  --download-only     Download wheels locally, skip upload"
            echo "  --profile           Databricks CLI profile for upload"
            echo "  --frozen            Path to frozen requirements file (default: app-bundle/app/requirements-frozen.txt)"
            exit 0
            ;;
        *)
            if [[ -z "$DOWNLOAD_DIR" ]]; then
                DOWNLOAD_DIR="$1"
            elif [[ -z "$VOLUME_PATH" ]]; then
                VOLUME_PATH="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$DOWNLOAD_DIR" ]]; then
    echo "Usage: $0 <local_dir> [<volume_path>] [--download-only] [--profile <name>]"
    exit 1
fi

if [[ "$DOWNLOAD_ONLY" == false && -z "$VOLUME_PATH" ]]; then
    echo "ERROR: Provide a volume_path or use --download-only"
    exit 1
fi

if [[ ! -f "$FROZEN_FILE" ]]; then
    echo "ERROR: Frozen requirements not found: $FROZEN_FILE"
    echo "Generate with: uv pip compile app-bundle/app/requirements.in --python-version 3.11 --python-platform linux --output-file $FROZEN_FILE"
    exit 1
fi

PACKAGE_COUNT=$(grep -cv "^#\|^$\|^ " "$FROZEN_FILE")

echo "═══════════════════════════════════════════════════"
echo "  AgentOPS App Wheel Sync"
echo "═══════════════════════════════════════════════════"
echo "  Frozen file:  $FROZEN_FILE"
echo "  Packages:     $PACKAGE_COUNT"
echo "  Python:       3.11 (manylinux x86_64)"
echo "  Download dir: $DOWNLOAD_DIR"
echo "  Volume path:  ${VOLUME_PATH:-N/A (download only)}"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Download platform-specific wheels ──
echo "[1/3] Downloading wheels..."
mkdir -p "$DOWNLOAD_DIR"

pip download \
    -r "$FROZEN_FILE" \
    --python-version 311 \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_28_x86_64 \
    --platform linux_x86_64 \
    --only-binary=:all: \
    -d "$DOWNLOAD_DIR" \
    --quiet 2>&1

# Catch any pure-Python sdist packages missed by --only-binary
pip download \
    -r "$FROZEN_FILE" \
    --no-deps \
    -d "$DOWNLOAD_DIR" \
    --quiet 2>/dev/null || true

WHEEL_COUNT=$(ls "$DOWNLOAD_DIR"/*.whl 2>/dev/null | wc -l | tr -d ' ')
echo "  Downloaded $WHEEL_COUNT wheels"
echo ""

# ── Step 2: Verify all packages are present ──
echo "[2/3] Verifying..."
MISSING=0
# Build list of downloaded package names (normalized)
WHEEL_NAMES=$(ls "$DOWNLOAD_DIR"/*.whl 2>/dev/null | xargs -I{} basename {} | while read w; do
    echo "$w" | sed 's/-.*//' | tr '[:upper:]' '[:lower:]' | tr '-' '_'
done | sort -u)

while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" || "$line" =~ ^[[:space:]] ]] && continue
    PKG=$(echo "$line" | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    if echo "$WHEEL_NAMES" | grep -q "^${PKG}$"; then
        : # found
    else
        echo "  MISSING: $line"
        MISSING=$((MISSING + 1))
    fi
done < "$FROZEN_FILE"

if [[ $MISSING -eq 0 ]]; then
    echo "  All $PACKAGE_COUNT packages present"
else
    echo "  WARNING: $MISSING packages missing"
fi
echo ""

# ── Step 3: Upload or skip ──
if [[ "$DOWNLOAD_ONLY" == true ]]; then
    TOTAL_SIZE=$(du -sh "$DOWNLOAD_DIR" | awk '{print $1}')
    echo "[3/3] Skipped upload (--download-only)"
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Done! $WHEEL_COUNT wheels ($TOTAL_SIZE) in $DOWNLOAD_DIR"
    echo ""
    echo "  To upload:"
    echo "    $0 $DOWNLOAD_DIR <volume_path> [--profile <name>]"
    echo "═══════════════════════════════════════════════════"
    exit 0
fi

echo "[3/3] Uploading to $VOLUME_PATH..."

PROFILE_ARG=""
if [[ -n "$PROFILE" ]]; then
    PROFILE_ARG="--profile $PROFILE"
fi

UPLOADED=0
SKIPPED=0
FAILED=0

for whl in "$DOWNLOAD_DIR"/*.whl; do
    if [[ -f "$whl" ]]; then
        FNAME=$(basename "$whl")
        EXISTS=$(databricks fs ls "dbfs:${VOLUME_PATH}/${FNAME}" $PROFILE_ARG 2>/dev/null && echo "yes" || echo "no")
        if [[ "$EXISTS" == "yes" ]]; then
            SKIPPED=$((SKIPPED + 1))
        else
            if databricks fs cp "$whl" "dbfs:${VOLUME_PATH}/${FNAME}" $PROFILE_ARG 2>/dev/null; then
                UPLOADED=$((UPLOADED + 1))
                echo "  Uploaded: $FNAME"
            else
                FAILED=$((FAILED + 1))
                echo "  FAILED:   $FNAME"
            fi
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Done!"
echo "  Uploaded: $UPLOADED | Skipped: $SKIPPED | Failed: $FAILED"
echo "═══════════════════════════════════════════════════"
