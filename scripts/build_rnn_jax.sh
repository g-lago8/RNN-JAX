#!/usr/bin/env bash
# Build script for the rnn_jax package
set -euo pipefail

# Use PYTHON env var if set, otherwise prefer python3
PYTHON=${PYTHON:-python3}

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Building rnn_jax package from: $ROOT_DIR"

# Ensure dist directory is clean
rm -rf dist
mkdir -p dist

# Check python version minimally (requires >=3.11 as per pyproject.toml)
if ! "$PYTHON" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
  echo "Error: \`$PYTHON\` is not available. Install Python 3.11+ or set PYTHON to a valid python executable." >&2
  exit 2
fi

# Ensure build backend tool is available (PEP 517 build frontend)
if ! "$PYTHON" -m build --version >/dev/null 2>&1; then
  echo "\nThe 'build' package is not installed for $PYTHON. Installing it now (user install)..."
  "$PYTHON" -m pip install --user build || {
    echo "Failed to install 'build'. Please install it manually: '$PYTHON -m pip install build'" >&2
    exit 3
  }
fi

echo "Running build (sdist + wheel)..."
# Build the project (pyproject.toml defines the package)
"$PYTHON" -m build --sdist --wheel -o dist

echo "Build finished. Artifacts in: $ROOT_DIR/dist"
ls -la dist || true

echo "Done. To install the built wheel run:"
echo "  $PYTHON -m pip install dist/*.whl"
