"""Published dependency-image seed (see `recipe.admitted_seed`).

Rewritten as a whole by `scripts/deps_image.py` from `.github/workflows/deps-image.yml`; do not
edit by hand. An empty image builds the environment in the engine.
"""

DEPS_IMAGE = (
    "ghcr.io/pshokeen/ssgrep-ci-deps"
    "@sha256:ae4a80603cf14c9b59dc46827c4db8d6e78cbb325757a0d5f2a711312552b5ac"
)
DEPS_IMAGE_BASE = (
    "python:3.13-slim@sha256:2d44738ebf4ce5f63b9d9ea871e9c7cfdc0426f7b0c97cac915fd88ded1d86bb"
)
