#!/usr/bin/env bash
# Generic Slurm entry point. Supply site-specific resources to sbatch itself.

set -euo pipefail

: "${INFINI_GRAM_MINI_ROOT:?Set this to an infini-gram-mini checkout}"
: "${CI_INDEX_DIR:?Set this to an Infini-gram Mini index directory}"

CI_HOST="${CI_HOST:-0.0.0.0}"
CI_PORT="${CI_PORT:-5555}"
CI_ADDRESS_FILE="${CI_ADDRESS_FILE:-runtime/ci-server.address}"

exec python -m longtail_creativity.cli serve-index \
  --engine-root "${INFINI_GRAM_MINI_ROOT}" \
  --index-dir "${CI_INDEX_DIR}" \
  --host "${CI_HOST}" \
  --port "${CI_PORT}" \
  --address-file "${CI_ADDRESS_FILE}"
