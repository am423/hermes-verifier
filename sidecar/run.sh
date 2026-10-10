#!/usr/bin/env bash
# Start the d1-omni decision sidecar.
#
# The plugin fails OPEN: if this is not running, every turn is reviewed and
# nothing visibly breaks. Check reachability with /verify decisions, which
# reports `gate: ok` or `gate: unreachable`.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "no .venv - run: python3 -m venv .venv && . .venv/bin/activate && pip install 'transformers>=5.15' torch torchvision pillow" >&2
  exit 1
fi

# shellcheck disable=SC1091
. .venv/bin/activate
exec python server.py "$@"
