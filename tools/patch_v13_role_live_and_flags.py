#!/usr/bin/env python3
from __future__ import annotations

# Deprecated compatibility wrapper.
# The old version of this helper could be run twice and duplicate
# --no-export-architecture-memory in argparse. Keep this filename safe by
# delegating to the robust idempotent patcher.

from patch_v13_stability_v2 import main


if __name__ == "__main__":
    main()
