#!/usr/bin/env python
"""Django management entry point for the Topic 17 demo site."""

from __future__ import annotations

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.topic17_web.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
