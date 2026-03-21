"""Streamlit Cloud 진입점.

``dashboard/app.py`` 의 ``main()`` 을 호출한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.app import main

main()
