#!/usr/bin/env python
"""Django 管理命令入口。

直接执行本文件时，会把项目根目录加入 Python 模块搜索路径，设置 Django
配置模块，然后把 ``runserver``、``check`` 等命令行参数原样交给 Django。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    # web/manage.py 位于项目根目录的下一层，先加入根目录才能导入 src 与 web 包。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # 仅在调用方未显式设置时选择本项目配置，便于测试或部署环境覆盖。
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.topic17_web.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
