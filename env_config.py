#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
轻量 .env 读取模块（无第三方依赖）。

约定：
- .env 位于项目根目录（与本文件同级），格式 `KEY=VALUE`，支持 `#` 注释与引号包裹
- 优先级：系统环境变量 > .env 文件 > default

用法：
    from env_config import get_api_key
    DEEPSEEK_API_KEY = get_api_key("DEEPSEEK_API_KEY")

tools/ 与 real-time/ 下的脚本无法直接 import 时，先引导路径：
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from env_config import get_api_key
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

_cache = None


def _parse_env_file(path):
    data = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                data[key.strip()] = val.strip().strip('"').strip("'")
    except OSError:
        pass
    return data


def _env_cache():
    global _cache
    if _cache is None:
        _cache = _parse_env_file(_ENV_PATH)
    return _cache


def get_api_key(name, default=""):
    """读取 API Key：系统环境变量优先，其次项目根 .env，最后 default。"""
    return os.environ.get(name) or _env_cache().get(name) or default


def is_configured(name):
    """判断某个 Key 是否已配置（用于启动前自检提示）。"""
    return bool(get_api_key(name))
