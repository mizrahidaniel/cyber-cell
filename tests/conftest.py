"""Shared pytest configuration — initialize Taichi once per session.

Taichi fields are allocated at module import time. Calling ti.init() more than
once in a process corrupts field handles. This conftest ensures a single init
before any test imports simulation modules.
"""

import taichi as ti
import pytest


@pytest.fixture(scope="session", autouse=True)
def init_taichi():
    ti.init(arch=ti.cpu, random_seed=42)
