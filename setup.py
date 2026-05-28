"""
setup.py — fallback for older pip/setuptools that don't read pyproject.toml.
All canonical metadata lives in pyproject.toml.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()