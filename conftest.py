import sys
import os

# Add the project root to sys.path so `from src.xxx import yyy` works
# when running `pytest` from the project root directory.
sys.path.insert(0, os.path.dirname(__file__))
