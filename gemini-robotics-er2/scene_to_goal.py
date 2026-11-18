'''
SETUP:
- create venv and run pip install -r requirements.txt from w/in this folder
- export GEMINI_API_KEY=<key> (aistudio.google.com)

Photo requirements:
- current photo: objects in a grid
- goal photo: either a 2d goal state (objects move locations in the grid) or a 3d goal state (cup stacking).
              for 3d goal state, only the target shape matters (not the location, i.e. it can build it anywhere on the grid).
              for cup stacking, each cup is read as either resting on the table or on 1 - 2 other cups, which is then turned into
              positions relative to the first bottom cup

Usage:
    python scene_to_goal.py current.jpg goal.jpg
    python scene_to_goal.py current.jpg goal.jpg -o states.json --config my_grid.toml
    python scene_to_goal.py a b --show-prompt (prints VLM prompt only)

Output will be (for example):
    {
      "current": {"red cup":   {"on": "table", "cell": [0, 1], "level": 0, "xyz_mm": [150.0, 50.0, 0.0]},
                  "green cup": {"on": ["red cup"], "level": 1, "xyz_mm": [150.0, 50.0, 110.0]}, ...},
      "goal_shape": {"red cup":  {"on": "table", "level": 0, "xyz_rel_mm": [0.0, 0.0, 0.0]},
                     "pink cup": {"on": ["red cup", "blue cup"], "level": 1, "xyz_rel_mm": [45.0, 0.0, 110.0]}, ...},
      "base_order": ["red cup", "blue cup", "green cup"],
      "build_order": ["red cup", "blue cup", "green cup", "pink cup", ...]
    }
'''
import argparse
import json
import math
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEFAULT_CONFIG = Path(__file__).with_name("grid_config.toml")
TABLE = "table"
MAX_SUPPORTS = 2

# PARSE CONFIG AND CREATE PROMPT

# PARSE VLM OUTPUT

# CHECK STRUCTURE

# GET GEOMETRY

# VALIDATE & BUILD OUTPUT

# VLM

def main():
    return

if __name__ == "__main__":
    main()
