"""
All components that interact with VLM.

    read_grid(vlm, cfg, photo)   -> {name: {"cell": [row, col], "on": [...]}}
    read_shape(vlm, cfg, photo)  -> {"on": {name: [...]}, "base_order": [...]}
"""
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel

from checks import parse_grid, parse_shape

try:
    import tomllib  
except ModuleNotFoundError:
    import tomli as tomllib  


# CONFIG

DEFAULT_CONFIG = Path(__file__).with_name("grid_config.toml")
MODES = ("2d", "3d")

DEFAULTS = {
    "task": {"mode": "2d"},
    "grid": {"origin_marker": "", "max_objects_per_cell": 0},
    "objects": {"kind": "object", "names": []},
    "stacking": {"spacing_mm": 0.0, "level_height_mm": 0.0, "max_supports": 2},
    "robot": {"grid_origin_mm": [0.0, 0.0], "grid_rotation_deg": 0.0},
    "model": {"model_id": "gemini-robotics-er-2-preview", "api_version": "v1beta", 
              "requests_per_minute": 0, "rate_limit_retries": 5,
              "votes": 2, "max_attempts": 5, "max_image_px": 1600},
}
REQUIRED = {"grid": ("rows", "cols", "cell_width_mm", "cell_height_mm", "axis_hint")}


def load_config(path=DEFAULT_CONFIG, mode: str | None = None) -> dict:
    """Read grid_config.toml, fill in defaults. `mode` overrides [task] mode."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for section, values in DEFAULTS.items():
        cfg.setdefault(section, {})
        for key, value in values.items():
            cfg[section].setdefault(key, value)
    for section, keys in REQUIRED.items():
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(f"{path}: [{section}] is missing '{key}'")
    cfg["objects"]["names"] = [n.strip().lower() for n in cfg["objects"]["names"]]
    if mode:
        cfg["task"]["mode"] = mode
    if cfg["task"]["mode"] not in MODES:
        raise ValueError(f"[task] mode must be one of {MODES}, got {cfg['task']['mode']!r}")
    st = cfg["stacking"]
    if cfg["task"]["mode"] == "3d" and (st["spacing_mm"] <= 0 or st["level_height_mm"] <= 0):
        raise ValueError("3d mode needs [stacking] spacing_mm and level_height_mm > 0")
    return cfg


# LOAD PHOTOS

def load_jpeg(path, max_px: int = 1600) -> bytes:
    # Adjusts size of photo to ~1600px
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_px:
        img.thumbnail((max_px, max_px))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# SET RESPONSE FORMATS AS API RESPONSE SCHEMA 

class Placement(BaseModel):
    object: str
    row: int
    col: int


class GridReading(BaseModel):         
    objects: list[Placement]


class StackedPlacement(BaseModel):
    object: str
    row: int
    col: int
    on: list[str]                        


class StackedGridReading(BaseModel):     
    objects: list[StackedPlacement]


class ShapePart(BaseModel):
    object: str
    on: list[str]


class ShapeReading(BaseModel):          
    objects: list[ShapePart]
    base_order: list[str]               


# VLM PROMPTS

def _objects_rule(cfg: dict) -> str:
    kind, names = cfg["objects"]["kind"], cfg["objects"]["names"]
    if names:
        return (f"The possible {kind}s are: {', '.join(names)}. Use these names exactly. "
                f"Not every {kind} has to be present; only report ones you can see.")
    return f"Name each {kind} by its color, e.g. 'red {kind}'."


def grid_prompt(cfg: dict, allow_stacks: bool) -> tuple[str, str]:
    g, kind = cfg["grid"], cfg["objects"]["kind"]
    lines = [
        f"You read where {kind}s are on a grid from an overhead photo.",
        f"The grid has {g['rows']} rows (0 to {g['rows'] - 1}) and "
        f"{g['cols']} columns (0 to {g['cols'] - 1}).",
        g["axis_hint"],
    ]
    if g["origin_marker"]:
        lines.append(f"The origin cell is marked with {g['origin_marker']}; "
                     f"the marker is not a {kind}.")
    lines.append(_objects_rule(cfg))
    lines.append(f"Report every {kind} with the row and col of the cell it is in.")
    lines.append(f"Each cell holds at most one {kind}." if g["max_objects_per_cell"] == 1
                 else f"A cell can hold more than one {kind}.")
    if allow_stacks:
        lines.append(f"If a {kind} lies on top of other {kind}s, list the {kind}s "
                     "directly underneath it in 'on'; otherwise leave 'on' empty.")
    return "\n".join(lines), "Read the grid in this photo."


def shape_prompt(cfg: dict) -> tuple[str, str]:
    kind = cfg["objects"]["kind"]
    lines = [
        f"You read the target shape of a stack of {kind}s from a photo.",
        "Only the shape matters, not where it is on the table.",
        _objects_rule(cfg),
        f"For each {kind}, list in 'on' the {kind}s directly underneath it, "
        f"or leave 'on' empty if it sits on the table.",
        f"In 'base_order', list the {kind}s that sit on the table, "
        "from left to right as seen in the photo.",
    ]
    return "\n".join(lines), "Read the target shape in this photo."


# MODEL CALL

class GeminiVLM:
    # GEMINI ROBOTICS-ER 2

    def __init__(self, model_id: str, api_version: str = "v1beta",
                 requests_per_minute: float = 0, rate_limit_retries: int = 5):
        from google import genai
        self.client = genai.Client(http_options={"api_version": api_version})
        self.model_id = model_id
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.rate_limit_retries = rate_limit_retries
        self._last_call = None

    def _wait_for_slot(self):
        if self._last_call is not None and self.min_interval:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                print(f"  (waiting {wait:.0f}s to stay under the rate limit)", file=sys.stderr)
                time.sleep(wait)
        self._last_call = time.monotonic()

    def generate_json(self, system: str, images: list[bytes], text: str, schema) -> dict:
        from google.genai import errors, types
        parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
        for retry in range(self.rate_limit_retries + 1):
            self._wait_for_slot()
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[*parts, text],
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                return json.loads(resp.text)
            except errors.ClientError as e:
                if e.code != 429 or retry == self.rate_limit_retries:
                    raise
                if "PerDay" in str(e):
                    raise RuntimeError("Daily free-tier quota used up for this model; "
                                       "try again tomorrow or enable billing.") from e
                delay = _retry_delay_seconds(e)
                print(f"  (rate limited; retrying in {delay:.0f}s)", file=sys.stderr)
                time.sleep(delay)


def _retry_delay_seconds(err, default: float = 30.0) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(err))
    return (float(match.group(1)) if match else default) + 1.0


def make_vlm(cfg: dict) -> GeminiVLM:
    m = cfg["model"]
    return GeminiVLM(m["model_id"], m["api_version"], m["requests_per_minute"], m["rate_limit_retries"])


# READ SINGLE PHOTO

class PerceptionError(RuntimeError):
    pass


def ask_until_agree(vlm, cfg: dict, system: str, user: str, image: bytes,
                    schema, parse, label: str):
    votes, max_attempts = cfg["model"]["votes"], cfg["model"]["max_attempts"]
    counts: dict[str, int] = {}
    for attempt in range(1, max_attempts + 1):
        try:
            raw = vlm.generate_json(system, [image], user, schema)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[{label} {attempt}] bad response: {e}", file=sys.stderr)
            continue
        result, errors = parse(raw)
        if errors:
            print(f"[{label} {attempt}] rejected: {errors}", file=sys.stderr)
            continue
        key = json.dumps(result, sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
        print(f"[{label} {attempt}] valid ({counts[key]}/{votes} matching)", file=sys.stderr)
        if counts[key] >= votes:
            return result
    raise PerceptionError(f"{label}: no {votes} matching valid answers in {max_attempts} attempts")


def read_grid(vlm, cfg: dict, photo, allow_stacks: bool = False, label: str = "grid") -> dict:
    system, user = grid_prompt(cfg, allow_stacks)
    schema = StackedGridReading if allow_stacks else GridReading
    image = load_jpeg(photo, cfg["model"]["max_image_px"])
    return ask_until_agree(vlm, cfg, system, user, image, schema,
                           lambda raw: parse_grid(raw, cfg, label, allow_stacks), label)


def read_shape(vlm, cfg: dict, photo, label: str = "goal") -> dict:
    system, user = shape_prompt(cfg)
    image = load_jpeg(photo, cfg["model"]["max_image_px"])
    return ask_until_agree(vlm, cfg, system, user, image, ShapeReading,
                           lambda raw: parse_shape(raw, cfg, label), label)
