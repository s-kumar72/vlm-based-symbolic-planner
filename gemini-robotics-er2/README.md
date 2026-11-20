# Gemini Robotics-ER 2 

VLM implementation using Gemini Robotics-ER 2. Given 2 input photos of initial state and goal state, it sends each photo to Google's Gemini Robotics-ER 2 model and returns where each object is both as a grid cell and as a position in mm. This info can be used by the planner.

The VLM pipeline can be tested using the images in test_images/ (see [Testing](#testing)).

---

## Setup

Requires python >= 3.10.


### 1. Get a Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com) and sign in with a Google account.
2. Save API key using 'Get API key' -> 'Create API key'

**NOTE**: On the free tier, Google lets you send about 5 requests per minute to this model & there's also a daily limit (Gemini for students doesn't give more credits for AI Studio unfortunately). The code has a built in timer to slow down API calls & will error if the daily limit has been reached. 

### 2. Get the code

```bash
git clone https://github.com/s-kumar72/vlm-based-symbolic-planner.git
cd vlm-based-symbolic-planner
cd gemini-robotics-er2
```

### 3. Make a virtual environment using requirements.txt

### 4. Export API key

```bash
export GEMINI_API_KEY="paste-key-here"
```

### 5. Test baseline functionality

This prints the instructions that get sent to the model (does not use API but can test if everything was imported properly). 

```bash
python scene_to_goal.py --show-prompt
```

---

## Grid

Instead of hardcoding where each object starts, we put the objects on a grid with known measurements and let the model say which cell each object is in.

The test grid (used in test_images.py) is a sheet of letter paper split into a 3 × 3 grid with blue tape. Cells are counted from the photo's point of view:

```
          col 0    col 1    col 2
row 2   [      ] [      ] [      ]     <- top of the photo
row 1   [      ] [      ] [      ]
row 0   [      ] [      ] [      ]     <- bottom of the photo
```

Cell (0, 0) is the grid origin. It doesn't have a physical marker right now but if one is added that can be noted in `grid_config.toml` (toml file used instead of yaml because it let me add comments).

The model only answers "which cell is each token in?" The code turns the cells into mm using the cell size set in `grid_config.toml`

All of the grid settings are in `grid_config.toml`:

- `rows`, `cols`: how many cells.
- `cell_width_mm`, `cell_height_mm`: size of one cell. Right now these are rough estimates from the ruler photos (`test_images/grid_dimensions_x.JPG` and `_y.JPG`), about 72 × 93 mm. 
- `axis_hint`: a sentence telling the model which corner is (0, 0) and which way rows and columns go.
- `names`: the exact token names the model is allowed to use (can also set this to an empty list for which the model will generate its own names based on its      interpretation of the objects).
- `grid_origin_mm`, `grid_rotation_deg`: where the grid sits relative to the robot.

### General process 

1. The photo gets rotated upright and shrunk (this is because I did not rotate the photos before uploading to GitHub but should help for most phone photos). 
2. The photo goes to the model with a short set of instructions (the grid size, which corner is the origin, and the allowed token names). The model has to reply in a fixed format which is listed as part of the API response schema in `vlm.py`.
3. The code checks the answer via the ground truth toml file in test_images. If the model names a token that isn't on the list, or puts something outside the grid, that answer gets thrown out and we ask again.
4. Right now it keeps asking until two answers match to catch one-off errors but this can be changed in `votes` in `grid_config.toml` (also to help with the daily request limit). 

---

## 2D vs 3D

You can switch between 2d and 3d by changing `mode` in `grid_config.toml`, or by adding `--mode 2d` or `--mode 3d` when the command is run in terminal.

### 2D (moving tokens around the grid)

- Both photos show the grid.
- The goal photo is the same grid with the tokens moved to different cells.
- For each token you get its current cell and its goal cell, both with mm positions, and a list of which tokens need to move.
- Multiple tokens can share a cell.

Example output (cut down to two tokens):

```json
{
  "mode": "2d",
  "current": {
    "pink token":   {"cell": [2, 1], "xy_mm": [108.0, 232.5]},
    "yellow token": {"cell": [0, 1], "xy_mm": [108.0, 46.5]}
  },
  "goal": {
    "pink token":   {"cell": [1, 1], "xy_mm": [108.0, 139.5]},
    "yellow token": {"cell": [0, 0], "xy_mm": [36.0, 46.5]}
  },
  "changed": ["brown token", "lime green token", "pink token", "purple token", "yellow token"]
}
```

### 3D (stacking)

- The current photo (initial state) shows the grid with tokens & one token can be resting on top of another token. 
- The goal photo shows the shape we want to build. When the model creates the goal state, only the shape matters (not where it is on the grid). The model will output what each object is resting on plus the L-R order of the bottom row. 
- Extra settings in the `[stacking]` section can be changed in the toml file: `spacing_mm` (distance between neighboring cups in a row) and `level_height_mm` (how much height each level adds). (Just added some values for right now but will need to be remeasured).

So far 3D has only been tried with the flat test photos. Photo D has a purple token overlapping the pink one to replicate "stacking" but testing should still be done with the actual cups (or whatever object we'll stack).

---

## Testing

### The test photos

The photos in `test_images/` (A to I) were all taken of the same 3 × 3 grid, with six coloured tokens: pink, orange, lime green, brown, yellow, and purple. Each photo has the tokens in a different arrangement. I tried to add some difficult features or ones that would test 3D without the cups:

- **B**: purple and yellow share a cell.
- **D**: three tokens in the middle cell, with purple partly on top of pink.
- **E**: no purple token (errors).
- **I**: three tokens in the middle cell.

### Running the tests

From inside gemini-robotics-er2 folder:

```bash
python test_vlm.py --pair A C        # current = photo A, goal = photo C
python test_vlm.py                   # a random pair
python test_vlm.py --trials 3        # 3 random pairs
python test_vlm.py --all             # every photo once, with an overall accuracy at the end
python test_vlm.py --all --mode 3d   # same, but also checks which tokens are on top of others
```
### Trying other photos

```bash
python scene_to_goal.py current.jpg goal.jpg
python scene_to_goal.py current.jpg goal.jpg -o states.json   # also saves the result
```
