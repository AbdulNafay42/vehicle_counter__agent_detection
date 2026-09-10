# Vehicle Counting Agent

Counts vehicles in traffic footage using **YOLO11** for detection and multi-object
tracking for identity, so each vehicle is counted **exactly once** as it crosses a
virtual counting line.

Built as an agent: it perceives the scene, remembers what it has seen, decides
whether a genuine crossing occurred, and acts on that decision.

## Results

On the included `traffic.mp4` (1280x720, 25 fps, ~48 s of CCTV):

| Class      | Count |
| ---------- | ----- |
| car        | 32    |
| bus        | 3     |
| motorcycle | 1     |
| truck      | 0     |
| **Total**  | **36** |

`truck: 0` is correct, not a bug. The only truck in the clip (track id 4) is
already past the counting line in frame 1 at cy=358 and travels away from it, so
it never crosses.

## Getting started

Requires **Python 3.10 or newer** (developed on 3.13) and git.

```bash
# 1. Clone
git clone https://github.com/AbdulNafay42/vehicle_counter__agent_detection.git
cd vehicle_counter__agent_detection

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate            # Windows (cmd)
.\venv\Scripts\Activate.ps1     # Windows (PowerShell)
source venv/bin/activate         # macOS / Linux

# 4. Install dependencies (~2.5 GB, mostly torch)
pip install -r requirements.txt

# 5. Run it
python vehicle_counter.py --source traffic.mp4 --output result.mp4
```

Model weights are not committed. Ultralytics downloads `yolo11n.pt` (5.6 MB)
automatically on the first run, so step 5 needs an internet connection the
first time.

On CPU the sample clip takes a few minutes with `yolo11n.pt`, longer with
`yolo11s.pt`. Progress prints every 100 frames.

### If PowerShell blocks activation

Windows PowerShell may refuse to run `Activate.ps1`. Either use the cmd form
above, or allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Usage

```bash
# Basic run
python vehicle_counter.py --source traffic.mp4 --output result.mp4

# Larger, more accurate model
python vehicle_counter.py --source traffic.mp4 --output result.mp4 --model yolo11s.pt

# Live preview window, no file written
python vehicle_counter.py --source traffic.mp4 --output "" --show

# Webcam
python vehicle_counter.py --source 0
```

### Options

| Flag            | Default        | Description                                                |
| --------------- | -------------- | ---------------------------------------------------------- |
| `--source`      | *(required)*   | Video file path, or `0` for webcam                          |
| `--output`      | `output.mp4`   | Annotated video path; pass an empty string to skip writing  |
| `--model`       | `yolo11n.pt`   | YOLO weights (`yolo11n` = fast, `yolo11s` = more accurate)   |
| `--line`        | `0.6`          | Counting line as a fraction of frame height                 |
| `--conf`        | `0.30`         | Minimum detection confidence                                |
| `--band`        | `10`           | Dead-band in pixels either side of the line                 |
| `--min-age`     | `3`            | Frames a track must survive before it may count             |
| `--tracker`     | `botsort.yaml` | Tracker config (`botsort.yaml` or `bytetrack.yaml`)          |
| `--show`        | off            | Live preview window                                         |
| `--debug`       | off            | Print every detection and confidence                        |
| `--all-classes` | off            | Diagnostic: detect all 80 COCO classes, unfiltered          |

## How it works

### The agent loop

| Step         | What happens                                                            |
| ------------ | ----------------------------------------------------------------------- |
| **PERCEIVE** | YOLO11 detects vehicles; the tracker assigns each a persistent id        |
| **REMEMBER** | One record per track: last side of the line, age, and a class vote tally |
| **DECIDE**   | Did this vehicle genuinely cross, and is the track trustworthy?          |
| **ACT**      | Increment the count, draw the overlay, log to console                    |

### Detection and tracking

YOLO11 is a single-stage detector: one forward pass over the frame returns every
box, class and confidence. It is pretrained on COCO, and only four class ids are
kept:

```python
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
```

Detection alone cannot count, because YOLO has no memory across frames. A car
visible for three seconds produces 75 unrelated detections. `model.track(persist=True)`
adds a tracker (BoT-SORT) that predicts where each known vehicle should appear
next and matches it to the new detections, giving every vehicle a stable integer
id. That id is what makes counting-once possible.

### Counting reliably

Three guards keep the count honest:

**Dead band.** A vehicle is "above" only once its centroid clears `line_y - band`,
and "below" only once it clears `line_y + band`. Inside the band nothing is
decided. YOLO boxes breathe by several pixels every frame, so a bare threshold
would let a stationary vehicle sitting on the line flip sides and register a
crossing it never made.

**Minimum track age.** A track must survive 3 frames before it may count. A
brand-new id appearing right at the line is usually a tracker id-switch, not a
new vehicle.

**Majority-vote classification.** The counted class is whichever class YOLO
assigned the track most often across its whole life, not the class in the single
crossing frame. YOLO readily flips car / truck / bus between frames.

Track records are evicted 5 seconds after last sighting, so memory stays bounded
on long inputs.

### Reading the output video

- **Red line** - the counting line
- **Orange lines** - the dead-band edges at `line_y +/- band`
- **Yellow dot** - a tracked vehicle's centroid, not yet counted
- **Green dot** - this track has been counted and cannot count again

The dot is the exact pixel the counting logic tests, so every count can be
verified by eye.

## Known limitations

- The line spans the full frame width, so both carriageways feed a single total.
  Two of the 36 counts are opposite-carriageway traffic.
- Vehicles already past the line when the video starts are never counted, so the
  true throughput of the clip is 36 plus whatever had already passed.
- Counting uses the box centre, so a long vehicle counts when its middle crosses,
  not its front bumper.
- If a track vanishes for over 5 seconds and the tracker reissues the same id, it
  could be counted twice. This never triggers on the sample clip; it is the
  trade-off for bounded memory.
