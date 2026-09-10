"""
Vehicle Counting Agent
------------------------
Watches a video feed of traffic, tracks vehicles across frames, and counts
each one exactly once as it crosses a virtual counting line.

Agent loop:
    PERCEIVE  -> YOLO11 detects + tracks vehicles each frame
    REMEMBER  -> per-track memory: which side of the line it was last on,
                 how long we have seen it, and a running vote on its class
    DECIDE    -> has this vehicle genuinely crossed the line, and is this
                 track trustworthy enough to count?
    ACT       -> increment the counter, draw the HUD, log to console

Usage:
    python vehicle_counter.py --source path/to/traffic.mp4 --output out.mp4
    python vehicle_counter.py --source 0                       # webcam
    python vehicle_counter.py --source traffic.mp4 --line 0.5  # line at 50% of frame height
    python vehicle_counter.py --source traffic.mp4 --debug     # print every detection + confidence
    python vehicle_counter.py --source traffic.mp4 --debug --all-classes  # see ALL classes YOLO finds, unfiltered
"""

import argparse
import math
import cv2
from ultralytics import YOLO

# ---- COCO class IDs for vehicles -------------------------------------------
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
WATCHED_CLASS_IDS = list(VEHICLE_CLASSES.keys())

# ---- HUD colours (BGR) ------------------------------------------------------
LINE_COLOR = (0, 0, 255)
BAND_COLOR = (0, 140, 255)
HUD_TEXT = (255, 255, 255)
HUD_TOTAL = (0, 220, 255)
DOT_PENDING = (0, 255, 255)
DOT_COUNTED = (0, 255, 0)


class VehicleCounter:
    """The agent's MEMORY and DECISION logic.

    Memory is one record per track id. Each record holds the side of the line
    the vehicle was last confidently on, the frames over which we have seen it,
    a vote tally over the classes YOLO assigned it, and whether it has been
    counted.
    """

    def __init__(self, line_y, band=10, min_age=3, forget_after=125):
        # line_y      : y pixel of the counting line
        # band        : dead zone (+/- px) around the line. A vehicle must
        #               travel clear through it to register a crossing, so box
        #               jitter on a stopped vehicle cannot trigger a count.
        # min_age     : a track must have existed this many frames before it is
        #               allowed to count. A brand-new id appearing right at the
        #               line is usually a tracker id-switch, not a new vehicle.
        # forget_after: drop a track record this many frames after we last saw
        #               it, so memory stays bounded on long inputs.
        self.line_y = line_y
        self.band = band
        self.min_age = min_age
        self.forget_after = forget_after

        self.tracks = {}
        self.counts_by_class = {name: 0 for name in VEHICLE_CLASSES.values()}
        self.total = 0

    # -- REMEMBER -------------------------------------------------------------
    def _side(self, cy):
        """Which side of the line, or None while inside the dead band."""
        if cy < self.line_y - self.band:
            return "above"
        if cy > self.line_y + self.band:
            return "below"
        return None

    @staticmethod
    def _majority_class(record):
        """The class YOLO called this vehicle most often over its whole life.

        Taking the class from the single frame of the crossing is fragile:
        YOLO readily flips car <-> truck <-> bus frame to frame. Voting over
        the track's history is far more stable.
        """
        cls_id = max(record["votes"].items(), key=lambda kv: kv[1])[0]
        return VEHICLE_CLASSES[cls_id]

    # -- DECIDE ---------------------------------------------------------------
    def observe(self, track_id, cls_id, cy, frame_idx):
        """Feed one detection in. Returns (class_name, direction) if this
        observation completed a countable crossing, else None."""
        record = self.tracks.get(track_id)
        if record is None:
            record = self.tracks[track_id] = {
                "votes": {}, "side": None, "first": frame_idx,
                "last": frame_idx, "counted": False,
            }
        record["last"] = frame_idx
        record["votes"][cls_id] = record["votes"].get(cls_id, 0) + 1

        side = self._side(cy)
        if side is None:
            return None          # inside the dead band: hold the old side, decide nothing

        previous, record["side"] = record["side"], side

        if previous is None or previous == side:
            return None          # first confident sighting, or no change of side
        if record["counted"]:
            return None          # already counted; never double-count a track
        if frame_idx - record["first"] < self.min_age:
            return None          # too young to trust

        record["counted"] = True
        name = self._majority_class(record)
        self.counts_by_class[name] += 1
        self.total += 1
        direction = "downward" if side == "below" else "upward"
        return name, direction

    def forget_stale(self, frame_idx):
        """Evict tracks we have not seen for a while. Without this both the
        side memory and the counted-id set grow without bound."""
        stale = [tid for tid, record in self.tracks.items()
                 if frame_idx - record["last"] > self.forget_after]
        for tid in stale:
            del self.tracks[tid]

    def is_counted(self, track_id):
        record = self.tracks.get(track_id)
        return bool(record and record["counted"])


# -- ACT ----------------------------------------------------------------------
def draw_hud(frame, agent, height):
    """Counts panel, bottom-left: clear of YOLO's own labels (which stack at
    the top of the frame), the camera watermark, and the timestamp."""
    rows = [("TOTAL", agent.total, True)]
    rows += [(name, count, False) for name, count in agent.counts_by_class.items()]

    pad, line_h, panel_w = 12, 26, 190
    panel_h = pad * 2 + line_h * len(rows)
    x0, y0 = 12, height - panel_h - 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)

    y = y0 + pad + 18
    for label, value, is_total in rows:
        colour = HUD_TOTAL if is_total else HUD_TEXT
        scale = 0.72 if is_total else 0.6
        cv2.putText(frame, f"{label}: {value}", (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
        y += line_h


def draw_line(frame, agent, width):
    """The counting line plus its dead band, so the geometry is visible."""
    cv2.line(frame, (0, agent.line_y), (width, agent.line_y), LINE_COLOR, 2, cv2.LINE_AA)
    for offset in (-agent.band, agent.band):
        cv2.line(frame, (0, agent.line_y + offset), (width, agent.line_y + offset),
                 BAND_COLOR, 1, cv2.LINE_AA)


def run(source, output_path, model_path="yolo11n.pt", line_fraction=0.6,
        show=False, conf=0.30, debug=False, all_classes=False,
        band=10, min_age=3, tracker="botsort.yaml"):
    model = YOLO(model_path)
    COCO_NAMES = model.names  # dict of {class_id: class_name} for ALL 80 COCO classes

    # when --all-classes is on, don't restrict what YOLO looks for (diagnostic mode)
    track_classes = None if all_classes else WATCHED_CLASS_IDS

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0                                 # some sources report 0 fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    line_y = int(height * line_fraction)

    agent = VehicleCounter(line_y=line_y, band=band, min_age=min_age,
                           forget_after=int(fps * 5))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():                  # previously failed silently, leaving a 0-byte file
            cap.release()
            raise RuntimeError(f"Could not open video writer for: {output_path}")

    frames_note = f"  |  {total_frames} frames" if total_frames else ""
    print(f"Source {source}  |  {width}x{height} @ {fps:.1f} fps{frames_note}")
    print(f"Counting line at y={line_y} (+/-{band}px band), conf>={conf}, model={model_path}\n")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # ---- PERCEIVE ----
        results = model.track(
            frame,
            persist=True,
            classes=track_classes,
            conf=conf,
            tracker=tracker,
            verbose=False,
        )[0]

        # --- DEBUG: print every detection this frame (enable with --debug) ---
        if debug and results.boxes is not None:
            for cls_id, conf_score in zip(
                results.boxes.cls.int().cpu().tolist(),
                results.boxes.conf.cpu().tolist(),
            ):
                cls_name = COCO_NAMES.get(cls_id, f"class_{cls_id}")
                print(f"  frame {frame_idx}: detected {cls_name} conf={conf_score:.2f}")

        just_crossed = []
        centroids = []   # (x, y, already_counted) for the marker dots

        # ---- REMEMBER + DECIDE ----
        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id, cls_id in zip(
                results.boxes.xywh.cpu(),
                results.boxes.id.int().cpu().tolist(),
                results.boxes.cls.int().cpu().tolist(),
            ):
                # Only feed the counting logic classes we actually track/count.
                # (Matters when --all-classes is on and YOLO returns things
                # like "person" or "bicycle" that aren't in VEHICLE_CLASSES.)
                if cls_id not in VEHICLE_CLASSES:
                    continue
                cx, cy = float(box[0]), float(box[1])
                crossing = agent.observe(track_id, cls_id, cy, frame_idx)
                if crossing:
                    just_crossed.append((track_id, *crossing))
                centroids.append((int(cx), int(cy), agent.is_counted(track_id)))

        agent.forget_stale(frame_idx)

        for track_id, cls_name, direction in just_crossed:
            print(f"[COUNTED] frame {frame_idx}: {cls_name} (id {track_id}) crossed "
                  f"{direction}. Total so far: {agent.total}")

        # ---- ACT: draw everything ----
        annotated = results.plot()
        draw_line(annotated, agent, width)
        for cx, cy, counted in centroids:
            cv2.circle(annotated, (cx, cy), 4,
                       DOT_COUNTED if counted else DOT_PENDING, -1, cv2.LINE_AA)
        draw_hud(annotated, agent, height)

        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("Vehicle Counting Agent", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if total_frames and frame_idx % 100 == 0:
            print(f"  ... {frame_idx}/{total_frames} frames, total so far: {agent.total}")

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nDone. Processed {frame_idx} frames.")
    print(f"Total vehicles counted: {agent.total}")
    for cls_name, count in agent.counts_by_class.items():
        print(f"  {cls_name}: {count}")
    if output_path:
        print(f"Annotated video written to: {output_path}")

    return agent.counts_by_class, agent.total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Counting Agent")
    parser.add_argument("--source", required=True, help="Video file path or '0' for webcam")
    parser.add_argument("--output", default="output.mp4",
                        help="Path to save annotated video (pass an empty string to skip writing)")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    parser.add_argument("--line", type=float, default=0.6,
                         help="Counting line position as a fraction of frame height (0.0=top, 1.0=bottom)")
    parser.add_argument("--conf", type=float, default=0.30, help="Minimum detection confidence (0.0-1.0)")
    parser.add_argument("--band", type=int, default=10,
                         help="Dead-band in pixels either side of the line; a vehicle must clear it to count")
    parser.add_argument("--min-age", type=int, default=3,
                         help="Frames a track must survive before it may count (guards against id-switches)")
    parser.add_argument("--tracker", default="botsort.yaml",
                         help="Ultralytics tracker config (botsort.yaml or bytetrack.yaml)")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--debug", action="store_true", help="Print every detection + confidence to the terminal")
    parser.add_argument("--all-classes", action="store_true",
                         help="Diagnostic: don't restrict detection to vehicle classes, show everything YOLO finds")
    args = parser.parse_args()

    run(args.source, args.output, args.model, args.line, args.show, args.conf,
        args.debug, args.all_classes, args.band, args.min_age, args.tracker)
