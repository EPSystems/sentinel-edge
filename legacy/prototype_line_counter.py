import cv2
import torch
import numpy as np
import subprocess

from types import SimpleNamespace

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

from tracker.byte_tracker import BYTETracker

# ============================================================
# CONFIG -- everything you normally need to touch is here
# ============================================================

RTSP_INPUT = "rtsp://localhost:9554/safecam"
RTSP_OUTPUT = "rtsp://localhost:9554/tracked"
OUTPUT_FPS = 25

SHOW_PREVIEW = True        # local cv2 window (needs a display attached)
PUBLISH_RTSP = True        # push the annotated stream back out via ffmpeg

YOLOX_EXP = "/home/pavel/YOLOX/exps/default/yolox_s.py"
YOLOX_WEIGHTS = "/home/pavel/YOLOX/pretrained/yolox_s.pth"
DEVICE = "cuda"

# Counting line, in source-feed pixels (feed here is 1920x1080).
LINE_P1 = (771, 755)
LINE_P2 = (1051, 764)

# COCO class ids to detect, track and count.
# person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
COUNT_CLASSES = {0: "person"}
# COUNT_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Which point of the bounding box has to cross the line:
#   "bottom" - bottom-centre (feet / wheels). Use when the line is on the
#              ground: a person's box CENTRE is at hip height and may never
#              reach a ground line even though they clearly walk over it.
#   "center" - box centre. Use for a gate/doorway viewed side-on.
ANCHOR = "bottom"

# Labels for the two crossing directions (which is which depends on the
# order of LINE_P1/LINE_P2 -- swap the labels if they come out inverted).
DIR_POS_LABEL = "IN"
DIR_NEG_LABEL = "OUT"

MIN_DET_SCORE = 0.10       # detector floor fed to the tracker. Keep LOW:
                           # ByteTrack relies on low-confidence boxes for
                           # association; its own thresholds do the gating.
TRACK_THRESH = 0.50        # ByteTrack: score needed to confirm a track
TRACK_BUFFER = 30          # frames a lost track is kept alive
MATCH_THRESH = 0.80

MIN_TRACK_AGE = 3          # frames a track must exist before it may count
                           # (filters one-frame false detections)
MAX_JUMP_PX = 300          # anchor jump above this between two frames is
                           # treated as an ID glitch and not counted
LINE_OVERSHOOT = 0.05      # also count crossings up to this fraction past
                           # the segment endpoints (0 = exact segment only)

# ============================================================
# YOLOX
# ============================================================

exp = get_exp(YOLOX_EXP, None)

model = exp.get_model()

ckpt = torch.load(YOLOX_WEIGHTS, map_location="cpu")
model.load_state_dict(ckpt["model"])

model.to(DEVICE)
model.eval()

preproc = ValTransform()

print("YOLOX loaded")

# ============================================================
# BYTETRACK
# ============================================================

args = SimpleNamespace(
    track_thresh=TRACK_THRESH,
    track_buffer=TRACK_BUFFER,
    match_thresh=MATCH_THRESH,
    mot20=False
)

tracker = BYTETracker(args, frame_rate=OUTPUT_FPS)

CLASS_IDS = np.array(list(COUNT_CLASSES.keys()))

print("ByteTrack loaded")

# ============================================================
# RTSP INPUT
# ============================================================

cap = cv2.VideoCapture(RTSP_INPUT)

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Cannot open RTSP stream")

h, w = frame.shape[:2]

print(f"Resolution: {w}x{h}")

# ============================================================
# RTSP OUTPUT
# ============================================================

ffmpeg = None

if PUBLISH_RTSP:

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(OUTPUT_FPS),
        "-i", "-",

        "-an",

        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",

        "-pix_fmt", "yuv420p",

        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        RTSP_OUTPUT
    ]

    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE
    )

# ============================================================
# GEOMETRY
# ============================================================

LP1 = np.array(LINE_P1, dtype=float)
LP2 = np.array(LINE_P2, dtype=float)
LINE_VEC = LP2 - LP1


def anchor_point(x1, y1, x2, y2):
    if ANCHOR == "bottom":
        return np.array([(x1 + x2) / 2.0, y2])
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def crossing_direction(p_prev, p_curr):
    """+1 / -1 if the step p_prev -> p_curr intersects the counting
    segment (in either direction), 0 if it does not.

    Segment-segment intersection instead of a side-sign flip: it has no
    minimum/maximum speed, handles a point landing exactly on the line,
    and only fires on the drawn segment -- not its infinite extension.
    """
    d = p_curr - p_prev

    denom = d[0] * LINE_VEC[1] - d[1] * LINE_VEC[0]

    if denom == 0:  # moving parallel to the line
        return 0

    diff = LP1 - p_prev

    t = (diff[0] * LINE_VEC[1] - diff[1] * LINE_VEC[0]) / denom  # along step
    u = (diff[0] * d[1] - diff[1] * d[0]) / denom                # along line

    if 0.0 <= t <= 1.0 and -LINE_OVERSHOOT <= u <= 1.0 + LINE_OVERSHOOT:
        return 1 if denom > 0 else -1

    return 0


# ============================================================
# COUNTER STATE
# ============================================================

track_state = {}    # id -> {"pos": np.array, "age": int, "seen": frame_idx}
counted_ids = set()
counts = {1: 0, -1: 0}

frame_idx = 0

# ============================================================
# LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    img, _ = preproc(
        frame,
        None,
        exp.test_size
    )

    img = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():

        outputs = model(img)

        outputs = postprocess(
            outputs,
            exp.num_classes,
            exp.test_conf,
            exp.nmsthre
        )

    # --------------------------------------------------------
    # keep only the classes we count, as [x1, y1, x2, y2, score]
    # --------------------------------------------------------

    if outputs[0] is not None:
        dets = outputs[0].cpu().numpy()
        scores = dets[:, 4] * dets[:, 5]
        keep = np.isin(dets[:, 6].astype(int), CLASS_IDS) & (scores >= MIN_DET_SCORE)
        dets = np.concatenate([dets[keep, :4], scores[keep, None]], axis=1)
    else:
        dets = np.zeros((0, 5), dtype=np.float32)

    # always update the tracker (also on empty frames) so lost tracks age out
    online_targets = tracker.update(dets, [h, w], exp.test_size)

    for t in online_targets:

        tlwh = t.tlwh
        tid = int(t.track_id)

        x1 = int(tlwh[0])
        y1 = int(tlwh[1])
        x2 = int(tlwh[0] + tlwh[2])
        y2 = int(tlwh[1] + tlwh[3])

        pt = anchor_point(x1, y1, x2, y2)

        state = track_state.get(tid)

        if state is None:
            track_state[tid] = {"pos": pt, "age": 1, "seen": frame_idx}
        else:
            state["age"] += 1
            state["seen"] = frame_idx

            jump = float(np.hypot(*(pt - state["pos"])))

            if (
                state["age"] > MIN_TRACK_AGE
                and jump <= MAX_JUMP_PX
                and tid not in counted_ids
            ):
                direction = crossing_direction(state["pos"], pt)

                if direction != 0:
                    counted_ids.add(tid)
                    counts[direction] += 1

                    label = DIR_POS_LABEL if direction > 0 else DIR_NEG_LABEL
                    print(
                        f"CROSSING {label}! id={tid} "
                        f"{DIR_POS_LABEL}={counts[1]} "
                        f"{DIR_NEG_LABEL}={counts[-1]} "
                        f"TOTAL={counts[1] + counts[-1]}"
                    )

            state["pos"] = pt

        # ------------------------------------------------
        # draw
        # ------------------------------------------------

        color = (0, 165, 255) if tid in counted_ids else (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.circle(
            frame,
            (int(pt[0]), int(pt[1])),
            4,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"#{tid}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    # --------------------------------------------------------
    # drop state of tracks not seen for a while
    # --------------------------------------------------------

    if frame_idx % 100 == 0:
        stale = [
            tid for tid, s in track_state.items()
            if frame_idx - s["seen"] > TRACK_BUFFER * 2
        ]
        for tid in stale:
            track_state.pop(tid)
            counted_ids.discard(tid)

    # --------------------------------------------------------
    # line + counters
    # --------------------------------------------------------

    cv2.line(
        frame,
        LINE_P1,
        LINE_P2,
        (255, 0, 0),
        3
    )

    cv2.circle(frame, LINE_P1, 6, (255, 0, 0), -1)
    cv2.circle(frame, LINE_P2, 6, (255, 0, 0), -1)

    cv2.putText(
        frame,
        f"{DIR_POS_LABEL}: {counts[1]}  "
        f"{DIR_NEG_LABEL}: {counts[-1]}  "
        f"TOTAL: {counts[1] + counts[-1]}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    # --------------------------------------------------------
    # local preview
    # --------------------------------------------------------

    if SHOW_PREVIEW:

        cv2.imshow("YOLOX + ByteTrack", frame)

        key = cv2.waitKey(1)

        if key == 27:
            break

    # --------------------------------------------------------
    # RTSP publish
    # --------------------------------------------------------

    if ffmpeg is not None:
        ffmpeg.stdin.write(frame.tobytes())

# ============================================================
# CLEANUP
# ============================================================

cap.release()

if ffmpeg is not None:
    ffmpeg.stdin.close()
    ffmpeg.wait()

cv2.destroyAllWindows()
