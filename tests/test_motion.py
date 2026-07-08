import numpy as np

from sentinel_edge.detect import MotionGate


def frame(square_x=None):
    f = np.full((360, 640, 3), 40, dtype=np.uint8)
    if square_x is not None:
        f[150:210, square_x:square_x + 60] = 255
    return f


def test_static_scene_suppresses_at_least_80pct():
    gate = MotionGate()
    for _ in range(50):
        gate.process(frame())
    assert gate.suppression_rate >= 0.8


def test_moving_object_passes_the_gate():
    gate = MotionGate()
    gate.process(frame(0))
    passed = sum(gate.process(frame(x)) for x in range(10, 300, 10))
    assert passed >= 25  # essentially every moving frame


def test_object_stopping_absorbs_slowly_not_instantly():
    gate = MotionGate()
    gate.process(frame())
    for x in range(0, 200, 10):
        gate.process(frame(x))
    # object halts: the slowed learn rate keeps it visible for a good while
    # (the pipeline's active-track bypass owns long-stationary objects), then
    # the background absorbs it and the scene goes quiet again
    still = [gate.process(frame(200)) for _ in range(300)]
    assert still[0]                       # not instantly absorbed
    assert not still[-1]                  # eventually suppressed again
    assert gate.last_fraction < gate.min_fraction


def test_resolution_change_resets_background():
    gate = MotionGate()
    gate.process(frame())
    assert gate.process(np.zeros((180, 320, 3), dtype=np.uint8))  # new shape passes
