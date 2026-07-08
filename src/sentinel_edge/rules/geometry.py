"""Pure geometry for the rule engine. Everything operates in pixel space;
normalized contract coordinates are denormalized once at compile time.

Conventions (image coordinates, y grows downward):
- A "zone" polygon has >= 3 points; a 2-point polygon is a directed line A->B.
- side(p) = (Bx-Ax)(py-Ay) - (By-Ay)(px-Ax).
  Crossing from side<0 to side>0 is direction "lr"; the opposite is "rl".
  (The dashboard renders the A->B arrow so operators pick direction visually.)
"""
from __future__ import annotations

from typing import Optional, Sequence

Point = tuple[float, float]


def denormalize(polygon: Sequence[Sequence[float]], width: int, height: int) -> list[Point]:
    return [(float(x) * width, float(y) * height) for x, y in polygon]


def point_in_polygon(p: Point, polygon: Sequence[Point]) -> bool:
    """Ray-cast; points exactly on an edge count as inside (stable for
    presence tracking — a track sitting on the boundary doesn't flicker)."""
    x, y = p
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # edge hit -> inside
        if _on_segment((x, y), (xi, yi), (xj, yj)):
            return True
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _on_segment(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > eps * max(1.0, abs(b[0] - a[0]) + abs(b[1] - a[1])):
        return False
    dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    sq_len = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return -eps <= dot <= sq_len + eps


def line_side(p: Point, a: Point, b: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing_direction(
    p_prev: Point,
    p_curr: Point,
    a: Point,
    b: Point,
    overshoot: float = 0.0,
) -> Optional[str]:
    """Return "lr" / "rl" if the step p_prev->p_curr crosses the SEGMENT a->b,
    else None.

    Segment-segment intersection rather than a side-sign flip: no minimum or
    maximum step size, no miss when a point lands exactly on the line, and no
    false fire on the line's infinite extension. `overshoot` extends the
    segment by that fraction beyond each endpoint.
    """
    dx, dy = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
    lx, ly = b[0] - a[0], b[1] - a[1]
    denom = dx * ly - dy * lx
    if denom == 0:  # parallel (or zero movement)
        return None
    fx, fy = a[0] - p_prev[0], a[1] - p_prev[1]
    t = (fx * ly - fy * lx) / denom  # position along the step
    u = (fx * dy - fy * dx) / denom  # position along the line segment
    if 0.0 <= t <= 1.0 and -overshoot <= u <= 1.0 + overshoot:
        # side goes <0 -> >0 exactly when denom < 0
        return "lr" if denom < 0 else "rl"
    return None


def polygon_is_line(polygon: Sequence[Sequence[float]]) -> bool:
    return len(polygon) == 2


def validate_zone_polygon(polygon: Sequence[Sequence[float]]) -> None:
    """Contract-level validation happens in Pydantic; this adds the geometric
    check that a >=3-point polygon is not degenerate (zero area)."""
    if len(polygon) == 2:
        (x1, y1), (x2, y2) = polygon
        if x1 == x2 and y1 == y2:
            raise ValueError("line has zero length")
        return
    area2 = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    if abs(area2) < 1e-12:
        raise ValueError("polygon has zero area")
