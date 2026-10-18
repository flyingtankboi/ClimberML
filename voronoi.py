"""Poisson-disk sampling, Delaunay triangulation (Bowyer-Watson) and Voronoi diagram."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

type Triangle = tuple[int, int, int]  # indices into a points list
type Edge = tuple[int, int]           # always (smaller, larger), build with make_edge


# ---------------------------------------------------------------- geometry

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> Point2D:
        return Point2D(self.x * k, self.y * k)

    __rmul__ = __mul__

    def dot(self, other: Point2D) -> float:
        return self.x * other.x + self.y * other.y

    def distance(self, other: Point2D) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def distance_sq(self, other: Point2D) -> float:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f})"


class Segment2D:
    """Unordered segment: Segment2D(a, b) == Segment2D(b, a)."""

    def __init__(self, p1: Point2D, p2: Point2D):
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment2D):
            return NotImplemented
        return {self.p1, self.p2} == {other.p1, other.p2}

    def __hash__(self) -> int:
        return hash(frozenset((self.p1, self.p2)))

    def __str__(self) -> str:
        return f"Seg[{self.p1} - {self.p2}]"

    __repr__ = __str__


def make_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def circumcircle(a: Point2D, b: Point2D, c: Point2D) -> tuple[Point2D, float]:
    """Return (center, radius squared) of the circle through a, b, c."""
    d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
    if abs(d) < 1e-12:
        raise ValueError("points are collinear")
    a2, b2, c2 = a.dot(a), b.dot(b), c.dot(c)
    ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d
    uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d
    center = Point2D(ux, uy)
    return center, center.distance_sq(a)


# ---------------------------------------------------------------- sampling

def poisson_sampling(width: float, height: float, min_distance: float, num_points: int,
                     max_tries: int = 30, seed: int | None = None) -> list[Point2D]:
    rng = np.random.default_rng(seed)
    points: list[Point2D] = []
    cell_size = min_distance / 2

    cells_per_width = int(width // cell_size + 1)
    cells_per_height = int(height // cell_size + 1)
    cells = [[-1] * cells_per_width for _ in range(cells_per_height)]

    def is_valid(p: Point2D, x_cell: int, y_cell: int) -> bool:
        if cells[y_cell][x_cell] != -1:
            return False
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x_n, y_n = x_cell + dx, y_cell + dy
                if not (0 <= x_n < cells_per_width and 0 <= y_n < cells_per_height):
                    continue
                idx = cells[y_n][x_n]
                if idx != -1 and p.distance(points[idx]) < min_distance:
                    return False
        return True

    failed_tries = 0
    while len(points) < num_points and failed_tries <= max_tries:
        p = Point2D(rng.random() * width, rng.random() * height)
        x_cell, y_cell = int(p.x // cell_size), int(p.y // cell_size)
        if is_valid(p, x_cell, y_cell):
            cells[y_cell][x_cell] = len(points)
            points.append(p)
            failed_tries = 0
        else:
            failed_tries += 1
    return points


# ---------------------------------------------------------------- delaunay

def _bowyer_watson(points: list[Point2D], width: float, height: float
                   ) -> tuple[list[Point2D], dict[Triangle, tuple[Point2D, float]]]:
    """Full triangulation, super-triangle included.

    Returns all points (input points, then the 3 super-triangle points)
    and every triangle mapped to its circumcircle (center, radius squared).
    """
    n = len(points)
    # The super-triangle must be huge: if one of its corners falls inside the
    # circumcircle of a thin triangle on the hull, that triangle goes missing.
    # Even at this size, extremely flat slivers on the hull can still be lost.
    size = 10000 * max(width, height)
    cx, cy = width / 2, height / 2
    all_points = list(points) + [
        Point2D(cx - size, cy - size),
        Point2D(cx + size, cy - size),
        Point2D(cx, cy + size),
    ]
    super_tri: Triangle = (n, n + 1, n + 2)
    triangles = {super_tri: circumcircle(*(all_points[i] for i in super_tri))}

    for i, p in enumerate(points):
        # 1. triangles whose circumcircle contains p
        bad = [t for t, (c, r2) in triangles.items() if p.distance_sq(c) < r2 - 1e-12]

        # 2. boundary of the cavity = edges used by exactly one bad triangle
        edge_count: dict[Edge, int] = defaultdict(int)
        for a, b, c in bad:
            for e in (make_edge(a, b), make_edge(b, c), make_edge(a, c)):
                edge_count[e] += 1
        boundary = [e for e, k in edge_count.items() if k == 1]

        # 3. remove bad triangles, 4. connect p to each boundary edge
        for t in bad:
            del triangles[t]
        for a, b in boundary:
            t = (a, b, i)
            triangles[t] = circumcircle(all_points[a], all_points[b], p)

    return all_points, triangles


def delaunay(points: list[Point2D], width: float, height: float) -> list[Triangle]:
    """Delaunay triangulation of points lying in [0, width] x [0, height]."""
    n = len(points)
    _, triangles = _bowyer_watson(points, width, height)
    return [t for t in triangles if max(t) < n]


# ---------------------------------------------------------------- voronoi

def clip_segment(p: Point2D, q: Point2D, width: float, height: float) -> Segment2D | None:
    """Liang-Barsky clipping of segment pq to [0, width] x [0, height]."""
    dx, dy = q.x - p.x, q.y - p.y
    t0, t1 = 0.0, 1.0
    for denom, num in ((-dx, p.x), (dx, width - p.x), (-dy, p.y), (dy, height - p.y)):
        if denom == 0:
            if num < 0:
                return None          # parallel to this border and outside
            continue
        t = num / denom
        if denom < 0:
            t0 = max(t0, t)          # entering
        else:
            t1 = min(t1, t)          # leaving
        if t0 > t1:
            return None
    return Segment2D(p + (q - p) * t0, p + (q - p) * t1)


def voronoi(points: list[Point2D], width: float, height: float) -> list[Segment2D]:
    """Voronoi edges of points, clipped to the wall rectangle.

    Each Delaunay edge shared by two triangles gives one Voronoi edge between
    their circumcenters. Triangles touching the super-triangle are kept here:
    their far-away circumcenters stand in for the infinite rays on the border.
    """
    _, triangles = _bowyer_watson(points, width, height)

    edge_to_tris: dict[Edge, list[Triangle]] = defaultdict(list)
    for t in triangles:
        a, b, c = t
        for e in (make_edge(a, b), make_edge(b, c), make_edge(a, c)):
            edge_to_tris[e].append(t)

    segments: list[Segment2D] = []
    for tris in edge_to_tris.values():
        if len(tris) != 2:
            continue                 # outer edge of the super-triangle
        c1, c2 = triangles[tris[0]][0], triangles[tris[1]][0]
        seg = clip_segment(c1, c2, width, height)
        if seg is not None and seg.p1 != seg.p2:
            segments.append(seg)
    return segments


# ---------------------------------------------------------------- plotting

def plot(points: list[Point2D], triangles: list[Triangle], segments: list[Segment2D],
         width: float, height: float) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    for a, b, c in triangles:
        xs = [points[i].x for i in (a, b, c, a)]
        ys = [points[i].y for i in (a, b, c, a)]
        ax.plot(xs, ys, color="tab:blue", linewidth=0.8, alpha=0.6)
    for s in segments:
        ax.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y], color="tab:red", linewidth=1.2)
    ax.scatter([p.x for p in points], [p.y for p in points], color="black", s=12, zorder=3)
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color="gray")
    ax.set_aspect("equal")
    ax.set_title("Delaunay (blue) and Voronoi (red)")
    plt.show()


if __name__ == "__main__":
    W, H = 10, 10
    pts = poisson_sampling(W, H, 1.0, 60, max_tries=100, seed=0)
    tris = delaunay(pts, W, H)
    segs = voronoi(pts, W, H)
    print(f"{len(pts)} points, {len(tris)} triangles, {len(segs)} Voronoi edges")
    plot(pts, tris, segs, W, H)
