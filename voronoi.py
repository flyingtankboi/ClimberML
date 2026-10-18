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

@dataclass
class VoronoiDiagram:
    """Indexed Voronoi diagram clipped to a [0, width] x [0, height] wall.

    vertices is the single source of points; everything else is indices into it.
    cells[i] is the polygon (counter-clockwise vertex indices) around sites[i].
    """
    sites: list[Point2D]
    vertices: list[Point2D]
    edges: list[Edge]
    cells: list[list[int]]

    def segments(self) -> list[Segment2D]:
        """Edges as standalone segments (handy for plotting)."""
        return [Segment2D(self.vertices[a], self.vertices[b]) for a, b in self.edges]


def _clip_polygon(polygon: list[Point2D], width: float, height: float) -> list[Point2D]:
    """Sutherland-Hodgman clipping of a convex polygon to [0, width] x [0, height]."""
    # each border as (side value function): inside when value >= 0
    borders = (
        lambda p: p.x,            # left
        lambda p: width - p.x,    # right
        lambda p: p.y,            # bottom
        lambda p: height - p.y,   # top
    )
    for side in borders:
        if not polygon:
            break
        clipped: list[Point2D] = []
        for k, cur in enumerate(polygon):
            nxt = polygon[(k + 1) % len(polygon)]
            s_cur, s_nxt = side(cur), side(nxt)
            if s_cur >= 0:
                clipped.append(cur)
            if (s_cur >= 0) != (s_nxt >= 0):          # edge crosses the border
                t = s_cur / (s_cur - s_nxt)
                clipped.append(cur + (nxt - cur) * t)
        polygon = clipped
    return polygon


def delaunay_voronoi(points: list[Point2D], width: float, height: float,
                     decimals: int = 9) -> tuple[list[Triangle], VoronoiDiagram]:
    """Compute the Delaunay triangulation and the indexed Voronoi diagram in one pass.

    A site's Voronoi cell is the polygon of circumcenters of the triangles around
    it, sorted by angle. Triangles touching the super-triangle are kept so border
    cells are closed (far away), then every cell is clipped to the wall.
    Points created by clipping are shared between neighbouring cells by rounding
    their coordinates to `decimals` digits.
    """
    n = len(points)
    _, triangles = _bowyer_watson(points, width, height)

    tris_around: dict[int, list[Triangle]] = defaultdict(list)
    for t in triangles:
        for i in t:
            if i < n:
                tris_around[i].append(t)

    vertices: list[Point2D] = []
    vertex_index: dict[tuple[float, float], int] = {}

    def add_vertex(p: Point2D) -> int:
        key = (round(p.x, decimals) + 0.0, round(p.y, decimals) + 0.0)  # + 0.0 turns -0.0 into 0.0
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append(Point2D(*key))
        return vertex_index[key]

    cells: list[list[int]] = []
    edges: set[Edge] = set()
    for i, site in enumerate(points):
        centers = [triangles[t][0] for t in tris_around[i]]
        centers.sort(key=lambda c: math.atan2(c.y - site.y, c.x - site.x))
        polygon = _clip_polygon(centers, width, height)

        cell: list[int] = []
        for p in polygon:
            v = add_vertex(p)
            if not cell or cell[-1] != v:               # drop repeated vertices
                cell.append(v)
        if len(cell) > 1 and cell[0] == cell[-1]:
            cell.pop()
        cells.append(cell)

        for k in range(len(cell)):
            a, b = cell[k], cell[(k + 1) % len(cell)]
            if a != b:
                edges.add(make_edge(a, b))

    delaunay_tris = [t for t in triangles if max(t) < n]
    diagram = VoronoiDiagram(list(points), vertices, sorted(edges), cells)
    return delaunay_tris, diagram


def voronoi(points: list[Point2D], width: float, height: float) -> VoronoiDiagram:
    """Indexed Voronoi diagram of points, clipped to the wall rectangle."""
    return delaunay_voronoi(points, width, height)[1]


# ---------------------------------------------------------------- plotting

def plot(points: list[Point2D], triangles: list[Triangle], diagram: VoronoiDiagram,
         width: float, height: float) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    for a, b, c in triangles:
        xs = [points[i].x for i in (a, b, c, a)]
        ys = [points[i].y for i in (a, b, c, a)]
        ax.plot(xs, ys, color="tab:blue", linewidth=0.8, alpha=0.6)
    for a, b in diagram.edges:
        p, q = diagram.vertices[a], diagram.vertices[b]
        ax.plot([p.x, q.x], [p.y, q.y], color="tab:red", linewidth=1.2)
    ax.scatter([p.x for p in points], [p.y for p in points], color="black", s=12, zorder=3)
    ax.set_aspect("equal")
    ax.set_title("Delaunay (blue) and Voronoi (red)")
    plt.show()


if __name__ == "__main__":
    W, H = 10, 10
    pts = poisson_sampling(W, H, 1.0, 60, max_tries=100, seed=0)
    tris, diagram = delaunay_voronoi(pts, W, H)
    print(f"{len(pts)} points, {len(tris)} triangles, "
          f"{len(diagram.vertices)} Voronoi vertices, {len(diagram.edges)} edges")
    plot(pts, tris, diagram, W, H)
