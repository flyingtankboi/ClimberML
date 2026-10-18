"""Lift a 2D Voronoi diagram into a 3D wall surface with a height function, and place holds on it."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from voronoi import Edge, Point2D, VoronoiDiagram

type HeightFunction = Callable[[float, float], float]  # (x, y) -> z
type Triangle3D = tuple[Point3D, Point3D, Point3D]


# ---------------------------------------------------------------- geometry

@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> Point3D:
        return Point3D(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def __neg__(self) -> Point3D:
        return Point3D(-self.x, -self.y, -self.z)

    def dot(self, other: Point3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Point3D) -> Point3D:
        return Point3D(self.y * other.z - self.z * other.y,
                       self.z * other.x - self.x * other.z,
                       self.x * other.y - self.y * other.x)

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Point3D:
        n = self.length()
        if n == 0:
            raise ValueError("cannot normalize a zero vector")
        return self * (1 / n)

    def distance(self, other: Point3D) -> float:
        return (self - other).length()

    def distance_sq(self, other: Point3D) -> float:
        d = self - other
        return d.dot(d)

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


def _barycentric(x: float, y: float, a: Point3D, b: Point3D, c: Point3D
                 ) -> tuple[float, float, float] | None:
    """Barycentric weights of (x, y) in triangle abc, using only x and y."""
    d = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
    if abs(d) < 1e-12:
        return None                                   # flat (degenerate) triangle
    w1 = ((x - a.x) * (c.y - a.y) - (c.x - a.x) * (y - a.y)) / d
    w2 = ((b.x - a.x) * (y - a.y) - (x - a.x) * (b.y - a.y)) / d
    return 1.0 - w1 - w2, w1, w2


# ---------------------------------------------------------------- surface

@dataclass
class VoronoiDiagram3D:
    """Same topology as the 2D diagram (edges, cells), with lifted points.

    Convention: x, y span the wall plane (y is up), z is how far it sticks out.
    """
    sites: list[Point3D]
    vertices: list[Point3D]
    edges: list[Edge]
    cells: list[list[int]]

    def faces(self) -> list[Triangle3D]:
        """Triangles covering the surface: each cell is fanned around its site.

        Lifted cells are not flat, so they can't be a single polygon face.
        """
        return [tri for i in range(len(self.sites)) for tri in self._cell_triangles(i)]

    def _cell_triangles(self, i: int) -> list[Triangle3D]:
        site, cell = self.sites[i], self.cells[i]
        return [(site, self.vertices[cell[k]], self.vertices[cell[(k + 1) % len(cell)]])
                for k in range(len(cell))]

    def _locate(self, x: float, y: float) -> tuple[Triangle3D, tuple[float, float, float]]:
        """Triangle under (x, y) and the barycentric weights of (x, y) in it.

        (x, y) lies in the Voronoi cell of its nearest site, so only that cell's
        fan triangles need checking. Falls back to every triangle if rounding
        puts the point just outside them (e.g. exactly on a cell border).
        The best triangle is the one whose smallest weight is largest
        (>= 0 means inside).
        """
        def best(triangles: list[Triangle3D]):
            found, best_score = None, -math.inf
            for tri in triangles:
                w = _barycentric(x, y, *tri)
                if w is not None and min(w) > best_score:
                    found, best_score = (tri, w), min(w)
            return found, best_score

        nearest = min(range(len(self.sites)),
                      key=lambda i: (self.sites[i].x - x) ** 2 + (self.sites[i].y - y) ** 2)
        found, score = best(self._cell_triangles(nearest))
        if found is None or score < -1e-9:
            found, score = best(self.faces())
        if found is None:
            raise ValueError(f"no triangle found under ({x}, {y})")
        return found

    def height_at(self, x: float, y: float) -> float:
        """Height of the mesh surface at (x, y), interpolated on the triangle below."""
        (a, b, c), w = self._locate(x, y)
        return w[0] * a.z + w[1] * b.z + w[2] * c.z

    def normal_at(self, x: float, y: float) -> Point3D:
        """Unit normal of the face under (x, y), pointing out of the wall (z > 0)."""
        (a, b, c), _ = self._locate(x, y)
        n = (b - a).cross(c - a).normalized()
        return n if n.z > 0 else -n

    def lift(self, p: Point2D) -> Point3D:
        """Put a 2D point exactly on the wall surface."""
        return Point3D(p.x, p.y, self.height_at(p.x, p.y))

    def lift_points(self, points: list[Point2D]) -> list[Point3D]:
        return [self.lift(p) for p in points]


# ---------------------------------------------------------------- wall and holds

@dataclass(frozen=True)
class Hold:
    position: Point3D   # on the wall surface
    normal: Point3D     # unit vector, direction the hold faces


@dataclass
class Wall:
    surface: VoronoiDiagram3D
    holds: list[Hold]

    @classmethod
    def from_points(cls, surface: VoronoiDiagram3D, hold_points: list[Point2D]) -> Wall:
        """Place a hold at each 2D point, on the surface, facing out of its face."""
        holds = [Hold(surface.lift(p), surface.normal_at(p.x, p.y)) for p in hold_points]
        return cls(surface, holds)


# ---------------------------------------------------------------- lifting

def voronoi_to_3d(diagram: VoronoiDiagram, height: HeightFunction) -> VoronoiDiagram3D:
    """Give every point of the diagram a z = height(x, y). Topology is unchanged."""
    def lift(p: Point2D) -> Point3D:
        return Point3D(p.x, p.y, height(p.x, p.y))

    return VoronoiDiagram3D(
        sites=[lift(s) for s in diagram.sites],
        vertices=[lift(v) for v in diagram.vertices],
        edges=list(diagram.edges),
        cells=[list(c) for c in diagram.cells],
    )


# ---------------------------------------------------------------- height fields

class HeightField:
    """A height function you can combine with +, -, * (and plain numbers)."""

    def __init__(self, f: HeightFunction):
        self.f = f

    def __call__(self, x: float, y: float) -> float:
        return self.f(x, y)

    def __add__(self, other: HeightFunction | float) -> HeightField:
        return HeightField(lambda x, y: self(x, y) + _call(other, x, y))

    def __mul__(self, other: HeightFunction | float) -> HeightField:
        return HeightField(lambda x, y: self(x, y) * _call(other, x, y))

    def __sub__(self, other: HeightFunction | float) -> HeightField:
        return HeightField(lambda x, y: self(x, y) - _call(other, x, y))

    def __neg__(self) -> HeightField:
        return HeightField(lambda x, y: -self(x, y))

    def __radd__(self, other: HeightFunction | float) -> HeightField:
        return HeightField(lambda x, y: _call(other, x, y) + self(x, y))

    def __rmul__(self, other: HeightFunction | float) -> HeightField:
        return HeightField(lambda x, y: _call(other, x, y) * self(x, y))

    def __rsub__(self, other: HeightFunction | float) -> HeightField:
        return HeightField(lambda x, y: _call(other, x, y) - self(x, y))


def _call(h: HeightFunction | float, x: float, y: float) -> float:
    return h(x, y) if callable(h) else h   # lets you add or multiply by plain numbers


def perlin_height(feature_size: float = 3.0, amplitude: float = 1.0,
                  octaves: int = 4, seed: int | None = None) -> HeightField:
    """Height function from Perlin noise.

    feature_size: rough width of one bump, in wall units.
    amplitude: scales the noise output (about -0.5..0.5) to wall units.
    """
    from perlin_noise import PerlinNoise

    noise = PerlinNoise(octaves=octaves, seed=seed)

    def height(x: float, y: float) -> float:
        return amplitude * noise([x / feature_size, y / feature_size])

    return HeightField(height)


def linear_bias_height(weight: float = 0.1) -> HeightField:
    """Tilt the wall along y: weight > 0 is an overhang, < 0 a slab."""
    def height(x: float, y: float) -> float:
        return weight * y

    return HeightField(height)


# ---------------------------------------------------------------- plotting

def plot_3d(wall: Wall, normal_length: float = 0.4) -> None:
    """Draw the wall standing up: data (x, y, z) is shown as matplotlib (x, z, y),
    so y (up the wall) is vertical and z (sticking out) points toward the viewer.
    Holds are drawn as dots with a short line along their normal.
    """
    def screen(p: Point3D) -> tuple[float, float, float]:   # data -> matplotlib axes
        return (p.x, p.z, p.y)

    surface = wall.surface
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False          # draw in the order below, so holds stay on top

    faces = [[screen(p) for p in tri] for tri in surface.faces()]
    ax.add_collection3d(Poly3DCollection(faces, facecolor="tan", edgecolor="none", alpha=0.9))
    for a, b in surface.edges:
        p, q = screen(surface.vertices[a]), screen(surface.vertices[b])
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="saddlebrown", linewidth=0.8)

    for hold in wall.holds:
        p = screen(hold.position)
        q = screen(hold.position + hold.normal * normal_length)
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="tab:green", linewidth=1.2)
    if wall.holds:
        hx, hy, hz = zip(*(screen(h.position) for h in wall.holds))
        ax.scatter(hx, hy, hz, color="tab:green", edgecolor="black", s=25, depthshade=False)

    xs, ys, zs = zip(*(screen(v) for v in surface.vertices))
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_zlim(min(zs), max(zs))
    ax.set_xlabel("x (across)")
    ax.set_ylabel("z (sticks out)")
    ax.set_zlabel("y (up)")
    ax.set_aspect("equal")
    ax.view_init(elev=10, azim=-60)     # look at the wall from the front, slightly to the side
    plt.show()


if __name__ == "__main__":
    from voronoi import poisson_sampling, voronoi

    W, H = 10, 10
    pts = poisson_sampling(W, H, 1.0, 60, max_tries=100, seed=0)
    height = perlin_height(feature_size=10.0, amplitude=1.4, octaves=4, seed=0) + linear_bias_height(weight=0.1)

    surface = voronoi_to_3d(voronoi(pts, W, H), height)
    wall = Wall.from_points(surface, poisson_sampling(W, H, 0.5, 100, max_tries=100, seed=1))

    print(f"{len(surface.vertices)} vertices, {len(surface.edges)} edges, "
          f"{len(surface.faces())} faces, {len(wall.holds)} holds")

    plot_3d(wall)
