"""
K-means model for the teaching demo.

Pure Python, no UI code, so it can be used by the Streamlit app, the original
tkinter app, or unit tests. Every step returns a text explanation with the full
arithmetic (distances, means and the objective J) for showing in class.
"""

import math
import random

MAX_K = 6

WORKED_EXAMPLE = [("A", 3, 3), ("B", 5, 8), ("C", 4, 1), ("D", 4, 7), ("E", 6, 6), ("F", 5, 2)]
WORKED_START = [[5, 8], [6, 6]]  # start at B and E, as in the slides

DATASETS = ["Example", "Random blobs", "Empty (add your own)"]
INIT_METHODS = ["First K points", "Random points", "k-means++", "Manual (click the plot)"]


def fmt(v):
    """Short number format: 4.50 -> 4.5, 3.00 -> 3."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def point_name(i):
    """0 -> A, 25 -> Z, 26 -> AA, ..."""
    letters = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        letters = chr(65 + r) + letters
    return letters


def next_free_name(points):
    used = {p[0] for p in points}
    i = 0
    while point_name(i) in used:
        i += 1
    return point_name(i)


def make_blobs(k, per_cluster=7, spread=0.7):
    """K well-separated Gaussian blobs inside the 0-10 square."""
    centers = []
    while len(centers) < k:
        c = (random.uniform(2, 8), random.uniform(2, 8))
        if all(math.hypot(c[0] - a, c[1] - b) > 3.0 for a, b in centers) or len(centers) > 40:
            centers.append(c)
    pts = []
    for cx, cy in centers:
        for _ in range(per_cluster):
            pts.append([None, round(random.gauss(cx, spread), 1), round(random.gauss(cy, spread), 1)])
    random.shuffle(pts)
    for i, p in enumerate(pts):
        p[0] = point_name(i)
    return pts


def plot_limits(points, worked=False):
    if worked:
        return (0, 9, 0, 9)
    if not points:
        return (0, 10, 0, 10)
    vals = [p[1] for p in points] + [p[2] for p in points]
    lo = min(0, math.floor(min(vals)) - 1)
    hi = max(10, math.ceil(max(vals)) + 1)
    return (lo, hi, lo, hi)


class KMeansModel:
    def __init__(self):
        self.points = []          # list of [name, x, y]
        self.k = 2
        self.centroids = []
        self.clear_run()

    def clear_run(self):
        """Forget assignments but keep points (and centroids)."""
        self.labels = [None] * len(self.points)
        self.changed = set()
        self.iteration = 0
        self.phase = "ready" if len(self.centroids) == self.k and self.k > 0 else "no centroids"
        self.J_hist = []
        self.prev_centroids = None

    # --- helpers
    @staticmethod
    def dist(p, c):
        return math.hypot(p[1] - c[0], p[2] - c[1])

    def distance_matrix(self):
        return [[self.dist(p, c) for c in self.centroids] for p in self.points]

    def distance_formula(self, i, j):
        name, x, y = self.points[i]
        cx, cy = self.centroids[j]
        dx2, dy2 = (x - cx) ** 2, (y - cy) ** 2
        return (f"d({name}, μ{j + 1}) = √(({fmt(x)} − {fmt(cx)})² + ({fmt(y)} − {fmt(cy)})²)"
                f" = √({fmt(dx2)} + {fmt(dy2)}) = √{fmt(dx2 + dy2)} = {fmt(math.sqrt(dx2 + dy2))}")

    def next_step_name(self):
        return {"no centroids": "Initialise", "ready": "Assign", "assigned": "Update",
                "updated": "Assign", "converged": "Done"}[self.phase]

    def explain_point(self, i):
        """Distances from one point to every centroid, written out."""
        name, x, y = self.points[i]
        lines = [f"POINT {name}  (x = {fmt(x)}, y = {fmt(y)})", ""]
        if len(self.centroids) == self.k:
            D = [self.dist(self.points[i], c) for c in self.centroids]
            for j in range(self.k):
                lines.append("  " + self.distance_formula(i, j))
            best = min(range(self.k), key=D.__getitem__)
            lines.append(f"\n  Smallest distance: μ{best + 1}  →  {name} belongs to cluster {best + 1}")
        else:
            lines.append("  Place the centroids first to see the distances.")
        return "\n".join(lines) + "\n"

    # --- initialisation
    def init_centroids(self, method):
        pts = [(p[1], p[2]) for p in self.points]
        if len(pts) < self.k:
            raise ValueError(f"Need at least K = {self.k} points to initialise.")
        if method == "First K points":
            cs = pts[:self.k]
        elif method == "Random points":
            cs = random.sample(pts, self.k)
        elif method == "k-means++":
            cs = [random.choice(pts)]
            while len(cs) < self.k:
                d2 = [min((px - cx) ** 2 + (py - cy) ** 2 for cx, cy in cs) for px, py in pts]
                total = sum(d2)
                if total == 0:
                    cs.append(random.choice(pts))
                    continue
                r, acc = random.uniform(0, total), 0.0
                for p, w in zip(pts, d2):
                    acc += w
                    if acc >= r:
                        cs.append(p)
                        break
        else:
            raise ValueError("Manual initialisation: click the plot to place the centroids.")
        return self.set_centroids([list(c) for c in cs], method)

    def set_centroids(self, centroids, method="Manual"):
        self.centroids = [[float(c[0]), float(c[1])] for c in centroids]
        self.clear_run()
        names = []
        for c in self.centroids:
            match = [p[0] for p in self.points if (p[1], p[2]) == tuple(c)]
            names.append(f"{match[0]} " if match else "")
        return ("INITIALISE  (" + method + ")\n" +
                "\n".join(f"  μ{j + 1} = {names[j]}({fmt(c[0])}, {fmt(c[1])})"
                          for j, c in enumerate(self.centroids)) +
                "\n\nPress Step to run the first ASSIGN step.\n")

    # --- the two K-means steps
    def assign(self):
        D = self.distance_matrix()
        new = [min(range(self.k), key=lambda j: D[i][j]) for i in range(len(self.points))]
        first = self.iteration == 0 or any(lab is None for lab in self.labels)
        self.changed = set() if first else {i for i, (a, b) in enumerate(zip(self.labels, new)) if a != b}
        self.labels = new
        self.iteration += 1
        J = sum(D[i][new[i]] ** 2 for i in range(len(new)))
        self.J_hist.append(J)
        self.prev_centroids = None
        self.phase = "converged" if (not first and not self.changed) else "assigned"

        n = len(self.points)
        lines = [f"ITERATION {self.iteration} · ASSIGN — each point joins its nearest centroid", ""]
        for i, p in enumerate(self.points):
            if n <= 8:
                for j in range(self.k):
                    lines.append("  " + self.distance_formula(i, j))
            ds = ", ".join(f"μ{j + 1}: {fmt(D[i][j])}" for j in range(self.k))
            tag = "   ← changed" if i in self.changed else ""
            lines.append(f"  {p[0]}  →  cluster {new[i] + 1}   ({ds}){tag}")
            if n <= 8:
                lines.append("")
        lines.append("")
        lines.append(self.j_formula(D))
        if self.phase == "converged":
            lines.append("\n✔ No point changed cluster → K-means has CONVERGED.")
        else:
            if not first:
                changed = ", ".join(self.points[i][0] for i in sorted(self.changed)) or "—"
                lines.append(f"\nPoints that changed cluster: {changed}")
            lines.append("Next: UPDATE (move each centroid to the mean of its points).")
        return "\n".join(lines) + "\n"

    def j_formula(self, D):
        sq = [D[i][self.labels[i]] ** 2 for i in range(len(self.points))]
        J = sum(sq)
        if len(sq) <= 12:
            return "J = Σ d²(point, own centroid) = " + " + ".join(fmt(s) for s in sq) + f" = {fmt(J)}"
        return f"J = Σ d²(point, own centroid) over {len(sq)} points = {fmt(J)}"

    def update(self):
        old = [c[:] for c in self.centroids]
        lines = [f"ITERATION {self.iteration} · UPDATE — each centroid moves to the mean of its points", ""]
        new = []
        for j in range(self.k):
            members = [p for p, lab in zip(self.points, self.labels) if lab == j]
            if not members:
                new.append(old[j][:])
                lines.append(f"  Cluster {j + 1} is empty → μ{j + 1} stays at "
                             f"({fmt(old[j][0])}, {fmt(old[j][1])})\n")
                continue
            m = len(members)
            sx, sy = sum(p[1] for p in members), sum(p[2] for p in members)
            mx, my = sx / m, sy / m
            names = ", ".join(p[0] for p in members)
            lines.append(f"  Cluster {j + 1} = {{{names}}}")
            if m <= 8:
                lines.append(f"    x̄ = ({' + '.join(fmt(p[1]) for p in members)}) / {m} = {fmt(sx)} / {m} = {fmt(mx)}")
                lines.append(f"    ȳ = ({' + '.join(fmt(p[2]) for p in members)}) / {m} = {fmt(sy)} / {m} = {fmt(my)}")
            else:
                lines.append(f"    x̄ = (sum of {m} x-values) / {m} = {fmt(sx)} / {m} = {fmt(mx)}")
                lines.append(f"    ȳ = (sum of {m} y-values) / {m} = {fmt(sy)} / {m} = {fmt(my)}")
            lines.append(f"    μ{j + 1}: ({fmt(old[j][0])}, {fmt(old[j][1])})  →  ({fmt(mx)}, {fmt(my)})\n")
            new.append([mx, my])
        self.prev_centroids = old
        self.centroids = new
        self.phase = "updated"
        lines.append("Next: ASSIGN again with the new centroids.")
        return "\n".join(lines) + "\n"

    def step(self):
        if self.phase in ("ready", "updated"):
            return self.assign()
        if self.phase == "assigned":
            return self.update()
        return ""
