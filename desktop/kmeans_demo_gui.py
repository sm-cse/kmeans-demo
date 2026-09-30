"""
K-Means Clustering - Interactive Teaching Demo
==============================================

A classroom tool that runs K-means one step at a time and shows every number:
  * the plot shows each point's (x, y) and each centroid's position
  * the Points table shows the distance of every point to every centroid,
    the chosen cluster (the smaller distance is ticked) and the squared distance
  * the Centroids table shows each centroid's coordinates and members
  * the Calculations panel writes out the full formulas for every step
    (Euclidean distances for the Assign step, means for the Update step, and J)

Requirements:  Python 3.8+  and  matplotlib      ->   pip install matplotlib
(tkinter ships with Python; on Ubuntu/Debian:  sudo apt install python3-tk)

Run:   python kmeans_demo_gui.py

Controls
  Step (Space)      next step: Assign, then Update, then Assign, ...
  Run / Stop        play all steps automatically until convergence
  Initialise        place new starting centroids with the chosen method
  Left-click plot   add a point (or select one if you click on it)
  Right-click plot  remove the nearest point
  Double-click row  edit a point's x, y values
  Manual init       choose "Manual (click)" then click K places on the plot
"""

import math
import random
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PALETTE = ["#2b5cb8", "#b8336a", "#e08a1e", "#2f9e7a", "#7b4fc9", "#8a6d3b"]
LIGHT = ["#dbe5f6", "#f5dbe6", "#fde9cf", "#d7efe6", "#e7ddf7", "#eee5d7"]
GREY = "#b9bdc5"
INK = "#1c2230"
MAX_K = 6

WORKED_EXAMPLE = [("A", 3, 3), ("B", 5, 8), ("C", 4, 1), ("D", 4, 7), ("E", 6, 6), ("F", 5, 2)]
DATASETS = ["Worked example (6 points)", "Random blobs", "Empty (click to add)"]
INIT_METHODS = ["First K points", "Random points", "k-means++", "Manual (click)"]


def fmt(v):
    """Short number format: 4.50 -> 4.5, 3.00 -> 3."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def point_name(i):
    letters = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        letters = chr(65 + r) + letters
    return letters


# --------------------------------------------------------------------------- model
class KMeansModel:
    def __init__(self):
        self.points = []          # list of [name, x, y]
        self.k = 2
        self.clear_run()
        self.centroids = []

    def clear_run(self):
        """Forget assignments but keep points (and centroids)."""
        self.labels = [None] * len(self.points)
        self.changed = set()
        self.iteration = 0
        self.phase = "ready" if getattr(self, "centroids", []) else "no centroids"
        self.J_hist = []
        self.prev_centroids = None

    # --- helpers
    @staticmethod
    def dist(p, c):
        return math.hypot(p[1] - c[0], p[2] - c[1])

    def distance_matrix(self):
        return [[self.dist(p, c) for c in self.centroids] for p in self.points]

    def distance_formula(self, i, j):
        _, x, y = self.points[i]
        cx, cy = self.centroids[j]
        dx2, dy2 = (x - cx) ** 2, (y - cy) ** 2
        name = self.points[i][0]
        return (f"d({name}, μ{j + 1}) = √(({fmt(x)} − {fmt(cx)})² + ({fmt(y)} − {fmt(cy)})²)"
                f" = √({fmt(dx2)} + {fmt(dy2)}) = √{fmt(dx2 + dy2)} = {fmt(math.sqrt(dx2 + dy2))}")

    def next_step_name(self):
        return {"no centroids": "Initialise", "ready": "Assign", "assigned": "Update",
                "updated": "Assign", "converged": "Done"}[self.phase]

    # --- initialisation
    def init_centroids(self, method):
        pts = [(p[1], p[2]) for p in self.points]
        if len(pts) < self.k:
            raise ValueError(f"Need at least K = {self.k} points.")
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
            raise ValueError("Manual initialisation is done by clicking on the plot.")
        self.centroids = [list(c) for c in cs]
        self.clear_run()
        self.phase = "ready"
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
        first = self.iteration == 0 or any(l is None for l in self.labels)
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
            changed = ", ".join(self.points[i][0] for i in sorted(self.changed)) or "—"
            if not first:
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
            members = [p for p, l in zip(self.points, self.labels) if l == j]
            if not members:
                new.append(old[j][:])
                lines.append(f"  Cluster {j + 1} is empty → μ{j + 1} stays at ({fmt(old[j][0])}, {fmt(old[j][1])})\n")
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


# --------------------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("K-Means Clustering — Interactive Teaching Demo")
        self.geometry("1500x880")
        self.minsize(1200, 720)
        ttk.Style(self).theme_use("clam")
        self.model = KMeansModel()
        self.busy = False          # animation in progress
        self.running = False       # auto-run
        self.selected = None       # selected point index
        self.lims = (0, 10, 0, 10)
        self._build_ui()
        self.load_dataset()
        self.bind("<space>", lambda e: self.on_step())

    # --- layout
    def _build_ui(self):
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="Data:").pack(side=tk.LEFT)
        self.dataset_var = tk.StringVar(value=DATASETS[0])
        cb = ttk.Combobox(bar, textvariable=self.dataset_var, values=DATASETS, width=24, state="readonly")
        cb.pack(side=tk.LEFT, padx=(4, 12))
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_dataset())

        ttk.Label(bar, text="K:").pack(side=tk.LEFT)
        self.k_var = tk.IntVar(value=2)
        sp = ttk.Spinbox(bar, from_=1, to=MAX_K, textvariable=self.k_var, width=4, command=self.on_k_change)
        sp.pack(side=tk.LEFT, padx=(4, 12))
        sp.bind("<Return>", lambda e: self.on_k_change())

        ttk.Label(bar, text="Start:").pack(side=tk.LEFT)
        self.init_var = tk.StringVar(value=INIT_METHODS[0])
        ttk.Combobox(bar, textvariable=self.init_var, values=INIT_METHODS, width=15,
                     state="readonly").pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(bar, text="① Initialise", command=self.on_init).pack(side=tk.LEFT, padx=4)
        self.step_btn = ttk.Button(bar, text="Step ▶", command=self.on_step)
        self.step_btn.pack(side=tk.LEFT, padx=4)
        self.run_btn = ttk.Button(bar, text="Run ▶▶", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Reset", command=self.on_reset).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="New data", command=self.load_dataset).pack(side=tk.LEFT, padx=4)

        bar2 = ttk.Frame(self, padding=(10, 0, 10, 6))
        bar2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar2, text="Animation speed:").pack(side=tk.LEFT)
        self.speed = tk.DoubleVar(value=1.0)
        ttk.Scale(bar2, from_=0.3, to=3.0, variable=self.speed, length=110).pack(side=tk.LEFT, padx=4)
        self.show_coords = tk.BooleanVar(value=True)
        self.show_lines = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar2, text="Show (x, y) labels", variable=self.show_coords,
                        command=self.redraw).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(bar2, text="Show distance lines", variable=self.show_lines,
                        command=self.redraw).pack(side=tk.LEFT, padx=6)

        ttk.Label(bar2, text="Left-click: add/select point · Right-click: remove · Double-click a row: edit x, y · Space: step",
                  foreground="#5b6475").pack(side=tk.RIGHT)
        info = ttk.Frame(self, padding=(10, 0))
        info.pack(side=tk.TOP, fill=tk.X)
        self.status = tk.StringVar()
        self.next_lbl = tk.StringVar()
        ttk.Label(info, textvariable=self.status, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.next_lbl, font=("Segoe UI", 12), foreground="#b86a12").pack(side=tk.RIGHT)

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # plot
        left = ttk.Frame(body)
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        body.add(left, weight=3)

        # tables + log
        right = ttk.Frame(body)
        body.add(right, weight=4)
        ttk.Label(right, text="Points  (double-click a row to edit x, y)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        pf = ttk.Frame(right)
        pf.pack(fill=tk.BOTH, expand=True)
        self.ptree = ttk.Treeview(pf, show="headings", height=10)
        ys = ttk.Scrollbar(pf, orient=tk.VERTICAL, command=self.ptree.yview)
        self.ptree.configure(yscrollcommand=ys.set)
        self.ptree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        self.ptree.bind("<<TreeviewSelect>>", self.on_row_select)
        self.ptree.bind("<Double-1>", self.on_row_edit)
        for j in range(MAX_K):
            self.ptree.tag_configure(f"c{j}", background=LIGHT[j])
        self.ptree.tag_configure("changed", background="#ffd28f")

        ttk.Label(right, text="Centroids", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 0))
        self.ctree = ttk.Treeview(right, show="headings", height=4,
                                  columns=("c", "x", "y", "n", "members"))
        for col, txt, w in (("c", "Centroid", 80), ("x", "x", 80), ("y", "y", 80),
                            ("n", "Count", 60), ("members", "Members", 380)):
            self.ctree.heading(col, text=txt)
            self.ctree.column(col, width=w, anchor="center" if col != "members" else "w")
        for j in range(MAX_K):
            self.ctree.tag_configure(f"c{j}", background=LIGHT[j])
        self.ctree.pack(fill=tk.X)

        ttk.Label(right, text="Calculations", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 0))
        lf = ttk.Frame(right)
        lf.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(lf, height=14, wrap="none", font=("Consolas", 11), bg="#fbfaf7", relief="flat")
        lys = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.log.yview)
        lxs = ttk.Scrollbar(lf, orient=tk.HORIZONTAL, command=self.log.xview)
        self.log.configure(yscrollcommand=lys.set, xscrollcommand=lxs.set)
        lxs.pack(side=tk.BOTTOM, fill=tk.X)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lys.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.tag_configure("head", font=("Consolas", 11, "bold"), foreground=INK)

    # --- helpers
    def write_log(self, text, clear=True):
        self.log.configure(state="normal")
        if clear:
            self.log.delete("1.0", tk.END)
        for line in text.splitlines():
            tag = "head" if line.isupper() or line.startswith(("ITERATION", "INITIALISE", "✔")) else ""
            self.log.insert(tk.END, line + "\n", tag)
        self.log.see("1.0" if clear else tk.END)

    def compute_limits(self):
        pts = self.model.points
        if not pts:
            self.lims = (0, 10, 0, 10)
            return
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        lo = min(0, math.floor(min(xs + ys)) - 1)
        hi = max(10, math.ceil(max(xs + ys)) + 1)
        if self.dataset_var.get().startswith("Worked"):
            lo, hi = 0, 9
        self.lims = (lo, hi, lo, hi)

    # --- data
    def load_dataset(self):
        self.stop_run()
        m = self.model
        ds = self.dataset_var.get()
        m.centroids = []
        if ds.startswith("Worked"):
            m.points = [[n, x, y] for n, x, y in WORKED_EXAMPLE]
            m.k = 2
            self.k_var.set(2)
            m.centroids = [[5, 8], [6, 6]]      # start at B and E, as in the slides
            m.clear_run()
            msg = ("WORKED EXAMPLE — 6 points, K = 2\n"
                   "Starting centroids: μ1 = B (5, 8), μ2 = E (6, 6)\n\n"
                   "Press Step (or Space) to run the first ASSIGN step.\n")
        elif ds.startswith("Random"):
            m.k = self.k_var.get()
            m.points = []
            centers = []
            while len(centers) < m.k:
                c = (random.uniform(2, 8), random.uniform(2, 8))
                if all(math.hypot(c[0] - a, c[1] - b) > 3.0 for a, b in centers) or len(centers) > 40:
                    centers.append(c)
            for cx, cy in centers:
                for _ in range(7):
                    m.points.append([None, round(random.gauss(cx, 0.7), 1), round(random.gauss(cy, 0.7), 1)])
            random.shuffle(m.points)
            for i, p in enumerate(m.points):
                p[0] = point_name(i)
            m.clear_run()
            msg = self.safe_init()
        else:
            m.points = []
            m.k = self.k_var.get()
            m.clear_run()
            msg = ("EMPTY CANVAS\nLeft-click on the plot to add points, right-click to remove.\n"
                   "Then press ① Initialise.\n")
        self.selected = None
        self.compute_limits()
        self.write_log(msg)
        self.refresh()

    def safe_init(self):
        method = self.init_var.get()
        if method == "Manual (click)":
            self.model.centroids = []
            self.model.clear_run()
            return f"MANUAL START\nClick {self.model.k} places on the plot to put the centroids.\n"
        try:
            return self.model.init_centroids(method)
        except ValueError as e:
            self.model.centroids = []
            self.model.clear_run()
            return str(e) + "\n"

    def data_changed(self):
        """Points were added / removed / edited: restart the run, keep centroids."""
        self.stop_run()
        self.model.clear_run()
        self.refresh()

    # --- buttons
    def on_k_change(self):
        try:
            k = max(1, min(MAX_K, int(self.k_var.get())))
        except (tk.TclError, ValueError):
            return
        self.k_var.set(k)
        if k != self.model.k:
            self.model.k = k
            self.stop_run()
            self.write_log(self.safe_init())
            self.refresh()

    def on_init(self):
        self.stop_run()
        self.model.k = self.k_var.get()
        self.write_log(self.safe_init())
        self.refresh()

    def on_reset(self):
        """Back to the same starting centroids as the last initialisation."""
        self.stop_run()
        if self.dataset_var.get().startswith("Worked"):
            self.load_dataset()
            return
        m = self.model
        if m.J_hist and m.centroids:
            start = getattr(self, "_start_centroids", None)
            if start:
                m.centroids = [c[:] for c in start]
        m.clear_run()
        self.write_log("RESET\nSame points and starting centroids. Press Step.\n")
        self.refresh()

    def on_step(self):
        if self.busy:
            return
        m = self.model
        if m.phase == "no centroids":
            messagebox.showinfo("K-means", "Place the starting centroids first (① Initialise).")
            return
        if m.phase == "converged":
            self.stop_run()
            return
        if m.phase == "ready":
            self._start_centroids = [c[:] for c in m.centroids]
        text = m.step()
        self.write_log(text)
        if m.phase == "updated":
            self.animate(m.prev_centroids, m.centroids)
        else:
            self.refresh()

    def on_run(self):
        if self.running:
            self.stop_run()
            return
        if self.model.phase in ("no centroids", "converged"):
            return
        self.running = True
        self.run_btn.configure(text="Stop ■")
        self.run_loop()

    def run_loop(self):
        if not self.running:
            return
        if self.model.phase == "converged" or self.model.iteration > 100:
            self.stop_run()
            return
        if not self.busy:
            self.on_step()
        self.after(int(1100 / self.speed.get()), self.run_loop)

    def stop_run(self):
        self.running = False
        if hasattr(self, "run_btn"):
            self.run_btn.configure(text="Run ▶▶")

    # --- plot interaction
    def on_click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None or self.busy:
            return
        m = self.model
        x, y = round(ev.xdata, 1), round(ev.ydata, 1)
        span = self.lims[1] - self.lims[0]
        # manual centroid placement
        if self.init_var.get() == "Manual (click)" and len(m.centroids) < m.k and ev.button == 1:
            m.centroids.append([x, y])
            m.clear_run()
            left = m.k - len(m.centroids)
            self.write_log(f"MANUAL START\nμ{len(m.centroids)} placed at ({fmt(x)}, {fmt(y)}). "
                           + (f"Click {left} more." if left else "Press Step to begin.") + "\n")
            self.refresh()
            return
        near = None
        if m.points:
            d = [math.hypot(p[1] - x, p[2] - y) for p in m.points]
            i = min(range(len(d)), key=d.__getitem__)
            if d[i] < span * 0.03:
                near = i
        if ev.button == 1:
            if near is not None:
                self.select_point(near)
                return
            used = {p[0] for p in m.points}
            i = 0
            while point_name(i) in used:
                i += 1
            m.points.append([point_name(i), x, y])
            self.write_log(f"Added point {point_name(i)} ({fmt(x)}, {fmt(y)}). The run restarts.\n")
            self.data_changed()
        elif ev.button == 3 and m.points:
            d = [math.hypot(p[1] - x, p[2] - y) for p in m.points]
            i = min(range(len(d)), key=d.__getitem__)
            p = m.points.pop(i)
            self.selected = None
            self.write_log(f"Removed point {p[0]} ({fmt(p[1])}, {fmt(p[2])}). The run restarts.\n")
            self.data_changed()

    def on_row_select(self, _ev):
        sel = self.ptree.selection()
        if sel:
            i = int(sel[0])
            if i != self.selected:
                self.select_point(i, from_table=True)

    def select_point(self, i, from_table=False):
        m = self.model
        self.selected = i
        name, x, y = m.points[i]
        lines = [f"POINT {name}  (x = {fmt(x)}, y = {fmt(y)})", ""]
        if m.centroids and len(m.centroids) == m.k:
            D = [m.dist(m.points[i], c) for c in m.centroids]
            for j in range(m.k):
                lines.append("  " + m.distance_formula(i, j))
            best = min(range(m.k), key=D.__getitem__)
            lines.append(f"\n  Smallest distance: μ{best + 1}  →  {name} belongs to cluster {best + 1}")
        self.write_log("\n".join(lines) + "\n")
        if not from_table:
            self.ptree.selection_set(str(i))
            self.ptree.see(str(i))
        self.redraw()

    def on_row_edit(self, ev):
        row = self.ptree.identify_row(ev.y)
        if not row:
            return
        i = int(row)
        p = self.model.points[i]
        s = simpledialog.askstring("Edit point", f"New x, y for point {p[0]}:",
                                   initialvalue=f"{fmt(p[1])}, {fmt(p[2])}", parent=self)
        if not s:
            return
        try:
            x, y = (float(v) for v in s.replace(";", ",").split(","))
        except ValueError:
            messagebox.showerror("Edit point", "Please type two numbers, e.g.  3.5, 7")
            return
        p[1], p[2] = x, y
        self.compute_limits()
        self.write_log(f"Point {p[0]} moved to ({fmt(x)}, {fmt(y)}). The run restarts.\n")
        self.data_changed()

    # --- animation of the update step
    def animate(self, old, new, frame=0, frames=18):
        self.busy = True
        t = frame / frames
        e = 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
        cents = [[a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e] for a, b in zip(old, new)]
        self.redraw(cents)
        if frame < frames:
            self.after(max(10, int(30 / self.speed.get())), self.animate, old, new, frame + 1, frames)
        else:
            self.busy = False
            self.refresh()

    # --- drawing
    def refresh(self):
        self.update_tables()
        self.update_status()
        self.redraw()

    def update_status(self):
        m = self.model
        phase = {"no centroids": "No centroids yet", "ready": "Ready", "assigned": "Assigned",
                 "updated": "Updated", "converged": "✔ Converged"}[m.phase]
        s = f"Iteration {m.iteration}   ·   {phase}   ·   {len(m.points)} points, K = {m.k}"
        if m.changed:
            s += f"   ·   {len(m.changed)} changed"
        if m.J_hist:
            s += "   ·   J: " + " → ".join(fmt(j) for j in m.J_hist[-6:])
        self.status.set(s)
        self.next_lbl.set(f"Next step: {m.next_step_name()}")

    def update_tables(self):
        m = self.model
        k = m.k
        cols = ["pt", "x", "y"] + [f"d{j}" for j in range(k)] + ["cl", "sq"]
        self.ptree.configure(columns=cols, displaycolumns=cols)
        heads = {"pt": ("Point", 60), "x": ("x", 70), "y": ("y", 70), "cl": ("Cluster", 110), "sq": ("d² (own)", 90)}
        for j in range(k):
            c = m.centroids[j] if j < len(m.centroids) and k <= 2 else None
            heads[f"d{j}"] = (f"d to μ{j + 1}" + (f" ({fmt(c[0])}, {fmt(c[1])})" if c else ""), 150 if c else 100)
        for c in cols:
            self.ptree.heading(c, text=heads[c][0])
            self.ptree.column(c, width=heads[c][1], anchor="center", stretch=True)
        self.ptree.delete(*self.ptree.get_children())
        have_c = len(m.centroids) == k
        D = m.distance_matrix() if have_c else None
        for i, (name, x, y) in enumerate(m.points):
            vals = [name, fmt(x), fmt(y)]
            lab = m.labels[i] if i < len(m.labels) else None
            if have_c:
                for j in range(k):
                    mark = "  ✔" if lab is not None and j == lab else ""
                    vals.append(f"{D[i][j]:.2f}{mark}")
            else:
                vals += ["—"] * k
            if lab is not None:
                vals += [f"{lab + 1}" + ("  (changed)" if i in m.changed else ""), fmt(D[i][lab] ** 2)]
                tag = "changed" if i in m.changed else f"c{lab}"
            else:
                vals += ["—", "—"]
                tag = ""
            self.ptree.insert("", tk.END, iid=str(i), values=vals, tags=(tag,) if tag else ())
        if self.selected is not None and self.selected < len(m.points):
            self.ptree.selection_set(str(self.selected))

        self.ctree.delete(*self.ctree.get_children())
        for j, c in enumerate(m.centroids):
            members = [p[0] for p, l in zip(m.points, m.labels) if l == j]
            self.ctree.insert("", tk.END, values=(f"μ{j + 1}", fmt(c[0]), fmt(c[1]), len(members),
                                                  ", ".join(members) if members else "—"),
                              tags=(f"c{j}",))

    def redraw(self, cents=None):
        m = self.model
        ax = self.ax
        ax.clear()
        x0, x1, y0, y1 = self.lims
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        step = 1 if (x1 - x0) <= 12 else 2
        ax.set_xticks(range(int(x0), int(x1) + 1, step))
        ax.set_yticks(range(int(y0), int(y1) + 1, step))
        ax.grid(True, color="#e6e3dc", linewidth=1)
        ax.set_axisbelow(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        cents = cents if cents is not None else m.centroids
        n = len(m.points)
        small = n <= 30

        # distance lines (assign step)
        if self.show_lines.get() and m.phase in ("assigned", "converged") and cents:
            for p, l in zip(m.points, m.labels):
                if l is not None:
                    ax.plot([p[1], cents[l][0]], [p[2], cents[l][1]], linestyle="--",
                            color=PALETTE[l], alpha=0.45, linewidth=1.4, zorder=1)
        # ghosts of old centroids (update step)
        if m.prev_centroids and m.phase == "updated":
            for j, (a, b) in enumerate(zip(m.prev_centroids, m.centroids)):
                ax.plot(*a, marker="D", markersize=13, color=PALETTE[j], alpha=0.2,
                        markeredgecolor=PALETTE[j], zorder=2)
                ax.annotate("", xy=b, xytext=a, zorder=2,
                            arrowprops=dict(arrowstyle="->", color=PALETTE[j], linestyle=":", lw=1.8))
        # points
        for i, (name, x, y) in enumerate(m.points):
            l = m.labels[i] if i < len(m.labels) else None
            col = GREY if l is None else PALETTE[l]
            ax.scatter([x], [y], s=150 if small else 70, color=col, edgecolors="white", linewidths=1.5, zorder=3)
            if i in m.changed:
                ax.scatter([x], [y], s=520 if small else 260, facecolors="none", edgecolors="#e08a1e",
                           linewidths=2.2, zorder=3)
            if i == self.selected:
                ax.scatter([x], [y], s=700, facecolors="none", edgecolors=INK, linewidths=2.2, zorder=3)
            if small:
                full = self.show_coords.get() and (n <= 12 or i == self.selected)
                label = f"{name} ({fmt(x)}, {fmt(y)})" if full else name
                ax.annotate(label, (x, y), xytext=(8, 7), textcoords="offset points",
                            fontsize=10, color=INK, zorder=4)
        # selected point's distances to every centroid
        if self.selected is not None and self.selected < n and len(cents) == m.k:
            _, sx, sy = m.points[self.selected]
            for j, c in enumerate(cents):
                d = math.hypot(sx - c[0], sy - c[1])
                ax.plot([sx, c[0]], [sy, c[1]], color=PALETTE[j], linewidth=2, zorder=2)
                ax.annotate(f"{d:.2f}", ((sx + c[0]) / 2, (sy + c[1]) / 2), fontsize=10, fontweight="bold",
                            color=PALETTE[j], ha="center",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE[j], lw=1), zorder=5)
        # centroids
        for j, c in enumerate(cents):
            ax.plot(*c, marker="D", markersize=16, color=PALETTE[j], markeredgecolor=INK,
                    markeredgewidth=2, zorder=6)
            txt = f"μ{j + 1} ({fmt(c[0])}, {fmt(c[1])})" if self.show_coords.get() else f"μ{j + 1}"
            ax.annotate(txt, c, xytext=(10, -16), textcoords="offset points", fontsize=11,
                        fontweight="bold", color="white", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc=PALETTE[j], ec="none"))
        title = f"K-means  ·  K = {m.k}  ·  iteration {m.iteration}"
        if m.phase == "converged":
            title += "  ·  ✔ converged"
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    App().mainloop()
