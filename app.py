import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Set up page styling and config
st.set_page_config(
    page_title="K-Means Clustering - Interactive Teaching Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette configurations matching the original look
PALETTE = ["#2b5cb8", "#b8336a", "#e08a1e", "#2f9e7a", "#7b4fc9", "#8a6d3b"]
LIGHT = ["#dbe5f6", "#f5dbe6", "#fde9cf", "#d7efe6", "#e7ddf7", "#eee5d7"]
GREY = "#b9bdc5"
INK = "#1c2230"

WORKED_EXAMPLE = [("A", 3.0, 3.0), ("B", 5.0, 8.0), ("C", 4.0, 1.0), ("D", 4.0, 7.0), ("E", 6.0, 6.0), ("F", 5.0, 2.0)]

def fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s

def point_name(i):
    letters = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        letters = chr(65 + r) + letters
    return letters

class KMeansModel:
    def __init__(self):
        self.points = []          # list of [name, x, y]
        self.k = 2
        self.centroids = []
        self.labels = []
        self.changed = set()
        self.iteration = 0
        self.phase = "no centroids"
        self.J_hist = []
        self.prev_centroids = None

    def clear_run(self):
        self.labels = [None] * len(self.points)
        self.changed = set()
        self.iteration = 0
        self.phase = "ready" if self.centroids else "no centroids"
        self.J_hist = []
        self.prev_centroids = None

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

    def init_centroids(self, method):
        pts = [(p[1], p[2]) for p in self.points]
        if len(pts) < self.k:
            return f"Error: Need at least K = {self.k} points."
        
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
        
        self.centroids = [list(c) for c in cs]
        self.clear_run()
        self.phase = "ready"
        
        names = []
        for c in self.centroids:
            match = [p[0] for p in self.points if (p[1], p[2]) == tuple(c)]
            names.append(f"{match[0]} " if match else "")
            
        lines = ["INITIALISE  (" + method + ")"]
        for j, c in enumerate(self.centroids):
            lines.append(f"  μ{j + 1} = {names[j]}({fmt(c[0])}, {fmt(c[1])})")
        lines.append("\nPress 'Step' to run the first ASSIGN step.")
        return "\n".join(lines)

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
        return "\n".join(lines)

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
        return "\n".join(lines)

    def step(self):
        if self.phase in ("ready", "updated"):
            return self.assign()
        if self.phase == "assigned":
            return self.update()
        return ""

# Initialize session state objects to hold variables across steps
if 'model' not in st.session_state:
    st.session_state.model = KMeansModel()
if 'log_text' not in st.session_state:
    st.session_state.log_text = "Select dataset and initialization variables, then press Initialise."
if 'dataset_prev' not in st.session_state:
    st.session_state.dataset_prev = None
if 'k_prev' not in st.session_state:
    st.session_state.k_prev = None

model = st.session_state.model

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Controls")

dataset_var = st.sidebar.selectbox(
    "Data Source",
    ["Worked example (6 points)", "Random blobs", "Custom inputs (Table Below)"]
)

k_var = st.sidebar.number_input("Number of Clusters (K)", min_value=1, max_value=6, value=2)
init_method = st.sidebar.selectbox("Initialisation Method", ["First K points", "Random points", "k-means++"])

# Check for settings adjustments to trigger reset
if dataset_var != st.session_state.dataset_prev or k_var != st.session_state.k_prev:
    st.session_state.dataset_prev = dataset_var
    st.session_state.k_prev = k_var
    model.k = k_var
    model.centroids = []
    
    if dataset_var.startswith("Worked"):
        model.points = [[n, x, y] for n, x, y in WORKED_EXAMPLE]
        model.k = 2
        model.centroids = [[5.0, 8.0], [6.0, 6.0]]
        model.clear_run()
        st.session_state.log_text = "WORKED EXAMPLE — 6 points, K = 2\nStarting centroids: μ1 = B (5, 8), μ2 = E (6, 6)\n\nPress Step to run the first ASSIGN step."
    elif dataset_var.startswith("Random"):
        model.points = []
        centers = []
        while len(centers) < model.k:
            c = (random.uniform(2, 8), random.uniform(2, 8))
            if all(math.hypot(c[0] - a, c[1] - b) > 3.0 for a, b in centers) or len(centers) > 40:
                centers.append(c)
        for cx, cy in centers:
            for _ in range(7):
                model.points.append([None, round(random.gauss(cx, 0.7), 1), round(random.gauss(cy, 0.7), 1)])
        random.shuffle(model.points)
        for i, p in enumerate(model.points):
            p[0] = point_name(i)
        model.clear_run()
        st.session_state.log_text = model.init_centroids(init_method)
    else:
        # Default starting points for manual entry
        model.points = [[point_name(i), float(x), float(y)] for i, (x, y) in enumerate([(2,3), (3,8), (7,2), (8,6), (6,7)])]
        model.clear_run()
        st.session_state.log_text = "CUSTOM INPUTS\nModify the items in the Data Points table under the plot, then click 'Initialise'."

# Buttons row
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("① Initialise", use_container_width=True):
    if dataset_var.startswith("Custom"):
        pass # points parsed below from dataframe changes
    st.session_state.log_text = model.init_centroids(init_method)

step_btn_active = model.phase != "converged" and model.phase != "no centroids"
if c2.button("Step ▶", disabled=not step_btn_active, use_container_width=True):
    st.session_state.log_text = model.step()

if c3.button("Reset 🔄", use_container_width=True):
    if dataset_var.startswith("Worked"):
        model.points = [[n, x, y] for n, x, y in WORKED_EXAMPLE]
        model.centroids = [[5.0, 8.0], [6.0, 6.0]]
    model.clear_run()
    st.session_state.log_text = "RESET\nSame variables and initial centroids. Click 'Step'."

st.sidebar.markdown("---")
show_coords = st.sidebar.checkbox("Show (x, y) labels", value=True)
show_lines = st.sidebar.checkbox("Show distance lines", value=True)

# Status bar metrics
st.markdown(f"### 📈 Iteration: {model.iteration} | Phase: `{model.phase.upper()}` | Points: {len(model.points)} | Clusters (K): {model.k}")

# Main window partitions
col_plot, col_calc = st.columns([5, 4])

# Plotting engine using Matplotlib
with col_plot:
    st.subheader("📍 Cluster Space Canvas")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Compute display domain ranges
    if model.points:
        xs = [p[1] for p in model.points]
        ys = [p[2] for p in model.points]
        lo = min(0.0, math.floor(min(xs + ys)) - 1)
        hi = max(10.0, math.ceil(max(xs + ys)) + 1)
        if dataset_var.startswith("Worked"):
            lo, hi = 0.0, 9.0
    else:
        lo, hi = 0.0, 10.0
        
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, color="#e6e3dc", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Render assignment visual tracks
    if show_lines and model.phase in ("assigned", "converged") and model.centroids:
        for p, l in zip(model.points, model.labels):
            if l is not None:
                ax.plot([p[1], model.centroids[l][0]], [p[2], model.centroids[l][1]], linestyle="--",
                        color=PALETTE[l], alpha=0.45, linewidth=1.4, zorder=1)

    # Render points
    for i, (name, x, y) in enumerate(model.points):
        l = model.labels[i] if i < len(model.labels) else None
        col = GREY if l is None else PALETTE[l]
        ax.scatter([x], [y], s=120, color=col, edgecolors="white", linewidths=1.5, zorder=3)
        
        if i in model.changed:
            ax.scatter([x], [y], s=240, facecolors="none", edgecolors="#e08a1e", linewidths=2.2, zorder=3)
            
        if show_coords or len(model.points) <= 12:
            label = f"{name} ({fmt(x)}, {fmt(y)})" if len(model.points) <= 12 else name
            ax.annotate(label, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9, color=INK, zorder=4)

    # Render centroids
    for j, c in enumerate(model.centroids):
        ax.plot(*c, marker="D", markersize=12, color=PALETTE[j], markeredgecolor=INK, markeredgewidth=1.5, zorder=5)
        txt = f"μ{j + 1} ({fmt(c[0])}, {fmt(c[1])})" if show_coords else f"μ{j + 1}"
        ax.annotate(txt, c, xytext=(8, -12), textcoords="offset points", fontsize=10,
                    fontweight="bold", color="white", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=PALETTE[j], ec="none"))

    st.pyplot(fig)

with col_calc:
    st.subheader("🔢 Mathematical Formulas & Step Executions")
    st.text_area("Calculations Log Output", value=st.session_state.log_text, height=480, disabled=True)
    
    if model.J_hist:
        st.markdown("**Cost Function Value History (J):**")
        st.code(" → ".join(fmt(j) for j in model.J_hist))

# Bottom grids showing structures
st.markdown("---")
st.subheader("📋 Data Structural Frameworks")

t_col1, t_col2 = st.columns([5, 4])

with t_col1:
    st.markdown("**Points Matrix**")
    pt_data = []
    have_c = len(model.centroids) == model.k
    D = model.distance_matrix() if have_c else None
    
    for i, (name, x, y) in enumerate(model.points):
        row = {"Point": name, "x": x, "y": y}
        lab = model.labels[i] if i < len(model.labels) else None
        if have_c:
            for j in range(model.k):
                row[f"d to μ{j+1}"] = f"{D[i][j]:.2f}" + (" ✔" if lab == j else "")
        else:
            for j in range(model.k):
                row[f"d to μ{j+1}"] = "—"
        row["Assigned Cluster"] = f"{lab + 1}" if lab is not None else "—"
        row["d² (own)"] = fmt(D[i][lab] ** 2) if lab is not None else "—"
        pt_data.append(row)
        
    df_pts = pd.DataFrame(pt_data)
    
    if dataset_var.startswith("Custom"):
        st.markdown("*Custom mode enabled: You can modify coordinate values directly inside the dataframe matrix grid below to live update structural plots.*")
        edited_df = st.data_editor(df_pts, disabled=["Point", "Assigned Cluster", "d² (own)"] + [f"d to μ{j+1}" for j in range(model.k)])
        # Sync changes back to points list
        updated_points = []
        for idx, r in edited_df.iterrows():
            updated_points.append([r["Point"], float(r["x"]), float(r["y"])])
        if updated_points != model.points:
            model.points = updated_points
            model.clear_run()
            st.rerun()
    else:
        st.dataframe(df_pts, hide_index=True, use_container_width=True)

with t_col2:
    st.markdown("**Centroids Summary**")
    ct_data = []
    for j, c in enumerate(model.centroids):
        members = [p[0] for p, l in zip(model.points, model.labels) if l == j]
        ct_data.append({
            "Centroid": f"μ{j + 1}",
            "x": fmt(c[0]),
            "y": fmt(c[1]),
            "Count": len(members),
            "Members": ", ".join(members) if members else "—"
        })
    if ct_data:
        st.dataframe(pd.DataFrame(ct_data), hide_index=True, use_container_width=True)
    else:
        st.info("Centroids are uninitialised.")
