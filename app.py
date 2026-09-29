"""
K-Means Clustering - Interactive Teaching Demo (Streamlit version)
==================================================================

Runs K-means one step at a time and shows every number:
  * the plot shows each point's (x, y) and each centroid's position
  * the Points table shows the distance of every point to every centroid,
    the chosen cluster (ticked) and the squared distance
  * the Centroids table shows each centroid's coordinates and members
  * the Calculations panel writes out the full formulas for every step
  * click the plot to add, select or remove points, or to place centroids by hand

Run locally:   streamlit run app.py
"""

import io
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from kmeans_model import (
    DATASETS, INIT_METHODS, MAX_K, WORKED_EXAMPLE, WORKED_START,
    KMeansModel, fmt, make_blobs, next_free_name, plot_limits,
)

PALETTE = ["#2b5cb8", "#b8336a", "#e08a1e", "#2f9e7a", "#7b4fc9", "#8a6d3b"]
LIGHT = ["#dbe5f6", "#f5dbe6", "#fde9cf", "#d7efe6", "#e7ddf7", "#eee5d7"]
CHANGED = "#ffd28f"
GREY = "#b9bdc5"
INK = "#1c2230"

st.set_page_config(page_title="K-Means Teaching Demo", page_icon="🎯", layout="wide")
ss = st.session_state


# --------------------------------------------------------------------------- state & actions
def is_worked():
    return ss.dataset.startswith("Worked")


def is_manual():
    return ss.init_method.startswith("Manual")


def remember_start():
    ss.start_centroids = [c[:] for c in ss.model.centroids]


def safe_init():
    m = ss.model
    if is_manual():
        m.centroids = []
        m.clear_run()
        return (f"MANUAL START\nClick {m.k} places on the plot to put the centroids "
                "(or type exact positions in the sidebar).\n")
    try:
        msg = m.init_centroids(ss.init_method)
    except ValueError as e:
        m.centroids = []
        m.clear_run()
        msg = str(e) + "\n"
    remember_start()
    return msg


def load_dataset():
    m = ss.model
    ss.running = False
    ss.animate_from = None
    m.centroids = []
    if is_worked():
        m.points = [[n, x, y] for n, x, y in WORKED_EXAMPLE]
        m.k = ss.k = 2
        m.centroids = [c[:] for c in WORKED_START]
        m.clear_run()
        remember_start()
        msg = ("WORKED EXAMPLE — 6 points, K = 2\n"
               "Starting centroids: μ1 = B (5, 8), μ2 = E (6, 6)\n\n"
               "Press Step (or Space) to run the first ASSIGN step.\n")
    elif ss.dataset.startswith("Random"):
        m.k = ss.k
        m.points = make_blobs(m.k)
        m.clear_run()
        msg = safe_init()
    else:
        m.points = []
        m.k = ss.k
        m.clear_run()
        msg = ("EMPTY CANVAS\nClick on the plot to add points (or use the Edit points table), "
               "then press Initialise.\n")
    ss.lims = plot_limits(m.points, is_worked())
    ss.log = msg
    ss.selected = None
    ss.version += 1


def points_changed(msg):
    """Points were added / removed / edited: restart the run, keep centroids."""
    ss.model.clear_run()
    ss.running = False
    ss.animate_from = None
    ss.selected = None
    ss.log = msg
    ss.version += 1


def select_point(i):
    ss.selected = i
    ss.table_ver += 1          # re-key the table so the row shows as selected


def on_plot_click():
    """Turn a click on the plot image into data coordinates and act on it."""
    v, g = ss.get("plot_click"), ss.get("plot_geom")
    if not v or not g or v.get("unix_time") == ss.last_click:
        return
    ss.last_click = v.get("unix_time")
    W, H, (bx0, by0, bx1, by1), (x0, x1, y0, y1) = g
    fx = v["x"] * W / v["width"]
    fy = H - v["y"] * H / v["height"]
    if not (bx0 <= fx <= bx1 and by0 <= fy <= by1):
        return                  # clicked outside the axes
    x = round(x0 + (fx - bx0) / (bx1 - bx0) * (x1 - x0), 1)
    y = round(y0 + (fy - by0) / (by1 - by0) * (y1 - y0), 1)
    m = ss.model
    ss.running = False

    # manual centroid placement
    if is_manual() and len(m.centroids) < m.k:
        m.centroids.append([x, y])
        m.clear_run()
        left = m.k - len(m.centroids)
        if not left:
            remember_start()
        ss.log = (f"MANUAL START\nμ{len(m.centroids)} placed at ({fmt(x)}, {fmt(y)}). "
                  + (f"Click {left} more." if left else "Press Step to begin.") + "\n")
        return

    near = None
    if m.points:
        d = [math.hypot(p[1] - x, p[2] - y) for p in m.points]
        i = min(range(len(d)), key=d.__getitem__)
        if d[i] < (x1 - x0) * 0.03:
            near = i

    if ss.click_mode.startswith("Remove"):
        if near is None:
            ss.flash = "No point there. Click right on a point to remove it."
            return
        p = m.points.pop(near)
        points_changed(f"Removed point {p[0]} ({fmt(p[1])}, {fmt(p[2])}). The run restarts.\n")
    elif near is not None:
        select_point(near)
    else:
        name = next_free_name(m.points)
        m.points.append([name, x, y])
        points_changed(f"Added point {name} ({fmt(x)}, {fmt(y)}). The run restarts.\n")


def on_k_change():
    ss.model.k = ss.k
    ss.running = False
    ss.log = safe_init()


def on_init():
    ss.running = False
    ss.model.k = ss.k
    ss.log = safe_init()


def on_place_manual():
    m = ss.model
    m.k = ss.k
    cents = [(ss[f"mx{j}"], ss[f"my{j}"]) for j in range(m.k)]
    ss.log = m.set_centroids(cents, "Manual")
    remember_start()
    ss.running = False


def on_reset():
    ss.running = False
    ss.animate_from = None
    if is_worked():
        load_dataset()
        return
    m = ss.model
    if ss.start_centroids and len(ss.start_centroids) == m.k:
        m.centroids = [c[:] for c in ss.start_centroids]
    m.clear_run()
    ss.log = "RESET\nSame points and starting centroids. Press Step.\n"


def do_step():
    m = ss.model
    if m.phase == "no centroids":
        ss.flash = "Place the starting centroids first (Initialise)."
        ss.running = False
        return
    if m.phase == "converged":
        ss.running = False
        return
    if m.phase == "ready":
        remember_start()
    ss.log = m.step()
    if m.phase == "updated" and ss.animate:
        ss.animate_from = [c[:] for c in m.prev_centroids]


def on_run():
    if ss.running:
        ss.running = False
    elif ss.model.phase not in ("no centroids", "converged"):
        ss.running = True
    else:
        ss.flash = ("Place the starting centroids first (Initialise)." if ss.model.phase == "no centroids"
                    else "Already converged. Press Reset to run again.")


def init_state():
    if "model" in ss:
        return
    ss.model = KMeansModel()
    ss.dataset = DATASETS[0]
    ss.k = 2
    ss.init_method = INIT_METHODS[0]
    ss.show_coords = True
    ss.show_lines = True
    ss.animate = True
    ss.speed = 1.0
    ss.version = 0
    ss.running = False
    ss.animate_from = None
    ss.start_centroids = None
    ss.flash = None
    ss.click_mode = "Add or select a point"
    ss.selected = None
    ss.table_ver = 0
    ss.last_click = None
    ss.plot_geom = None
    load_dataset()


# --------------------------------------------------------------------------- drawing
def draw(m, cents, selected, lims, show_coords, show_lines):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    x0, x1, y0, y1 = lims
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
    n = len(m.points)
    small = n <= 30

    # distance lines (assign step)
    if show_lines and m.phase in ("assigned", "converged") and cents:
        for p, lab in zip(m.points, m.labels):
            if lab is not None:
                ax.plot([p[1], cents[lab][0]], [p[2], cents[lab][1]], linestyle="--",
                        color=PALETTE[lab], alpha=0.45, linewidth=1.4, zorder=1)
    # ghosts of old centroids (update step)
    if m.prev_centroids and m.phase == "updated":
        for j, (a, b) in enumerate(zip(m.prev_centroids, m.centroids)):
            ax.plot(*a, marker="D", markersize=13, color=PALETTE[j], alpha=0.2,
                    markeredgecolor=PALETTE[j], zorder=2)
            ax.annotate("", xy=b, xytext=a, zorder=2,
                        arrowprops=dict(arrowstyle="->", color=PALETTE[j], linestyle=":", lw=1.8))
    # points
    for i, (name, x, y) in enumerate(m.points):
        lab = m.labels[i] if i < len(m.labels) else None
        col = GREY if lab is None else PALETTE[lab]
        ax.scatter([x], [y], s=150 if small else 70, color=col, edgecolors="white", linewidths=1.5, zorder=3)
        if i in m.changed:
            ax.scatter([x], [y], s=520 if small else 260, facecolors="none", edgecolors="#e08a1e",
                       linewidths=2.2, zorder=3)
        if i == selected:
            ax.scatter([x], [y], s=700, facecolors="none", edgecolors=INK, linewidths=2.2, zorder=3)
        if small or i == selected:
            full = show_coords and (n <= 12 or i == selected)
            label = f"{name} ({fmt(x)}, {fmt(y)})" if full else name
            ax.annotate(label, (x, y), xytext=(8, 7), textcoords="offset points",
                        fontsize=10, color=INK, zorder=4)
    # selected point's distances to every centroid
    if selected is not None and selected < n and cents and len(cents) == m.k:
        _, sx, sy = m.points[selected]
        for j, c in enumerate(cents):
            d = ((sx - c[0]) ** 2 + (sy - c[1]) ** 2) ** 0.5
            ax.plot([sx, c[0]], [sy, c[1]], color=PALETTE[j], linewidth=2, zorder=2)
            ax.annotate(f"{d:.2f}", ((sx + c[0]) / 2, (sy + c[1]) / 2), fontsize=10, fontweight="bold",
                        color=PALETTE[j], ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE[j], lw=1), zorder=5)
    # centroids
    for j, c in enumerate(cents):
        ax.plot(*c, marker="D", markersize=16, color=PALETTE[j], markeredgecolor=INK,
                markeredgewidth=2, zorder=6)
        txt = f"μ{j + 1} ({fmt(c[0])}, {fmt(c[1])})" if show_coords else f"μ{j + 1}"
        ax.annotate(txt, c, xytext=(10, -16), textcoords="offset points", fontsize=11,
                    fontweight="bold", color="white", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc=PALETTE[j], ec="none"))
    title = f"K-means  ·  K = {m.k}  ·  iteration {m.iteration}"
    if m.phase == "converged":
        title += "  ·  ✔ converged"
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def fig_to_image(fig):
    """PNG of the figure, plus where the axes sit in it (for mapping clicks)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)   # no bbox_inches: keeps pixel geometry exact
    ax = fig.axes[0]
    bb = ax.get_window_extent()
    geom = (fig.bbox.width, fig.bbox.height, (bb.x0, bb.y0, bb.x1, bb.y1),
            (*ax.get_xlim(), *ax.get_ylim()))
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf), geom


def points_table(m):
    k = m.k
    have_c = len(m.centroids) == k
    D = m.distance_matrix() if have_c else None
    dcols = []
    for j in range(k):
        c = m.centroids[j] if have_c and k <= 3 else None
        dcols.append(f"d to μ{j + 1}" + (f" ({fmt(c[0])}, {fmt(c[1])})" if c else ""))
    rows, tags = [], []
    for i, (name, x, y) in enumerate(m.points):
        lab = m.labels[i] if i < len(m.labels) else None
        row = {"Point": name, "x": fmt(x), "y": fmt(y)}
        for j in range(k):
            if have_c:
                row[dcols[j]] = f"{D[i][j]:.2f}" + ("  ✔" if lab == j else "")
            else:
                row[dcols[j]] = "—"
        if lab is not None:
            row["Cluster"] = f"{lab + 1}" + ("  (changed)" if i in m.changed else "")
            row["d² (own)"] = fmt(D[i][lab] ** 2)
            tags.append(CHANGED if i in m.changed else LIGHT[lab])
        else:
            row["Cluster"], row["d² (own)"] = "—", "—"
            tags.append(None)
        rows.append(row)
    df = pd.DataFrame(rows, columns=["Point", "x", "y"] + dcols + ["Cluster", "d² (own)"])
    return df.style.apply(
        lambda r: [f"background-color: {tags[r.name]}" if tags[r.name] else ""] * len(r), axis=1)


def centroids_table(m):
    rows, tags = [], []
    for j, c in enumerate(m.centroids):
        members = [p[0] for p, lab in zip(m.points, m.labels) if lab == j]
        rows.append({"Centroid": f"μ{j + 1}", "x": fmt(c[0]), "y": fmt(c[1]),
                     "Count": len(members), "Members": ", ".join(members) if members else "—"})
        tags.append(LIGHT[j])
    df = pd.DataFrame(rows, columns=["Centroid", "x", "y", "Count", "Members"])
    return df.style.apply(lambda r: [f"background-color: {tags[r.name]}"] * len(r), axis=1)


# --------------------------------------------------------------------------- page
init_state()
m = ss.model

if ss.flash:
    st.toast(ss.flash)
    ss.flash = None

# sidebar controls
with st.sidebar:
    st.header("Setup")
    st.selectbox("Data", DATASETS, key="dataset", on_change=load_dataset)
    st.button("New data", on_click=load_dataset, width="stretch",
              help="Reload the chosen dataset (new random blobs for Random blobs).")
    st.number_input("K (number of clusters)", min_value=1, max_value=MAX_K, step=1,
                    key="k", on_change=on_k_change)
    st.selectbox("Starting centroids", INIT_METHODS, key="init_method", on_change=on_init)
    if is_manual():
        placed = len(m.centroids) if len(m.centroids) < m.k else m.k
        st.caption(f"Click the plot to place the centroids: {placed} of {m.k} placed.")
        st.button("Clear centroids and click again", on_click=on_init, width="stretch")
        lo, hi = ss.lims[0], ss.lims[1]
        with st.expander("Or type exact positions"), st.form("manual_form", border=False):
            for j in range(ss.k):
                default = m.centroids[j] if len(m.centroids) == ss.k else \
                    [round(lo + (j + 1) * (hi - lo) / (ss.k + 1), 1)] * 2
                c1, c2 = st.columns(2)
                c1.number_input(f"μ{j + 1} x", value=float(default[0]), step=0.5, key=f"mx{j}")
                c2.number_input(f"μ{j + 1} y", value=float(default[1]), step=0.5, key=f"my{j}")
            st.form_submit_button("Place centroids", on_click=on_place_manual, type="primary")
    else:
        st.button("Initialise", on_click=on_init, type="primary", width="stretch",
                  help="Place new starting centroids with the chosen method.")

    st.header("Clicking the plot")
    st.radio("Clicking the plot", ["Add or select a point", "Remove a point"],
             key="click_mode", label_visibility="collapsed")

    st.header("Display")
    st.toggle("Show (x, y) labels", key="show_coords")
    st.toggle("Show distance lines", key="show_lines")
    st.toggle("Animate centroid moves", key="animate")
    st.slider("Animation speed", 0.3, 3.0, step=0.1, key="speed")

    st.divider()
    st.caption("Tip: press Space to step. Click empty space on the plot to add a point; "
               "click a point (or its row in the Points table) to see its distance calculations.")

# header and main controls
st.title("K-means clustering, one step at a time")

b1, b2, b3, status_col = st.columns([1, 1, 1, 5], vertical_alignment="center")
b1.button("Step ▶", on_click=do_step, type="primary", width="stretch", shortcut="Space",
          disabled=m.phase == "converged")
b2.button("Stop ■" if ss.running else "Run ▶▶", on_click=on_run, width="stretch")
b3.button("Reset", on_click=on_reset, width="stretch")

phase = {"no centroids": "No centroids yet", "ready": "Ready", "assigned": "Assigned",
         "updated": "Updated", "converged": "✔ Converged"}[m.phase]
status = f"**Iteration {m.iteration}** · {phase} · {len(m.points)} points, K = {m.k}"
if m.changed:
    status += f" · {len(m.changed)} changed"
if m.J_hist:
    status += " · J: " + " → ".join(fmt(j) for j in m.J_hist[-6:])
status_col.markdown(status + f"  \n:orange[Next step: **{m.next_step_name()}**]")

left, right = st.columns([5, 6], gap="large")

# right column first, so the selected table row is known before drawing the plot
with right:
    st.subheader("Points")
    if ss.selected is not None and ss.selected >= len(m.points):
        ss.selected = None
    default = {"selection": {"rows": [ss.selected]}} if ss.selected is not None else None
    event = st.dataframe(points_table(m), hide_index=True, width="stretch",
                         height=min(38 + 35 * max(len(m.points), 1), 400),
                         on_select="rerun", selection_mode="single-row", selection_default=default,
                         key=f"ptable_{ss.version}_{ss.table_ver}")
    rows = event.selection.rows if event else []
    ss.selected = rows[0] if rows and rows[0] < len(m.points) else None
    selected = ss.selected

    st.subheader("Centroids")
    if m.centroids:
        st.dataframe(centroids_table(m), hide_index=True, width="stretch")
    else:
        st.caption("No centroids yet.")

    st.subheader("Calculations")
    st.code(ss.log, language=None, height=360)
    if selected is not None:
        st.code(m.explain_point(selected), language=None)

with left:
    plot_slot = st.empty()
    if ss.animate_from is not None and len(ss.animate_from) == len(m.centroids):
        old, new = ss.animate_from, m.centroids
        ss.animate_from = None
        frames = 12
        for f in range(frames):
            t = f / frames
            e = 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
            cents = [[a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e] for a, b in zip(old, new)]
            img, _ = fig_to_image(draw(m, cents, selected, ss.lims, ss.show_coords, ss.show_lines))
            plot_slot.image(img, width="stretch")
            time.sleep(0.03 / ss.speed)
    ss.animate_from = None
    img, ss.plot_geom = fig_to_image(draw(m, m.centroids, selected, ss.lims,
                                          ss.show_coords, ss.show_lines))
    with plot_slot.container():
        streamlit_image_coordinates(img, width="stretch", key="plot_click",
                                    on_click=on_plot_click, cursor="crosshair",
                                    png_compression_level=1)

    if len(m.J_hist) > 1:
        st.caption("Objective J after each Assign step (it never goes up)")
        st.line_chart(pd.DataFrame({"J": m.J_hist}, index=range(1, len(m.J_hist) + 1)), height=160)

    with st.expander("Edit points", expanded=not m.points):
        st.caption("Type exact values: change x or y, add rows at the bottom, or select rows and delete them. "
                   "Any edit restarts the run and keeps the current centroids.")
        base = pd.DataFrame([[p[0], float(p[1]), float(p[2])] for p in m.points],
                            columns=["Point", "x", "y"])
        edited = st.data_editor(
            base, num_rows="dynamic", hide_index=True, width="stretch", key=f"editor_{ss.version}",
            column_config={"Point": st.column_config.TextColumn("Point", help="Leave blank for an automatic name"),
                           "x": st.column_config.NumberColumn("x", step=0.1, format="%.2f"),
                           "y": st.column_config.NumberColumn("y", step=0.1, format="%.2f")})
        if edited[["x", "y"]].isna().any().any():
            st.caption(":orange[Fill in both x and y for every row to apply the change.]")
        else:
            new_pts = []
            for name, x, y in edited.itertuples(index=False):
                name = str(name).strip() if isinstance(name, str) and name.strip() else None
                new_pts.append([name, float(x), float(y)])
            for p in new_pts:
                if p[0] is None:
                    p[0] = next_free_name(new_pts)
            old_pts = [[p[0], float(p[1]), float(p[2])] for p in m.points]
            if new_pts != old_pts:
                m.points = new_pts
                in_box = all(0 <= v <= 9 for p in new_pts for v in p[1:])
                ss.lims = plot_limits(new_pts, is_worked() and in_box)
                points_changed("POINTS EDITED\nThe run restarts with the same centroids. Press Step.\n")
                st.rerun()

# auto-run: one step per rerun, so every step is drawn
if ss.running:
    if m.phase == "converged" or m.iteration > 100:
        ss.running = False
    else:
        time.sleep(1.1 / ss.speed)
        do_step()
    st.rerun()
