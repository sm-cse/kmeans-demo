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

import math
import time

import pandas as pd
import streamlit as st

from kmeans_model import (
    DATASETS, INIT_METHODS, MAX_K, WORKED_EXAMPLE, WORKED_START,
    KMeansModel, fmt, make_blobs, next_free_name, plot_limits,
)
from plot_component import kmeans_plot

PALETTE = ["#2b5cb8", "#b8336a", "#e08a1e", "#2f9e7a", "#7b4fc9", "#8a6d3b"]
LIGHT = ["#dbe5f6", "#f5dbe6", "#fde9cf", "#d7efe6", "#e7ddf7", "#eee5d7"]
CHANGED = "#ffd28f"

st.set_page_config(page_title="K-Means Teaching Demo", page_icon="🎯", layout="wide")
ss = st.session_state


# --------------------------------------------------------------------------- state & actions
def is_worked():
    return ss.dataset == DATASETS[0]


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
        msg = ("EXAMPLE — 6 points, K = 2\n"
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


def on_plot_clicks():
    """Handle clicks sent by the browser plot (queued, each with an increasing id)."""
    clicks = ss.plot.clicks or []
    for c in sorted(clicks, key=lambda c: c["id"]):
        if c["id"] <= ss.last_click_id:
            continue            # already handled
        ss.last_click_id = c["id"]
        handle_click(float(c["x"]), float(c["y"]))


def handle_click(x, y):
    m = ss.model
    ss.running = False
    x0, x1 = ss.lims[0], ss.lims[1]

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
        ss.anim_id += 1
        ss.animate_from = [c[:] for c in m.prev_centroids]
    else:
        ss.animate_from = None


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
    ss.last_click_id = 0
    ss.anim_id = 0
    load_dataset()


# --------------------------------------------------------------------------- tables
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
    # Using HTML for background color and bold text
    st.markdown(
        Prepared by '<span style="background-color: #FFFF00; font-weight: bold; padding: 2px 5px; border-radius: 3px;">Shukla Mondal</span>', 
        unsafe_allow_code=True
    )
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
    title = f"K-means  ·  K = {m.k}  ·  iteration {m.iteration}"
    if m.phase == "converged":
        title += "  ·  ✔ converged"
    kmeans_plot(
        data={
            "lims": list(ss.lims),
            "k": m.k,
            "points": [[p[0], p[1], p[2],
                        m.labels[i] if i < len(m.labels) and m.labels[i] is not None else -1,
                        i in m.changed] for i, p in enumerate(m.points)],
            "centroids": m.centroids,
            "prev": m.prev_centroids,
            "phase": m.phase,
            "selected": selected,
            "show_coords": ss.show_coords,
            "show_lines": ss.show_lines,
            "title": title,
            "palette": PALETTE,
            "mode": "remove" if ss.click_mode.startswith("Remove") else "add",
            "manual_left": max(0, m.k - len(m.centroids)) if is_manual() else 0,
            "anim": {"id": ss.anim_id, "from": ss.animate_from}
                    if ss.animate_from and m.phase == "updated" else None,
            "speed": ss.speed,
            "ack": ss.last_click_id,
        },
        key="plot", on_clicks=on_plot_clicks)

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
