"""
Interactive K-means plot drawn in the browser (SVG), using Streamlit's
built-in custom components (st.components.v2).

Why in the browser: a click shows up on the plot immediately, the plot never
reloads, and the centroid animation runs smoothly on the student's screen.
Python still does all the K-means work; this file only draws what it is given
and reports clicks back.

Clicks are queued with increasing ids and re-sent until Python confirms it has
processed them (the `ack` value), so fast clicking never loses a point.
"""

import streamlit as st

CSS = """
.km-wrap { position: relative; width: 100%; }
.km-wrap svg { width: 100%; height: auto; display: block; cursor: crosshair;
               user-select: none; -webkit-user-select: none; touch-action: manipulation;
               font-family: "Source Sans Pro", "Source Sans 3", system-ui, sans-serif; }
.km-readout { position: absolute; top: 6px; right: 10px; font-size: 13px; color: #5b6475;
              font-variant-numeric: tabular-nums; pointer-events: none; }
"""

JS = r"""
const NS = "http://www.w3.org/2000/svg";
const W = 600, H = 600, L = 48, R = 14, T = 34, B = 28;   // plot area is 538 x 538
const PW = W - L - R, PH = H - T - B;
const INK = "#1c2230", GREY = "#b9bdc5", GRID = "#e6e3dc", CHANGED = "#e08a1e";

function el(parent, tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e);
  return e;
}
function fmt(v) {
  let s = (Math.round(v * 100) / 100).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return (s === "-0" || s === "") ? "0" : s;
}
function diamond(g, cx, cy, r, attrs) {
  return el(g, "polygon", Object.assign(
    {points: `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`}, attrs));
}
function labelBox(g, x, y, text, fill, textFill, opts = {}) {
  const t = el(g, "text", {x, y, "font-size": opts.size || 13, "font-weight": opts.weight || 700,
                           fill: textFill, "text-anchor": opts.anchor || "start",
                           "dominant-baseline": "middle"}, text);
  const bb = t.getBBox();
  const rect = el(g, "rect", {x: bb.x - 4, y: bb.y - 2, width: bb.width + 8, height: bb.height + 4,
                              rx: 4, fill, stroke: opts.stroke || "none", "stroke-width": 1.2});
  g.insertBefore(rect, t);
}

function draw(root, d, cents) {
  const svg = root.svg;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const [x0, x1, y0, y1] = d.lims;
  const sx = x => L + (x - x0) / (x1 - x0) * PW;
  const sy = y => T + PH - (y - y0) / (y1 - y0) * PH;
  const pal = d.palette;

  // title
  el(svg, "text", {x: L + PW / 2, y: 18, "text-anchor": "middle", "font-size": 16,
                   "font-weight": 700, fill: INK}, d.title);
  // grid and ticks
  const step = (x1 - x0) <= 12 ? 1 : 2;
  const grid = el(svg, "g", {});
  for (let v = Math.ceil(x0); v <= x1; v += step) {
    el(grid, "line", {x1: sx(v), x2: sx(v), y1: T, y2: T + PH, stroke: GRID});
    el(grid, "text", {x: sx(v), y: T + PH + 16, "text-anchor": "middle", "font-size": 12, fill: "#5b6475"}, v);
  }
  for (let v = Math.ceil(y0); v <= y1; v += step) {
    el(grid, "line", {x1: L, x2: L + PW, y1: sy(v), y2: sy(v), stroke: GRID});
    el(grid, "text", {x: L - 8, y: sy(v), "text-anchor": "end", "dominant-baseline": "middle",
                      "font-size": 12, fill: "#5b6475"}, v);
  }
  el(grid, "line", {x1: L, x2: L + PW, y1: T + PH, y2: T + PH, stroke: "#8a8f99"});
  el(grid, "line", {x1: L, x2: L, y1: T, y2: T + PH, stroke: "#8a8f99"});
  el(grid, "text", {x: L + PW - 2, y: T + PH - 8, "text-anchor": "end", "font-size": 13,
                    "font-style": "italic", fill: "#5b6475"}, "x");
  el(grid, "text", {x: L + 8, y: T + 12, "font-size": 13, "font-style": "italic", fill: "#5b6475"}, "y");

  const pts = d.points, n = pts.length, small = n <= 30;
  const haveAll = cents.length === d.k;

  // dashed lines from each point to its own centroid (after Assign)
  if (d.show_lines && (d.phase === "assigned" || d.phase === "converged") && haveAll) {
    for (const [, x, y, lab] of pts) if (lab >= 0)
      el(svg, "line", {x1: sx(x), y1: sy(y), x2: sx(cents[lab][0]), y2: sy(cents[lab][1]),
                       stroke: pal[lab], "stroke-opacity": 0.5, "stroke-width": 1.5, "stroke-dasharray": "6 4"});
  }
  // old centroid ghosts and arrows (after Update)
  if (d.prev && d.phase === "updated") {
    d.prev.forEach((a, j) => {
      diamond(svg, sx(a[0]), sy(a[1]), 11, {fill: pal[j], "fill-opacity": 0.2, stroke: pal[j], "stroke-opacity": 0.4});
      const b = cents[j];
      const mk = "arrow" + j;
      const defs = el(svg, "defs", {});
      const m = el(defs, "marker", {id: mk, viewBox: "0 0 10 10", refX: 9, refY: 5,
                                    markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"});
      el(m, "path", {d: "M 0 0 L 10 5 L 0 10 z", fill: pal[j]});
      if (Math.hypot(sx(b[0]) - sx(a[0]), sy(b[1]) - sy(a[1])) > 16)
        el(svg, "line", {x1: sx(a[0]), y1: sy(a[1]), x2: sx(b[0]), y2: sy(b[1]), stroke: pal[j],
                         "stroke-width": 2, "stroke-dasharray": "2 4", "marker-end": `url(#${mk})`});
    });
  }
  // selected point: solid lines to every centroid with the distance
  const sel = d.selected;
  if (sel !== null && sel < n && haveAll) {
    const [, px, py] = pts[sel];
    cents.forEach((c, j) => {
      el(svg, "line", {x1: sx(px), y1: sy(py), x2: sx(c[0]), y2: sy(c[1]), stroke: pal[j], "stroke-width": 2.2});
    });
  }
  // points
  const size = small ? 8 : 5.5;
  pts.forEach(([name, x, y, lab, changed], i) => {
    if (changed) el(svg, "circle", {cx: sx(x), cy: sy(y), r: size + 7, fill: "none", stroke: CHANGED, "stroke-width": 2.4});
    if (i === sel) el(svg, "circle", {cx: sx(x), cy: sy(y), r: size + 11, fill: "none", stroke: INK, "stroke-width": 2.4});
    el(svg, "circle", {cx: sx(x), cy: sy(y), r: size, fill: lab >= 0 ? pal[lab] : GREY,
                       stroke: "white", "stroke-width": 1.8});
    if (small || i === sel) {
      const full = d.show_coords && (n <= 12 || i === sel);
      el(svg, "text", {x: sx(x) + 10, y: sy(y) - 9, "font-size": 13, fill: INK},
         full ? `${name} (${fmt(x)}, ${fmt(y)})` : name);
    }
  });
  // distance labels for the selected point (drawn above the points)
  if (sel !== null && sel < n && haveAll) {
    const [, px, py] = pts[sel];
    cents.forEach((c, j) => {
      const dist = Math.hypot(px - c[0], py - c[1]);
      labelBox(svg, (sx(px) + sx(c[0])) / 2, (sy(py) + sy(c[1])) / 2, dist.toFixed(2),
               "white", pal[j], {anchor: "middle", stroke: pal[j]});
    });
  }
  // centroids
  cents.forEach((c, j) => {
    diamond(svg, sx(c[0]), sy(c[1]), 13, {fill: pal[j], stroke: INK, "stroke-width": 2.2});
    const txt = d.show_coords ? `μ${j + 1} (${fmt(c[0])}, ${fmt(c[1])})` : `μ${j + 1}`;
    labelBox(svg, sx(c[0]) + 14, sy(c[1]) + 18, txt, pal[j], "white");
  });
  // clicks not yet processed by Python: show them straight away
  let manualLeft = d.manual_left;
  for (const c of root.pending) {
    if (c.id <= root.ack) continue;
    if (manualLeft > 0) {
      manualLeft -= 1;
      diamond(svg, sx(c.x), sy(c.y), 13, {fill: "white", stroke: INK, "stroke-width": 2, "stroke-dasharray": "3 3"});
    } else if (d.mode === "remove") {
      el(svg, "circle", {cx: sx(c.x), cy: sy(c.y), r: 12, fill: "none", stroke: "#b8336a", "stroke-width": 2});
    } else {
      el(svg, "circle", {cx: sx(c.x), cy: sy(c.y), r: size, fill: GREY, stroke: "white", "stroke-width": 1.8,
                         opacity: 0.8});
    }
  }
}

function toData(root, evt) {
  const p = root.svg.createSVGPoint();
  p.x = evt.clientX; p.y = evt.clientY;
  const q = p.matrixTransform(root.svg.getScreenCTM().inverse());
  if (q.x < L || q.x > L + PW || q.y < T || q.y > T + PH) return null;
  const [x0, x1, y0, y1] = root.data.lims;
  return [Math.round((x0 + (q.x - L) / PW * (x1 - x0)) * 10) / 10,
          Math.round((y0 + (T + PH - q.y) / PH * (y1 - y0)) * 10) / 10];
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let root = parentElement.__km;
  if (!root) {
    const wrap = document.createElement("div");
    wrap.className = "km-wrap";
    parentElement.appendChild(wrap);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    wrap.appendChild(svg);
    const readout = document.createElement("div");
    readout.className = "km-readout";
    wrap.appendChild(readout);
    root = parentElement.__km = {svg, readout, pending: [], ack: 0, lastAnim: null, raf: null, counter: 0};

    svg.addEventListener("click", (evt) => {
      const xy = toData(root, evt);
      if (!xy) return;
      const id = Date.now() * 100 + (root.counter++ % 100);
      root.pending.push({id, x: xy[0], y: xy[1]});
      if (!root.animating) draw(root, root.data, root.data.centroids);   // instant feedback
      root.send("clicks", root.pending.filter(c => c.id > root.ack));
    });
    svg.addEventListener("mousemove", (evt) => {
      const xy = toData(root, evt);
      root.readout.textContent = xy ? `x = ${fmt(xy[0])}, y = ${fmt(xy[1])}` : "";
    });
    svg.addEventListener("mouseleave", () => { root.readout.textContent = ""; });
  }
  root.data = data;
  root.send = setTriggerValue;
  root.ack = Math.max(root.ack, data.ack || 0);
  root.pending = root.pending.filter(c => c.id > root.ack);

  // animate centroids from their old to their new positions once per Update step
  const anim = data.anim;
  if (anim && anim.id !== root.lastAnim && anim.from.length === data.centroids.length) {
    root.lastAnim = anim.id;
    if (root.raf) cancelAnimationFrame(root.raf);
    const dur = 650 / (data.speed || 1), t0 = performance.now();
    root.animating = true;
    const frame = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const cents = anim.from.map((a, j) => [a[0] + (data.centroids[j][0] - a[0]) * e,
                                             a[1] + (data.centroids[j][1] - a[1]) * e]);
      draw(root, root.data, cents);
      if (t < 1) root.raf = requestAnimationFrame(frame);
      else { root.raf = null; root.animating = false; draw(root, root.data, root.data.centroids); }
    };
    root.raf = requestAnimationFrame(frame);
  } else if (!root.animating) {
    draw(root, data, data.centroids);
  }
}
"""

_kmeans_plot = st.components.v2.component("kmeans_plot", css=CSS, js=JS)


def kmeans_plot(data, key, on_clicks):
    """Mount the plot. `on_clicks` runs when the browser sends new clicks."""
    return _kmeans_plot(data=data, key=key, on_clicks_change=on_clicks)
