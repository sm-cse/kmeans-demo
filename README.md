# K-means clustering, one step at a time

An interactive classroom demo that runs K-means step by step and shows every number: the distance from each point to each centroid, which cluster each point joins, how each centroid's new position is calculated, and the objective J after every Assign step.

<!-- After deploying, replace the link below with your app URL -->
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)

## What students see

- **Plot**: points coloured by cluster, centroids as diamonds, dashed lines to each point's own centroid, and arrows showing where centroids move in the Update step. Points that changed cluster are circled.
- **Points table**: distance to every centroid (the nearest is ticked), the chosen cluster and the squared distance. Click a row to draw that point's distances on the plot and write out the full Euclidean distance formulas.
- **Centroids table**: coordinates, member count and member names.
- **Calculations**: the full arithmetic for each step — distances for Assign, means for Update, and J = Σ d².
- **J chart**: the objective after each Assign step, showing it never increases.

## Using it in class

1. Pick a dataset in the sidebar. **Worked example** is the 6-point, K = 2 example starting at B (5, 8) and E (6, 6).
2. Choose K and how to place the starting centroids (first K points, random points, k-means++, or **Manual**: click K places on the plot).
3. Press **Step** (or the Space bar) to alternate Assign → Update → Assign … until it converges, or **Run** to play through automatically.
4. **Reset** returns to the same starting centroids so you can replay the run.
5. **Click the plot** to add a point where you click, or click an existing point to show its distance calculations. Switch the sidebar to **Remove a point** to delete points by clicking them.
6. Open **Edit points** under the plot to type exact coordinates.

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/kmeans-teaching-demo.git
cd kmeans-teaching-demo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub (public repositories are simplest).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose this repository and branch `main`, and set the main file path to `app.py`.
4. Optionally pick a custom subdomain, then click **Deploy**.

Every push to `main` redeploys the app automatically.

## Project layout

```
kmeans-teaching-demo/
├── app.py                  # Streamlit interface
├── kmeans_model.py         # K-means logic and the written-out calculations (no UI code)
├── requirements.txt
├── .streamlit/config.toml  # theme
├── tests/test_model.py     # checks the worked example
└── desktop/
    └── kmeans_demo_gui.py  # original tkinter version (offline use)
```

Run the tests with `pip install pytest && pytest`.

## Desktop version

`desktop/kmeans_demo_gui.py` is the original tkinter app. It works offline and also supports right-click to remove points. It needs only `matplotlib`:

```bash
pip install matplotlib
python desktop/kmeans_demo_gui.py
```
