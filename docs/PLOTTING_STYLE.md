# Paper Plotting Style

Use this style for any new plot intended for the manuscript, supplementary
material, or reviewer response. The goal is to keep new figures visually
consistent with the existing paper figures.

## General Rules

- Use `seaborn` and `matplotlib`; do not introduce a new plotting library.
- Use PDF for paper-review previews. SVG/PNG can be exported additionally.
- Use `FONTSIZE = 23` for titles, axis labels, and major tick labels unless the
  figure is too dense.
- Prefer line plots with markers over bar charts for accuracy/runtime trends.
- Avoid `seaborn.relplot`/facet defaults for paper figures; use explicit
  `fig, axes = plt.subplots(...)` so spacing, titles, ticks, and legends match
  the paper.
- Keep legends boxed when they are inside a subplot: `frameon=True`,
  `fancybox=False`, linewidth about `0.8`, white facecolor, light gray edge.
- Use dashed grid lines for accuracy/qubit figures:
  `ax.grid(axis="both", linestyle="--", alpha=0.7)`.

## Accuracy vs Qubits Figures

Use this for Figure 6-style comparisons, transfer/generalization plots,
predefined feature-map baselines, random circuit baselines, and related qubit
sweeps.

- Theme: `sns.set_theme(style="white")`
- Layout: `fig, axes = plt.subplots(1, 3, figsize=(19, 7), sharex=True, sharey=True)`
- Dataset order: Breast Cancer, Digits, Wine
- Titles: `"{Dataset} Dataset"`, `fontsize=23`, `y=0.94`
- X-axis: label `"Number of Qubits"`, ticks `[3, 4, 5, 6, 7]`
- Y-axis: label `"Accuracy (%)"` only on the first subplot
- Y ticks: normally `np.arange(45, 101, 10)`
- X limits: approximately `(2.8, 7.2)`
- Line width: `2.5`
- Marker size: around `11-12` for matplotlib `plot/errorbar`
- Use shared visual encoding for common models:

```python
markers = {
    "GA-FQK": "o",
    "GA-PQK": "D",
    "FQK": "^",
    "PQK": "v",
    "RBF": "s",
}
dashes = {
    "GA-FQK": (3, 4, 1, 4),
    "GA-PQK": (6, 2),
    "FQK": (1, 1),
    "PQK": (6, 2),
    "RBF": (2, 1),
}
```

## Generation / Fitness Curves

Use this for Figure 4-style GA convergence and readout-noise curves.

- Theme: `sns.set_theme(style="whitegrid")`
- Single panel size: around `(10, 6)`
- Four-panel benchmark size: `(22, 6)`, `sharey=True`
- Use `plt.cm.viridis_r` or `sns.color_palette("viridis_r", ...)` for ordered
  parameter sweeps.
- Plot mean as a solid line and standard deviation as a transparent band when
  repeated runs exist:

```python
ax.plot(generation, mean, linestyle="-", linewidth=2.5, color=color)
ax.fill_between(generation, mean - std, mean + std, color=color, alpha=0.2)
```

- If there is only one run, do not draw fake variance. Plot only the observed
  curve and state in the caption/notes that no repeated-run standard deviation
  is available.
- Axis labels: `"Generation"` and `"Best Fitness"`.
- Legend: place inside the subplot when possible, usually lower right, boxed.

## Runtime Figures

- Use the same 1x3 dataset layout as accuracy-vs-qubit figures when comparing
  runtime across datasets.
- Use line plots with markers, not bars.
- Y-axis may be independent across datasets if runtimes differ by orders of
  magnitude; state this clearly if used.
- Label runtime units explicitly, e.g. `"Mean Runtime (min)"` or
  `"Runtime (s)"`.

## Output Locations

- Reviewer preview PDFs should go under `results/reviewer/plots_pdf/`.
- Stable plot scripts should live under `scripts/reviewer/`.
- Do not overwrite manuscript source figures unless explicitly requested.
