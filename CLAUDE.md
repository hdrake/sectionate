# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sectionate is a Python package for sampling grid-consistent hydrographic sections from structured ocean model outputs. It traces paths along C-grid velocity faces between geographic waypoints and computes transports/tracer values along those sections. It supports any structured model whose grid can be described by an `xgcm.Grid` object.

## Development Setup

**One conda environment per branch/worktree, named `docs_env_sectionate_<branch-or-worktree-name>`.**
Create it from `docs/environment.yml`, overriding the baked-in `name:` with `-n`:

```bash
ENV="docs_env_sectionate_$(git rev-parse --abbrev-ref HEAD)"
conda env create -f docs/environment.yml -n "$ENV"
conda activate "$ENV"
pip install -e .    # resolves the branch's own runtime deps from pyproject.toml
python -m ipykernel install --user --name "$ENV" --display-name "$ENV"
```

`docs/environment.yml` pins none of the runtime dependencies; `pip install -e .` is
what resolves them, so install in that order and re-run it after any dependency
change in `pyproject.toml`. Reuse the env across runs on the same branch; remove it
with `conda env remove -n "$ENV"` once the branch or worktree is gone.
(`ci/environment.yml` is the minimal CI environment — pytest only, no plotting or
notebook stack. It is not sufficient for the notebooks.)

A correctly set up environment runs the full suite with **0 skips**. Skips mean one of
two things: fold and multi-tile tests `skip` on an xgcm older than the floor in
`pyproject.toml`, and the ECCO LLC90 and MOM6-fold tests `skip` when their example data
is absent from `data/`. In a fresh worktree the latter is the usual cause — `data/`
holds ~1.4 GB of downloaded input that each checkout otherwise re-fetches. Symlink it
from a checkout that already has it rather than downloading again (`data/*.nc` is
gitignored, but `MOM5_global_example_grid.nc` is tracked, so link the individual files,
not the directory):

```bash
cd data && for f in /path/to/other/checkout/data/*.nc; do
  b=$(basename "$f"); [ -e "$b" ] || ln -s "$f" "$b"
done
```

## Commands

- **Run all tests:** `pytest`
- **Run a single test:** `pytest sectionate/tests/test_section.py::test_find_closest_grid_point`
- **Build docs:** `cd docs && make html` (requires `docs/environment.yml` environment).
  Add `SPHINXOPTS="-W"` to reproduce Read the Docs, which builds with `fail_on_warning: true`.

## Documentation pages and figures

`docs/source/` is Sphinx: `.rst` and MyST `.md` pages are hand-written, while
`docs/source/examples/` is **generated** — `conf.py` deletes and re-copies it from
`examples/*.ipynb` on every build, so never put a hand-written page or a static asset there.
Both `nbsphinx` (notebooks) and `myst_parser` (`.md`) are enabled. Every new page must be
added to the toctree in `docs/source/index.rst` in the same change, or the build fails.

The figures on `docs/source/algorithm.md` are **committed artifacts**, because Read the Docs
never executes anything (`nbsphinx_execute = "never"`). Regenerate them with:

```bash
python docs/make_algorithm_animation.py    # needs ffmpeg, from docs/environment.yml
```

It writes `docs/source/_static/algorithm/{walk.mp4,walk_steps.png}`. The script replays
`sectionate.section.infer_grid_path` step by step to record per-step state for drawing, then
**asserts that its replayed path matches `grid_section`** — so a change to the walk makes
regeneration fail loudly instead of letting the page drift out of sync. If that assertion
fires, the page prose almost certainly needs updating too. Keep the two artifacts small
(currently ~0.5 MB combined): `docs/**` is not excluded from the sdist by `pyproject.toml`,
so they ship to PyPI and live in git history permanently.

## Definition of Done (always, before committing or pushing)

Before committing **or pushing** any change to this repository, always do all of the
following — use parallel agents where it is faster. Both of the first two steps, every
time: not one or the other, and not only the notebooks that look related to the change.
A change that passes `pytest` but breaks a notebook is still a broken change, because
the notebooks are the rendered documentation.

Run them in this branch's own `docs_env_sectionate_<branch-or-worktree-name>` environment
(see *Development Setup*), not in whatever env happens to be active.

1. **Run the full test suite** (`pytest`) and confirm it passes with **0 skips**.
2. **Re-execute the example notebooks** in `examples/` and confirm they run cleanly:
   ```bash
   cd examples && jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.timeout=1800 \
     --ExecutePreprocessor.kernel_name=python3 *.ipynb
   ```
   Run from `examples/` — the notebooks resolve data paths relative to it. `--inplace`
   refreshes the committed outputs, which is intended; review the resulting diff.
   `kernel_name=python3` uses the active env's kernel, so the per-branch env name need
   not match the `kernelspec` recorded in each notebook.
3. **Scan the repo for inconsistencies / outdated information** introduced by the change —
   including docstrings, `README.md`, files under `docs/`, the example notebooks, and this
   `CLAUDE.md` — and update them so docs and code stay in sync.

## Keeping pull requests in sync

When pushing a new commit to a branch that already has an open PR, also update that PR's
top-comment description so it stays consistent with the most recent commit:

- Revise the summary / "Changes" prose to reflect what the latest commit actually did.
- Update any task/status checklist — check off completed items (`- [ ]` → `- [x]`) and add
  new ones as needed.
- Edit it in place with `gh pr edit <number> --repo <owner>/<repo> --body-file <file>` (find
  the PR for the current branch with `gh pr view --json number,url`).

## AI Usage Policy

AI-assisted contributions to sectionate are welcome, but the person running the AI is
responsible for every change. This project follows the
[xgcm AI Usage Policy](https://github.com/hdrake/xgcm/blob/add-Claude.md/docs/contributor_guide.md),
which in turn adapts [xarray's](https://docs.xarray.dev/en/stable/contribute/ai-policy.html).
It applies to every change regardless of whether it was written by hand, with AI
assistance, or generated entirely by an AI tool. The essentials:

- **You own the diff.** Before opening or updating a PR, you must have read and understood
  every line and be able to explain why each change is correct — the same bar as a
  hand-written PR. Keep changes small, single-purpose, and free of unrelated edits.
- **Disclose AI assistance openly.** Add a `Co-Authored-By:` trailer to AI-assisted commits
  and note on any PR or comment that was drafted with AI.
- **Communicate in your own words.** PR descriptions, issue comments, and review responses
  must be your own; do not paste AI-generated text as a comment. Using AI to polish your own
  writing (grammar, phrasing) is fine as long as it introduces no inaccuracies.
- **Every code change ships with tests** and, per the *Definition of Done* above, a green
  suite and re-executed notebooks.
- **Discuss large AI-assisted contributions first.** For a substantial refactor, new
  subsystem, or migration, open an issue to agree scope before generating the diff — a large
  diff is fast to produce and slow to review.

## Architecture

The package is organized around a pipeline: define sections → map to grid → compute transports/tracers.

### Core Modules

- **`topology.py`** — **Corner identity, and the one place it is decided.** One physical vorticity corner can be stored under several native `(face, j, i)` indices: the two ends of a periodic axis, the mirrored columns of a bipolar fold seam, a corner shared by two tiles. `CornerTopology` resolves which indices are the same corner **from the grid's declared metadata alone** — never from coordinates. Every identification a structured grid can express is an affine integer map on the corner lattice, uniform along its seam (periodic `(J,I)≡(J,I+Nxc)`; a fold's seam row glued to its own mirror, taken from xgcm's own `_seam_partner_indices` so a fold means the same thing here as when xgcm pads across one; each `face_connections` gluing, where `reverse` means the two faces meet on the *same* side and the tangential direction then follows from the gluing being orientation-preserving). The topology is the quotient of the lattice by the group these generate, labelled with `scipy.sparse.csgraph.connected_components`. Extra `Identification`s can be handed in for topology `face_connections` cannot express — its schema maps a whole tile edge to a whole tile edge, so it cannot describe an edge glued to several partners over different index ranges. Everything is on the **'outer'** lattice `(Nyc+1, Nxc+1)` per face whatever the native staggering, so every corner of every cell has a slot even where the arrays drop a row. Provides: `node_id`, `node_native` (canonical `(f,j,i)`, or none), `node_lon`/`node_lat` (exact where stored, else the spherical centroid of the tracer cells meeting there), `neighbors`/`valence`/`edges`, `edge_velocities` (each face in the frame it is stored in, with the step through it, so a fold's duplicated sign-flipped row cancels correctly), `face_handedness` (measured on the sphere, per face, and raising if a face is not consistently oriented), `padded_transports`, and `validate_positions` — the one place coordinates are consulted, to *check* the topology and never to set it.

- **`walk.py`** — The walk, on the corner graph rather than on indices. Three rules: enumerate the current corner's neighbours dropping the one just came from; admit those strictly closer to the endpoint; step to the one closest to the requested curve, ties broken by index. It has no notion of direction, no wrap-around arithmetic and no test for whether two indices are the same corner, because a periodic wrap, a fold seam and a rotated tile boundary are all just edges. Corners that are distinct but geographically coincident (a bipolar cap's singular meridian) are crossed as a *plateau*: the walk chooses where to leave the group from the neighbours outside it, so a flat metric cannot leave the route to index order. `native_path` then writes the visited corners back as native indices, emitting both spellings of a seam corner where the crossing changes frame — done while the walk still knows which side it is on, rather than repaired afterwards. Corners the grid stores on no face are real corners the walk may cross, but are routed around where a stored alternative exists, since a section through one cannot be written in native indices.

- **`section.py`** — Section definition and the public entry point. `Section` holds named waypoint coordinates; `GriddedSection` extends it with grid indices and `n_c`, the physical corner each point is (always derived, never persisted). `grid_section()` maps geographic waypoints to vorticity-point indices, always returning `(i_c, j_c, f_c, lons_c, lats_c)` — a single-tile grid is one face, so `f_c` is zeros rather than absent. `curve` is one of `"great circle"` (default), `"latitude circle"` (raises for a segment whose endpoints do not share a latitude), or `"latitude and great circle"` (decided per segment). Under all three each segment takes the **shortest** path between its waypoints — raw longitudes are never read as a request to go the long way round, so `0 -> 270` runs 90° *west*, and encircling the globe takes intermediate waypoints. `_check_segment_span` resolves each segment's curve and raises for endpoints written exactly half a circle apart. Grid topology is inferred entirely from `xgcm.Grid` metadata, so there is no `topology` keyword.

- **`transports.py`** — Transport computation along sections. `uvindices_from_qindices()` resolves the given indices to corner nodes and emits one velocity face per step between *distinct* nodes: two spellings of one corner compare equal as integers and emit nothing, which is how a seam crossing is written and why it is not counted twice. A pair that is not a grid edge raises rather than quietly losing a face. It also reports `q`, the corner step each face came from, since faces and steps are not one to one. Input is liberal — a section traced here, traced on a cell mask, or reloaded from saved indices all give the same faces. The per-edge sign `Lsign` is read entirely in the frame the velocity is stored in (does the step run along `+i` or `+j`, and is this face's `+x` or `+y`?), times a per-face handedness measured once; that is index arithmetic, so it is exact where a cross product of two local displacements is degenerate. `convergent_transport()` accumulates signed normal transports with configurable orientation (positive inward to the polygon defined by the section).

- **`tracers.py`** — `extract_tracer()` interpolates tracer data to U/V points along a section path for cross-section plotting.

- **`gridutils.py`** — Grid introspection only: `corner_position()` returns the shared vorticity corner position (`"outer"`, `"right"`, `"left"`) and `corner_offset()` the corresponding velocity index shift; `check_outer()` is a thin wrapper (True iff `"outer"`). `coord_dict()`, `get_geo_corners()` and `get_facedim()` extract coordinate/dimension names from `xgcm.Grid` metadata. (Package source names positions only by these xgcm labels; their correspondence to MOM6/MITgcm/ECCO conventions is documented under "Corner staggering" below and in the example notebooks.) All corner topology lives in `topology.py`.

- **`utils.py`** — Section catalog I/O. Sections can be loaded by name from JSON catalog files in `sectionate/catalog/`. Also provides `save_gridded_section()`/`load_gridded_section()` for persisting gridded sections.

### Key Concepts

- **Vorticity points (q-points):** Sections are paths through vorticity-point indices `(i_c, j_c, f_c)`. Consecutive q-points define velocity faces (either U or V).
- **Corner identity:** A grid may spell one corner several ways, and which indices denote the same corner is decided **combinatorially, from what the grid declares** — see `topology.py`. Geographically coincident corners that the metadata distinguishes stay distinct: a tripolar grid's singular meridian stores one position for a whole column of corners, and collapsing them would leave a closed section through the cap unable to say which of the fan's sectors it enclosed. Conversely a grid whose stored coordinates do not carry the fold it declares is still resolved correctly, and `validate_positions` reports the disagreement.
- **Corner staggering (three positions):** Vorticity sits at one of three xgcm positions: `"outer"` (MOM6 symmetric, M+1×N+1), `"right"` (MOM6 non-symmetric, M×N), or `"left"` (MITgcm/ECCO, incl. the lat-lon-cap, M×N). All three are native; they differ only by a per-position velocity index offset (`gridutils.corner_offset`: outer→0, right→+1, left→0). `"left"` indexes like `"outer"`; it differs only in array length and in that the *high* corner row/column is absent (so a section exactly on the north/east domain wall clips one row inside).
- **Sign conventions:** For a *closed* section, `convergent_transport()` determines orientation (clockwise/counterclockwise) using stereographic projection and signed polygon area, then applies sign corrections so positive transport means "inward" (toward the enclosed polygon). For an *open* section there is no enclosing polygon, so `positive_in` is undefined; it instead uses the **left-of-transect** convention — `positive_in=True` makes positive transport point to the left of the section as traversed from the first to the last waypoint — and emits a `UserWarning`. `is_section_counterclockwise()` is only consulted in the closed case.
- **xgcm.Grid dependency:** The package relies heavily on `xgcm.Grid` for grid metadata (axis boundaries, coordinate positions, dataset access via `grid._ds`).

## Version

**The git tag is the version.** There is no version string checked into the tree:
`hatch-vcs` derives it from the tag at build time and writes the gitignored
`sectionate/_version.py`, which ships inside the sdist and wheel. `sectionate/version.py`
is only a shim over that generated file — never add a version literal back to it. A
`0.0.0+unknown` from it means the package was imported without being built or installed,
not that a number is missing. Any checkout that installs the package needs its tags: a
shallow clone resolves a `.devN` version instead of the release line, which is why CI and
Read the Docs both fetch with full depth. See the Releasing section of `README.md`.
