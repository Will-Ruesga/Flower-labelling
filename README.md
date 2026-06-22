# Flower Labelling

A Flask web app for prompt-driven image labelling with SAM3. Point it at a folder of images or a CSV manifest, write a text prompt, and produce a CSV with per-image masks and review status.

## Architecture

Eight modules with a strict one-way dependency chain:

```
presentation → application → run_context
                           → data_loader
                           → inference
                           → results
                           → viewer_cache → inference
                           → shared (used by all)
```

Each module-public function returns `ModuleResult(ok, message, data)` — except `viewer_cache`, a pure utility that returns bytes / `None` / tuples directly. State lives in a single in-memory `RunContext` owned by `run_context` and mutated only by `application`. `viewer_cache` keeps its own private in-memory cache (decoded frames + encoded PNGs) and never touches `RunContext`.

## Requirements

- Python 3.10+
- Vendored `sam3/` package and `checkpoints/sam3.pt` in the project root.

### sam3 patch (one-time)

`sam3/model_builder.py` uses `pkg_resources` (from `setuptools`) to locate the BPE vocab file. This module is not reliably available in modern venvs (e.g. uv on Python 3.12). Apply this patch once after cloning:

```diff
-import pkg_resources
+from pathlib import Path as _Path
```

and replace both `pkg_resources.resource_filename(...)` calls with:

```python
str(_Path(__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
```

This patch is already applied in this repo. If you refresh `sam3/` from upstream, re-apply it.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Open http://127.0.0.1:5000.

## Workflow

1. **Setup — stage 1 (probe).** Choose source type (`folder` or `csv`), enter the path, click **Set source**. The app loads the dataset and detects pages-per-image from the first image.
2. **Setup — stage 2 (pages).** Tick which pages to label (with **All** / **None** shortcuts), click **Confirm**. This loads the SAM3 model and locks setup.
3. **Prompt.** Write a text prompt, choose `single` or `multiple` mask mode.
4. **Label.**
   - **Label page** runs SAM3 on the current page only; the mask + bounding box are cached and stay visible when you switch pages within the same image.
   - **Label image** runs SAM3 across every selected page of the current image.
   - **Label dataset** runs SAM3 unattended across every image for every selected page, auto-labels each row (`incorrect` if any mask was found, else `discard`), writes the CSV, and ends the run with a **Labelling finished!** screen.
   - **Clear page / Clear image** wipe the in-memory cache without touching the CSV.
5. **Decision.** **Correct / Incorrect / Discard** builds one row from the current image's cached pages, saves it, clears that image's cache, and advances to the next image.
6. **Reset.** Throws away the run and unlocks setup.

The per-image cache only survives navigation within the same image. Submitting a decision, navigating to a new image, or calling **Label dataset** wipes it.

## Output CSV schema (mandatory)

Output files are named `labels_<prompt-slug>_<mode>_p<pages>.csv` and written to the dataset folder (e.g. `labels_red-flower_single_p0-1.csv`). The slug is kebab-cased; fields are separated by single underscores so boundaries stay unambiguous. The schema is fixed — do not rearrange columns:

```
sep=;
fileName;status;ZoomX0;ZoomY0;ZoomWidth0;ZoomHeight0;…;ZoomX{N-1};ZoomY{N-1};ZoomWidth{N-1};ZoomHeight{N-1};Mask0;…;Mask{N-1}
```

- First line is the literal `sep=;` Excel hint. The separator is `;`.
- `fileName` is the image basename.
- `status` is one of `correct`, `incorrect`, `discard`.
- For each page `i` in `[0, N)`, four integer columns `ZoomX{i}`, `ZoomY{i}`, `ZoomWidth{i}`, `ZoomHeight{i}` hold the tight bounding rectangle around every predicted box on that page (xywh in image pixels). Empty strings for unlabelled pages.
- All page bboxes come first, then **all** `Mask{i}` columns at the end.
- `Mask{i}` is the full mask cell for page `i` — a nested polygon string with three levels:
  - Outer `[ … ]` — the **mask** (page-level cell).
  - Middle `[ … ]` — one slot per **object** (e.g. one flower) detected on the page.
  - Inner `{X:[…],Y:[…]}` — one entry per **region** (contour) of that object; an object split into disconnected pieces has multiple regions.

  Regions come from `cv2.findContours` with `RETR_TREE` + `CHAIN_APPROX_NONE`. Coordinates use compact float formatting (`0` for zero, integers with no decimal, fractions below 1 stripped of the leading zero). Example — two objects, first with 2 regions, second with 3:
  ```
  [[{X:[…],Y:[…]},{X:[…],Y:[…]}],[{X:[…],Y:[…]},{X:[…],Y:[…]},{X:[…],Y:[…]}]]
  ```
  Pages with no objects write the literal `"[[]]"`.

Example header for a 2-page dataset:

```
fileName;status;ZoomX0;ZoomY0;ZoomWidth0;ZoomHeight0;ZoomX1;ZoomY1;ZoomWidth1;ZoomHeight1;Mask0;Mask1
```

Rows are upserted by `fileName`: re-running against the same source replaces existing rows for the same file rather than duplicating them.
