"""Use-case orchestrator — the only module that mutates ``RunContext``."""

import os
from pathlib import Path
from typing import Iterator

import data_loader
import inference
import results
import run_context
import viewer_cache
from config import TASK_CORRECT, TASK_DISCARD, TASK_INCORRECT
from shared import ModuleResult, RunContext, RunViewState


_VALID_STATUSES = {TASK_CORRECT, TASK_INCORRECT, TASK_DISCARD}


def start_run() -> ModuleResult:
    """Return a view of the current context without mutating it."""
    return ModuleResult(ok=True, data=_view(run_context.get_context(), "info", ""))


def probe_source(source_type: str, source_path: str) -> ModuleResult:
    """Validate and scan the source, populate dataset state. Does not load the model."""
    ctx = run_context.get_context()
    if ctx.setup.locked:
        return _fail(ctx, "already locked; reset to change setup", "Already locked. Reset to change setup.")

    r = data_loader.validate_source(source_type, source_path)
    if not r.ok:
        return _fail(ctx, r.message)

    r = data_loader.load_dataset(source_type, source_path)
    if not r.ok:
        return _fail(ctx, r.message)
    manifest = r.data

    r = data_loader.probe_num_pages(manifest)
    if not r.ok:
        return _fail(ctx, r.message)
    expected = r.data

    ctx.setup.source_type = source_type
    ctx.setup.source_path = source_path
    ctx.setup.pages_to_label = []
    ctx.setup.probed = True
    ctx.dataset.imgs_paths = manifest.imgs_paths
    ctx.dataset.dataset_root = manifest.dataset_root
    ctx.dataset.expected_num_pages = expected
    ctx.dataset.header = []
    root = Path(manifest.dataset_root)
    ctx.dataset.has_subdirs = any(Path(p).parent != root for p in manifest.imgs_paths)
    viewer_cache.clear()

    msg = f"Source has {len(manifest.imgs_paths)} images, {expected} page(s) each."
    return ModuleResult(ok=True, data=_view(ctx, "ok", msg))


def confirm_setup(pages_to_label: list[int]) -> ModuleResult:
    """Validate the page selection, load the model, and lock the setup."""
    ctx = run_context.get_context()
    if ctx.setup.locked:
        return _fail(ctx, "already locked; reset to change setup", "Already locked. Reset to change setup.")
    if not ctx.setup.probed:
        return _fail(ctx, "set source first", "Set source first.")

    r = data_loader.validate_pages(pages_to_label, ctx.dataset.expected_num_pages)
    if not r.ok:
        return _fail(ctx, r.message)
    pages = r.data

    r = inference.ensure_model_ready()
    if not r.ok:
        return _fail(ctx, r.message)

    ctx.setup.pages_to_label = pages
    ctx.setup.locked = True
    ctx.dataset.header = data_loader.build_header_for_pages(ctx.dataset.expected_num_pages)
    ctx.navigation.current_image_idx = 0
    ctx.navigation.current_page_idx = pages[0]

    return ModuleResult(ok=True, data=_view(ctx, "ok", "Setup locked."))


def reset_run() -> ModuleResult:
    """Throw away the context and start a fresh one."""
    run_context.reset_context()
    viewer_cache.clear()
    return ModuleResult(ok=True, data=_view(run_context.get_context(), "info", "Reset."))


def set_prompt(text: str) -> ModuleResult:
    """Store the user's prompt text. Reject an empty/whitespace-only prompt."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    if not text.strip():
        return _fail(ctx, "prompt is empty", "Enter a prompt.")
    ctx.prompt.text = text
    return ModuleResult(ok=True, data=_view(ctx, "ok", "Prompt saved."))


def set_generation_mode(mode: str) -> ModuleResult:
    """Switch between ``single`` and ``multiple`` mask modes."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    if mode not in {"single", "multiple"}:
        return _fail(ctx, "invalid mode", "Invalid mode.")
    ctx.prompt.mode = mode
    return ModuleResult(ok=True, data=_view(ctx, "ok", ""))


def set_confidence_threshold(threshold: float) -> ModuleResult:
    """Store the threshold; if the current page has a cached backbone state, rerun heads now."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    if not (0.0 < threshold < 1.0):
        return _fail(ctx, "threshold out of range", "Threshold must be between 0 and 1.")

    ctx.prompt.threshold = threshold
    page_idx = ctx.navigation.current_page_idx
    state_cpu = ctx.output.sam_states.get(page_idx)
    if state_cpu is not None:
        r = inference.rerun_with_threshold(state_cpu, threshold, ctx.prompt.mode)
        if not r.ok:
            return _fail(ctx, r.message)
        output, new_state_cpu = r.data
        ctx.output.page_outputs[page_idx] = output
        ctx.output.sam_states[page_idx] = new_state_cpu

    return ModuleResult(ok=True, data=_view(ctx, "ok", ""))


def navigate_page(step: int) -> ModuleResult:
    """Move the page pointer by ``step`` within ``pages_to_label``, wrapping at the ends."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    pages = ctx.setup.pages_to_label
    current = ctx.navigation.current_page_idx
    idx_in_list = pages.index(current) if current in pages else 0
    ctx.navigation.current_page_idx = pages[(idx_in_list + step) % len(pages)]
    return ModuleResult(ok=True, data=_view(ctx, "ok", ""))


def jump_page(page_idx: int) -> ModuleResult:
    """Jump directly to ``page_idx`` (must be in ``pages_to_label``)."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    if page_idx not in ctx.setup.pages_to_label:
        return _fail(ctx, f"page {page_idx} not selected", "Page not selected.")
    ctx.navigation.current_page_idx = page_idx
    return ModuleResult(ok=True, data=_view(ctx, "ok", ""))


def label_current_page() -> ModuleResult:
    """Run SAM3 on the current page and cache the result by page index."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx) or _require_prompt(ctx)
    if guard is not None:
        return guard
    page_idx = ctx.navigation.current_page_idx
    err = _run_and_cache(ctx, [page_idx])
    if err is not None:
        return _fail(ctx, err)
    return ModuleResult(ok=True, data=_view(ctx, "ok", f"Labelled page {page_idx}."))


def label_current_image() -> ModuleResult:
    """Run SAM3 on every selected page of the current image; cache per page."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx) or _require_prompt(ctx)
    if guard is not None:
        return guard
    err = _run_and_cache(ctx, ctx.setup.pages_to_label)
    if err is not None:
        return _fail(ctx, err)
    name = Path(ctx.dataset.imgs_paths[ctx.navigation.current_image_idx]).name
    return ModuleResult(ok=True, data=_view(ctx, "ok", f"Labelled all pages of {name}."))


def label_dataset_stream() -> Iterator[ModuleResult]:
    """Stream a bulk pass over the whole dataset, yielding one view per labelled image.

    Skips images already present in the target CSV (resume-on-replay), saves each
    row immediately, and ends with the ``finished`` view (context reset).
    """
    ctx = run_context.get_context()
    guard = _require_locked(ctx) or _require_prompt(ctx)
    if guard is not None:
        yield guard
        return
    yield from _bulk_run(ctx, ctx.dataset.imgs_paths)
    yield _finish_run(ctx)


def label_subdirectory_stream() -> Iterator[ModuleResult]:
    """Stream a bulk pass over the current image's subfolder, then advance.

    Skips images already in the CSV. If this was the last subfolder in sort
    order, ends with the ``finished`` view; otherwise jumps the pointer to
    the first image of the next subfolder so the user can change the prompt.
    """
    ctx = run_context.get_context()
    guard = _require_locked(ctx) or _require_prompt(ctx)
    if guard is not None:
        yield guard
        return

    current_image = ctx.dataset.imgs_paths[ctx.navigation.current_image_idx]
    subfolder = Path(current_image).parent
    subdir_paths = [p for p in ctx.dataset.imgs_paths if Path(p).parent == subfolder]
    pre_labelled = _labelled_filenames(ctx)
    did_work = any(
        _rel_name(p, ctx.dataset.dataset_root) not in pre_labelled for p in subdir_paths
    )

    yield from _bulk_run(ctx, subdir_paths)

    rel = subfolder.relative_to(ctx.dataset.dataset_root).as_posix() or "."
    next_idx = _next_subfolder_first(ctx, subfolder)
    if next_idx is None:
        yield _finish_run(ctx)
        return

    ctx.navigation.current_image_idx = next_idx
    ctx.navigation.current_page_idx = ctx.setup.pages_to_label[0]
    ctx.output.page_outputs.clear()
    ctx.output.sam_states.clear()
    viewer_cache.evict_old({next_idx, next_idx + 1})
    if did_work:
        msg = f"Labelled subfolder {rel}/. Moved to next."
    else:
        msg = f"Subfolder {rel}/ already labelled. Moved to next."
    yield ModuleResult(ok=True, data=_view(ctx, "ok", msg))


def label_dataset() -> ModuleResult:
    """No-JS fallback: drain ``label_dataset_stream`` and return its final state."""
    return _drain(label_dataset_stream())


def label_subdirectory() -> ModuleResult:
    """No-JS fallback: drain ``label_subdirectory_stream`` and return its final state."""
    return _drain(label_subdirectory_stream())


def clear_current_page() -> ModuleResult:
    """Erase the cached output for the current page only."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    ctx.output.page_outputs.pop(ctx.navigation.current_page_idx, None)
    return ModuleResult(ok=True, data=_view(ctx, "ok", "Cleared page."))


def clear_current_image() -> ModuleResult:
    """Erase every cached page output for the current image."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    ctx.output.page_outputs.clear()
    return ModuleResult(ok=True, data=_view(ctx, "ok", "Cleared image."))


def get_viewer_png(img_idx: int, page_idx: int) -> ModuleResult:
    """Return PNG bytes for a single page (with overlay if it belongs to the current image).

    ``ok=False`` when ``img_idx`` is out of range so the caller can render a 404.
    """
    ctx = run_context.get_context()
    if img_idx < 0 or img_idx >= len(ctx.dataset.imgs_paths):
        return ModuleResult(ok=False, message="image index out of range")
    image_path = ctx.dataset.imgs_paths[img_idx]
    page_output = (
        ctx.output.page_outputs.get(page_idx)
        if img_idx == ctx.navigation.current_image_idx
        else None
    )
    png = viewer_cache.get_png(image_path, img_idx, page_idx, page_output or {})
    return ModuleResult(ok=True, data=png)


def submit_decision(status: str) -> ModuleResult:
    """Finalize the current image with ``status``, save CSV, advance to next."""
    ctx = run_context.get_context()
    guard = _require_locked(ctx)
    if guard is not None:
        return guard
    if status not in _VALID_STATUSES:
        return _fail(ctx, "invalid status", "Invalid status.")
    # Correct/Incorrect describe a detection, so they require one. Discard means
    # "nothing here, move on" — it's always allowed (even with no labelled page)
    # so the user can skip empty images straight to the next one.
    if status in (TASK_CORRECT, TASK_INCORRECT) and not _has_detection(ctx):
        return _fail(
            ctx,
            "no detection to mark correct/incorrect; use discard to skip",
            "Nothing detected — only Discard is available.",
        )

    image_path = ctx.dataset.imgs_paths[ctx.navigation.current_image_idx]
    num_pages = ctx.dataset.expected_num_pages
    page_outputs = [ctx.output.page_outputs.get(i) for i in range(num_pages)]

    r = results.build_row(
        image_path=image_path,
        num_pages=num_pages,
        page_outputs=page_outputs,
        status_label=status,
        header=ctx.dataset.header,
        dataset_root=ctx.dataset.dataset_root,
    )
    if not r.ok:
        return _fail(ctx, r.message)
    ctx.output.pending_rows.append(r.data)

    r = results.save_rows(ctx.output.pending_rows, _save_metadata(ctx))
    if not r.ok:
        return _fail(ctx, r.message)
    ctx.output.last_save_report = r.data

    total = len(ctx.dataset.imgs_paths)
    next_idx = ctx.navigation.current_image_idx + 1
    if next_idx >= total:
        ctx.navigation.current_image_idx = total - 1
        message = "Dataset complete."
    else:
        ctx.navigation.current_image_idx = next_idx
        message = f"Saved. {next_idx + 1}/{total}."
    ctx.navigation.current_page_idx = ctx.setup.pages_to_label[0]
    ctx.output.page_outputs.clear()
    ctx.output.sam_states.clear()

    keep = {ctx.navigation.current_image_idx}
    if ctx.navigation.current_image_idx + 1 < total:
        keep.add(ctx.navigation.current_image_idx + 1)
    viewer_cache.evict_old(keep)

    return ModuleResult(ok=True, data=_view(ctx, "ok", message))


def _view(ctx: RunContext, status: str, message: str) -> RunViewState:
    """Build a ``RunViewState`` snapshot. Includes the viewer URL when locked."""
    return RunViewState(
        setup=ctx.setup,
        prompt=ctx.prompt,
        navigation=ctx.navigation,
        dataset_size=len(ctx.dataset.imgs_paths),
        expected_num_pages=ctx.dataset.expected_num_pages,
        labelled_pages=sorted(ctx.output.page_outputs.keys()),
        viewer_image=_render_current_viewer(ctx) if ctx.setup.locked else None,
        status=status,
        message=message,
        has_subdirs=ctx.dataset.has_subdirs,
        current_subfolder=_current_subfolder(ctx),
        labelled_total_count=len(_labelled_filenames(ctx)),
        has_detection=_has_detection(ctx),
        finished=ctx.finished,
        finished_message=ctx.finished_message,
    )


def _require_locked(ctx: RunContext) -> ModuleResult | None:
    """Return an error ``ModuleResult`` if the setup is not yet locked, else None."""
    if not ctx.setup.locked:
        return _fail(ctx, "setup not locked", "Confirm setup first.")
    return None


def _has_detection(ctx: RunContext) -> bool:
    """True if any labelled page of the current image produced at least one mask."""
    return any(out and out.get("masks") for out in ctx.output.page_outputs.values())


def _require_prompt(ctx: RunContext) -> ModuleResult | None:
    """Return an error ``ModuleResult`` if the prompt is empty, else None."""
    if not ctx.prompt.text.strip():
        return _fail(ctx, "prompt is empty", "Enter a prompt.")
    return None


def _save_metadata(ctx: RunContext) -> dict:
    """Build the metadata dict consumed by ``results.save_rows``."""
    return {
        "prompt": ctx.prompt.text,
        "mode": ctx.prompt.mode,
        "pages_labeled": ctx.setup.pages_to_label,
        "dataset_root": ctx.dataset.dataset_root,
        "header": ctx.dataset.header,
    }


def _fail(ctx: RunContext, message: str, ui_message: str | None = None) -> ModuleResult:
    """Shorthand: ``ModuleResult(ok=False)`` with an error view already attached."""
    return ModuleResult(
        ok=False,
        message=message,
        data=_view(ctx, "error", ui_message if ui_message is not None else message),
    )


def _render_current_viewer(ctx: RunContext) -> str | None:
    """Return a ``/api/viewer`` URL for the current page and kick off prefetch workers.

    ``v=`` carries the overlay signature so the browser cache-busts on overlay change.
    """
    if not ctx.dataset.imgs_paths:
        return None
    img_idx = ctx.navigation.current_image_idx
    page_idx = ctx.navigation.current_page_idx
    image_path = ctx.dataset.imgs_paths[img_idx]
    page_output = ctx.output.page_outputs.get(page_idx) or {}

    num_pages = ctx.dataset.expected_num_pages
    viewer_cache.prefetch_image_pages(image_path, img_idx, num_pages)
    next_idx = img_idx + 1
    if next_idx < len(ctx.dataset.imgs_paths):
        viewer_cache.prefetch_image_pages(ctx.dataset.imgs_paths[next_idx], next_idx, num_pages)

    sig = viewer_cache.overlay_sig(page_output)
    return f"/api/viewer?img={img_idx}&page={page_idx}&v={sig}"


def _csv_name(ctx: RunContext) -> str:
    """Target CSV filename for the current prompt/mode/pages selection."""
    return results.build_output_csv_name(
        ctx.prompt.text, ctx.prompt.mode, ctx.setup.pages_to_label,
    )


def _labelled_filenames(ctx: RunContext) -> set[str]:
    """``fileName`` values already in the target CSV, or empty set if not derivable yet."""
    if not ctx.setup.locked or not ctx.prompt.text.strip():
        return set()
    return results.load_labelled_filenames(ctx.dataset.dataset_root, _csv_name(ctx))


def _current_subfolder(ctx: RunContext) -> str:
    """Relative posix path of the current image's parent (``"."`` for root, ``""`` if empty)."""
    if not ctx.dataset.imgs_paths or not ctx.dataset.dataset_root:
        return ""
    current = Path(ctx.dataset.imgs_paths[ctx.navigation.current_image_idx])
    rel = current.parent.relative_to(ctx.dataset.dataset_root).as_posix()
    return rel if rel else "."


def _rel_name(image_path: str, dataset_root: str) -> str:
    """Posix-style relative path used as the CSV ``fileName`` key."""
    return Path(os.path.relpath(image_path, dataset_root)).as_posix()


def _bulk_run(ctx: RunContext, image_paths: list[str]) -> Iterator[ModuleResult]:
    """Run inference for every un-labelled image in ``image_paths``, yielding per-image views.

    Images whose relative name is already in the target CSV are skipped (resume).
    Each row is saved immediately so partial progress survives a disconnect.
    Stops at the first inference/save error after yielding a ``_fail`` view.
    """
    labelled = _labelled_filenames(ctx)
    work = [p for p in image_paths if _rel_name(p, ctx.dataset.dataset_root) not in labelled]
    total = len(work)
    if total == 0:
        return

    num_pages = ctx.dataset.expected_num_pages
    for i, image_path in enumerate(work, start=1):
        page_outputs: list[dict | None] = [None] * num_pages
        has_any_mask = False
        for page_idx in ctx.setup.pages_to_label:
            r = inference.run_page(
                image_path, page_idx, ctx.prompt.text, ctx.prompt.mode, ctx.prompt.threshold,
            )
            if not r.ok:
                yield _fail(ctx, r.message)
                return
            output, _ = r.data
            page_outputs[page_idx] = output
            if output["masks"]:
                has_any_mask = True

        status_label = TASK_INCORRECT if has_any_mask else TASK_DISCARD
        r = results.build_row(
            image_path=image_path,
            num_pages=num_pages,
            page_outputs=page_outputs,
            status_label=status_label,
            header=ctx.dataset.header,
            dataset_root=ctx.dataset.dataset_root,
        )
        if not r.ok:
            yield _fail(ctx, r.message)
            return
        r = results.save_rows([r.data], _save_metadata(ctx))
        if not r.ok:
            yield _fail(ctx, r.message)
            return
        ctx.output.last_save_report = r.data
        yield ModuleResult(
            ok=True,
            data=_view(ctx, "info", f"Labelled {Path(image_path).name} ({i}/{total})."),
        )


def _finish_run(ctx: RunContext) -> ModuleResult:
    """Reset the context and switch to the finished card with row count + CSV path."""
    dataset_root = ctx.dataset.dataset_root
    csv_name = _csv_name(ctx)
    csv_path = str(Path(dataset_root) / csv_name) if dataset_root else ""
    count = len(results.load_labelled_filenames(dataset_root, csv_name)) if dataset_root else 0
    run_context.reset_context()
    ctx = run_context.get_context()
    ctx.finished = True
    ctx.finished_message = f"Labelled {count} image(s). Saved to {csv_path}."
    return ModuleResult(ok=True, data=_view(ctx, "ok", "Labelling finished."))


def _drain(stream: Iterator[ModuleResult]) -> ModuleResult:
    """Consume a streaming use-case and return its last yielded result (no-JS fallback)."""
    last: ModuleResult | None = None
    for r in stream:
        last = r
    if last is not None:
        return last
    return ModuleResult(
        ok=True, data=_view(run_context.get_context(), "info", "Nothing to label."),
    )


def _next_subfolder_first(ctx: RunContext, current_subfolder: Path) -> int | None:
    """Return the index of the first image after the current pointer whose parent differs."""
    start = ctx.navigation.current_image_idx
    for i in range(start, len(ctx.dataset.imgs_paths)):
        if Path(ctx.dataset.imgs_paths[i]).parent != current_subfolder:
            return i
    return None


def _run_and_cache(ctx: RunContext, pages: list[int]) -> str | None:
    """Run inference on the current image for ``pages`` and cache outputs/states.

    Returns ``None`` on success or an error message string on the first failure.
    """
    image_path = ctx.dataset.imgs_paths[ctx.navigation.current_image_idx]
    for page_idx in pages:
        r = inference.run_page(
            image_path, page_idx, ctx.prompt.text, ctx.prompt.mode, ctx.prompt.threshold,
            cached_state=ctx.output.sam_states.get(page_idx),
        )
        if not r.ok:
            return r.message
        output, state_cpu = r.data
        ctx.output.page_outputs[page_idx] = output
        ctx.output.sam_states[page_idx] = state_cpu
    return None
