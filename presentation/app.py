"""Flask entrypoint — single route, action dispatch, template rendering."""

import json
from dataclasses import asdict
from pathlib import Path

from flask import (
    Flask, Response, abort, jsonify, make_response, render_template, request,
    stream_with_context,
)

import application
from shared import ModuleResult, RunViewState


app = Flask(__name__)


# Actions that change page-level structure and need a full reload to re-render.
_FULL_RELOAD_ACTIONS = {"probe_source", "confirm_setup", "reset_run", "label_dataset"}


def _view_to_dict(view: RunViewState | None) -> dict | None:
    """Serialize a RunViewState to a JSON-safe dict (transport concern)."""
    if view is None:
        return None
    return asdict(view)


@app.route("/", methods=["GET", "POST"])
def index():
    """Serve the unified page (GET) and dispatch form actions (POST)."""
    if request.method == "GET":
        result = application.start_run()
    else:
        action = request.form.get("action", "")
        handler = _ACTIONS.get(action)
        if handler is None:
            base = application.start_run()
            result = ModuleResult(ok=False, message=f"unknown action: {action}", data=base.data)
        else:
            result = handler()
    return render_template(
        "index.html",
        view=result.data,
        ok=result.ok,
        message=result.message,
    )


def _probe_source() -> ModuleResult:
    source_type = request.form.get("source_type", "")
    source_path = request.form.get("source_path", "")
    return application.probe_source(source_type, source_path)


def _confirm_setup() -> ModuleResult:
    pages = [int(p) for p in request.form.getlist("pages")]
    return application.confirm_setup(pages)


def _reset_run() -> ModuleResult:
    return application.reset_run()


def _set_prompt() -> ModuleResult:
    return application.set_prompt(request.form.get("text", ""))


def _set_generation_mode() -> ModuleResult:
    return application.set_generation_mode(request.form.get("mode", ""))


def _set_confidence_threshold() -> ModuleResult:
    return application.set_confidence_threshold(float(request.form.get("threshold", "0.5")))


def _navigate_page() -> ModuleResult:
    return application.navigate_page(int(request.form.get("step", "0")))


def _jump_page() -> ModuleResult:
    return application.jump_page(int(request.form.get("page", "0")))


def _label_current_page() -> ModuleResult:
    return application.label_current_page()


def _label_current_image() -> ModuleResult:
    return application.label_current_image()


def _label_dataset() -> ModuleResult:
    return application.label_dataset()


def _label_subdirectory() -> ModuleResult:
    return application.label_subdirectory()


def _clear_current_page() -> ModuleResult:
    return application.clear_current_page()


def _clear_current_image() -> ModuleResult:
    return application.clear_current_image()


def _submit_decision() -> ModuleResult:
    return application.submit_decision(request.form.get("status", ""))


@app.route("/api/action", methods=["POST"])
def api_action():
    """Dispatch a form action and return the resulting view as JSON."""
    action = request.form.get("action", "")
    handler = _ACTIONS.get(action)
    if handler is None:
        return jsonify({"ok": False, "message": f"unknown action: {action}"}), 400
    result = handler()
    # Reload only on success — a failed action keeps the user on the same screen
    # so the error message is visible rather than wiped by a redirect.
    finished = bool(result.data is not None and getattr(result.data, "finished", False))
    full_reload = result.ok and (action in _FULL_RELOAD_ACTIONS or finished)
    return jsonify({
        "ok": result.ok,
        "message": result.message,
        "view": _view_to_dict(result.data),
        "full_reload": full_reload,
    })


_BULK_STREAMS = {
    "label_dataset": application.label_dataset_stream,
    "label_subdirectory": application.label_subdirectory_stream,
}


@app.route("/api/bulk")
def api_bulk():
    """Server-Sent Events feed for a bulk labelling action.

    Streams one ``data:`` event per labelled image (carrying the updated
    ``RunViewState``), followed by a single ``event: done`` event that tells
    the client to drop the EventSource and, if ``full_reload`` is true, reload
    the page (used when the run is finished).
    """
    action = request.args.get("action", "")
    stream_fn = _BULK_STREAMS.get(action)
    if stream_fn is None:
        return jsonify({"ok": False, "message": f"unknown bulk action: {action}"}), 400

    def generate():
        last_result = None
        for result in stream_fn():
            last_result = result
            event = {
                "ok": result.ok,
                "message": result.message,
                "view": _view_to_dict(result.data),
                "full_reload": False,
            }
            yield f"data: {json.dumps(event)}\n\n"
        finished = bool(
            last_result and last_result.data and getattr(last_result.data, "finished", False)
        )
        done = {"done": True, "full_reload": finished}
        yield f"event: done\ndata: {json.dumps(done)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/viewer")
def api_viewer():
    """Serve one page's PNG. Params ``img``, ``page`` (ints); ``v`` is cache-bust only."""
    r = application.get_viewer_png(
        int(request.args["img"]),
        int(request.args["page"]),
    )
    if not r.ok:
        abort(404)
    resp = make_response(r.data)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@app.route("/api/browse", methods=["GET"])
def api_browse():
    """List directories (and, in csv mode, .csv files) at ``path``; JSON response."""
    mode = request.args.get("mode", "folder")
    raw = request.args.get("path", "").strip()
    start = Path(raw).expanduser() if raw else Path.home()
    start = start.resolve()
    if not start.is_dir():
        start = start.parent if start.parent.is_dir() else Path.home().resolve()

    dirs, files = [], []
    for entry in sorted(start.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            dirs.append({"name": entry.name, "path": str(entry)})
        elif mode == "csv" and entry.is_file() and entry.suffix.lower() == ".csv":
            files.append({"name": entry.name, "path": str(entry)})

    crumbs = []
    acc = Path(start.anchor or "/")
    crumbs.append({"label": str(acc), "path": str(acc)})
    for part in start.relative_to(acc).parts:
        acc = acc / part
        crumbs.append({"label": part, "path": str(acc)})

    parent = str(start.parent) if start != start.parent else None

    return jsonify({
        "path": str(start),
        "parent": parent,
        "crumbs": crumbs,
        "dirs": dirs,
        "files": files,
    })


_ACTIONS = {
    "probe_source": _probe_source,
    "confirm_setup": _confirm_setup,
    "reset_run": _reset_run,
    "set_prompt": _set_prompt,
    "set_generation_mode": _set_generation_mode,
    "set_confidence_threshold": _set_confidence_threshold,
    "navigate_page": _navigate_page,
    "jump_page": _jump_page,
    "label_current_page": _label_current_page,
    "label_current_image": _label_current_image,
    "label_dataset": _label_dataset,
    "label_subdirectory": _label_subdirectory,
    "clear_current_page": _clear_current_page,
    "clear_current_image": _clear_current_image,
    "submit_decision": _submit_decision,
}
