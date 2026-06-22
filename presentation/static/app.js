(function () {
  "use strict";

  // Bulk actions are streamed via /api/bulk (Server-Sent Events) so the
  // progress bar / status pill / viewer update live as each image completes.
  var BULK_ACTIONS = { "label_dataset": true, "label_subdirectory": true };

  // --- AJAX submit interceptor. Posts forms to /api/action and applies the
  //     returned RunViewState in-place. Bulk actions are routed to /api/bulk
  //     instead, which streams events back. Falls back to a full reload when
  //     the server marks the action as a full-reload transition or the fetch
  //     fails.
  document.addEventListener("submit", function (e) {
    var form = e.target;
    if (form.hasAttribute("data-no-ajax")) return;
    e.preventDefault();
    var body = new FormData(form);
    var action = body.get("action");
    if (BULK_ACTIONS[action]) {
      runBulkStream(action);
      return;
    }
    document.body.classList.add("is-submitting");
    fetch("/api/action", { method: "POST", body: body })
      .then(function (r) { return r.json(); })
      .then(applyResponse)
      .catch(function () { location.reload(); })
      .finally(function () { document.body.classList.remove("is-submitting"); });
  });

  function runBulkStream(action) {
    document.body.classList.add("is-submitting");
    var src = new EventSource("/api/bulk?action=" + encodeURIComponent(action));
    var done = false;
    src.onmessage = function (ev) {
      applyResponse(JSON.parse(ev.data));
    };
    src.addEventListener("done", function (ev) {
      var data = JSON.parse(ev.data);
      done = true;
      src.close();
      document.body.classList.remove("is-submitting");
      if (data.full_reload) location.reload();
    });
    src.onerror = function () {
      if (done) return;
      src.close();
      document.body.classList.remove("is-submitting");
      location.reload();
    };
  }

  function applyResponse(resp) {
    if (resp.full_reload || !resp.view) {
      location.reload();
      return;
    }
    var view = resp.view;
    setStatus(view, resp.message);
    setViewer(view);
    setMeta(view);
    setPagePills(view);
    setPromptGate(view);
    setThresholdLabel(view);
    setModeSegment(view);
    setCurrentSubfolder(view);
    setProgress(view);
  }

  function setStatus(view, message) {
    var pill = document.querySelector("[data-status-pill]");
    if (!pill) return;
    var hasPrompt = view.prompt && view.prompt.text && view.prompt.text.trim();
    var text, kind;
    if (message) {
      text = message;
      kind = view.status || "info";
    } else if (view.setup.locked && !hasPrompt) {
      text = "Insert prompt to label.";
      kind = "info";
    } else {
      text = "Ready.";
      kind = "info";
    }
    pill.textContent = text;
    pill.className = "status status--" + kind + " labelling-header__status";
  }

  function setViewer(view) {
    var img = document.querySelector("[data-viewer-img]");
    if (img && view.viewer_image) img.src = view.viewer_image;
  }

  function setMeta(view) {
    var meta = document.querySelector("[data-meta]");
    if (!meta) return;
    var text;
    if (view.finished) {
      text = "Finished.";
    } else if (view.dataset_size > 0) {
      text = "Image " + (view.navigation.current_image_idx + 1)
           + " / " + view.dataset_size
           + " · Page " + view.navigation.current_page_idx;
    } else {
      text = "No dataset loaded.";
    }
    meta.textContent = text;
  }

  function setPagePills(view) {
    var current = view.navigation.current_page_idx;
    var labelled = view.labelled_pages || [];
    document.querySelectorAll("[data-page-pill]").forEach(function (pill) {
      var p = parseInt(pill.getAttribute("data-page-pill"), 10);
      var classes = ["page-pill"];
      if (p === current) classes.push("page-pill--current");
      if (labelled.indexOf(p) !== -1 && p !== current) classes.push("page-pill--labelled");
      pill.className = classes.join(" ");
      if (p === current) {
        pill.setAttribute("aria-current", "page");
      } else {
        pill.removeAttribute("aria-current");
      }
    });
  }

  function setPromptGate(view) {
    var hasPrompt = view.prompt && view.prompt.text && view.prompt.text.trim();
    document.querySelectorAll("[data-prompt-gate]").forEach(function (btn) {
      btn.disabled = !hasPrompt;
      if (!hasPrompt) {
        btn.setAttribute("title", "Insert a prompt first");
      } else {
        btn.removeAttribute("title");
      }
    });
  }

  function setThresholdLabel(view) {
    var label = document.querySelector("[data-threshold-label]");
    if (!label) return;
    label.textContent = "Confidence — " + view.prompt.threshold.toFixed(2);
  }

  function setModeSegment(view) {
    document.querySelectorAll("[data-mode-segment]").forEach(function (el) {
      var match = el.getAttribute("data-mode-value") === view.prompt.mode;
      el.classList.toggle("is-active", match);
      var radio = el.querySelector("input[type='radio']");
      if (radio) radio.checked = match;
    });
  }

  function setCurrentSubfolder(view) {
    var el = document.querySelector("[data-current-subfolder]");
    if (!el) return;
    var sub = view.current_subfolder || "";
    el.textContent = sub && sub !== "." ? sub + "/" : sub;
  }

  function setProgress(view) {
    var bar = document.querySelector("[data-progress-bar]");
    var caption = document.querySelector("[data-progress-caption]");
    if (!bar || !caption) return;
    bar.max = view.dataset_size;
    bar.value = view.labelled_total_count;
    caption.textContent = "Labelled " + view.labelled_total_count + " / " + view.dataset_size;
  }

  // --- Auto-submit forms tagged with data-autosubmit. requestSubmit() fires a
  //     bubbling submit event so the AJAX interceptor above picks it up.
  document.querySelectorAll("form[data-autosubmit]").forEach(function (form) {
    form.addEventListener("change", function () { form.requestSubmit(); });
  });

  // --- Viewer brightness/contrast sliders (client-side CSS filter only). ---
  //     Values persist in localStorage; they never leave the browser and never
  //     affect what the server sees. Sliders write CSS variables on <html>;
  //     the .viewer-frame img rule picks them up via filter:...
  var VIEW_DEFAULTS = { brightness: 1, contrast: 1 };
  var viewAdjustments = {
    brightness: parseFloat(localStorage.getItem("viewer-brightness") || VIEW_DEFAULTS.brightness),
    contrast:   parseFloat(localStorage.getItem("viewer-contrast")   || VIEW_DEFAULTS.contrast),
  };

  function applyViewAdjustments() {
    document.documentElement.style.setProperty("--viewer-brightness", viewAdjustments.brightness);
    document.documentElement.style.setProperty("--viewer-contrast",   viewAdjustments.contrast);
  }

  applyViewAdjustments();

  document.querySelectorAll("[data-view-adjust]").forEach(function (input) {
    var key = input.getAttribute("data-view-adjust");   // "brightness" | "contrast"
    input.value = viewAdjustments[key];
    input.addEventListener("input", function () {
      viewAdjustments[key] = parseFloat(input.value);
      localStorage.setItem("viewer-" + key, input.value);
      applyViewAdjustments();
    });
  });

  var viewResetBtn = document.querySelector("[data-view-reset]");
  if (viewResetBtn) {
    viewResetBtn.addEventListener("click", function () {
      viewAdjustments.brightness = VIEW_DEFAULTS.brightness;
      viewAdjustments.contrast   = VIEW_DEFAULTS.contrast;
      localStorage.removeItem("viewer-brightness");
      localStorage.removeItem("viewer-contrast");
      document.querySelectorAll("[data-view-adjust]").forEach(function (input) {
        input.value = viewAdjustments[input.getAttribute("data-view-adjust")];
      });
      applyViewAdjustments();
    });
  }

  // --- Pages form: All / None shortcuts. ---
  var pagesForm = document.getElementById("pages-form");
  if (pagesForm) {
    var allBtn = pagesForm.querySelector("[data-pages-all]");
    var noneBtn = pagesForm.querySelector("[data-pages-none]");
    var boxes = pagesForm.querySelectorAll("input[type='checkbox'][name='pages']");
    if (allBtn) {
      allBtn.addEventListener("click", function () {
        boxes.forEach(function (b) { b.checked = true; });
      });
    }
    if (noneBtn) {
      noneBtn.addEventListener("click", function () {
        boxes.forEach(function (b) { b.checked = false; });
      });
    }
  }

  // Browse dialog lives in browse.js (loaded as a separate <script> after this file).
})();
