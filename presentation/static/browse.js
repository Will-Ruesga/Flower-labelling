// Browse dialog — path/file picker for the setup-stage-1 source form.
// Loaded as a separate <script> after app.js. Self-exits when the dialog or
// its triggers aren't on the page (i.e. anywhere past setup stage 1).
(function () {
  "use strict";

  var dialog = document.getElementById("browse-dialog");
  var pathInput = document.getElementById("source_path");
  var openBtn = document.querySelector("[data-browse-open]");
  if (!dialog || !pathInput || !openBtn) return;

  var listEl = document.getElementById("browse-list");
  var crumbsEl = document.getElementById("browse-crumbs");
  var useBtn = document.getElementById("browse-use");
  var hintEl = document.getElementById("browse-mode-hint");
  var typeInputs = document.querySelectorAll("input[name='source_type']");

  var currentPath = null;
  var currentMode = "folder";
  var pickedFile = null;

  function currentSourceType() {
    for (var i = 0; i < typeInputs.length; i++) {
      if (typeInputs[i].checked) return typeInputs[i].value;
    }
    return "folder";
  }

  function openDialog() {
    currentMode = currentSourceType();
    pickedFile = null;
    hintEl.textContent = currentMode === "csv"
      ? "Click a .csv to use it, or navigate into folders."
      : "Navigate to the folder with your images.";
    useBtn.textContent = currentMode === "csv" ? "Use selected file" : "Use this folder";
    useBtn.disabled = currentMode === "csv";
    var start = pathInput.value || "";
    fetchAndRender(start);
    dialog.hidden = false;
  }

  function closeDialog() { dialog.hidden = true; }

  function fetchAndRender(path) {
    var url = "/api/browse?mode=" + encodeURIComponent(currentMode)
            + (path ? "&path=" + encodeURIComponent(path) : "");
    fetch(url).then(function (r) { return r.json(); }).then(function (data) {
      if (data.error) {
        listEl.innerHTML = "<div class='browser__empty'>" + data.error + "</div>";
        return;
      }
      currentPath = data.path;
      pickedFile = null;
      useBtn.disabled = currentMode === "csv";
      renderCrumbs(data);
      renderList(data);
    });
  }

  function renderCrumbs(data) {
    crumbsEl.innerHTML = "";
    var parts = data.crumbs || [];
    parts.forEach(function (c, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn--sm btn--soft";
      btn.textContent = c.label;
      btn.addEventListener("click", function () { fetchAndRender(c.path); });
      crumbsEl.appendChild(btn);
      if (i < parts.length - 1) {
        var sep = document.createElement("span");
        sep.textContent = "/";
        sep.className = "t6";
        crumbsEl.appendChild(sep);
      }
    });
  }

  function renderList(data) {
    listEl.innerHTML = "";
    if (data.parent) {
      listEl.appendChild(makeRow("..", data.parent, "dir"));
    }
    (data.dirs || []).forEach(function (d) {
      listEl.appendChild(makeRow(d.name, d.path, "dir"));
    });
    (data.files || []).forEach(function (f) {
      listEl.appendChild(makeRow(f.name, f.path, "file"));
    });
    if (!data.dirs.length && !data.files.length && !data.parent) {
      var empty = document.createElement("div");
      empty.className = "browser__empty";
      empty.textContent = "Empty.";
      listEl.appendChild(empty);
    }
  }

  function makeRow(label, path, kind) {
    var row = document.createElement("button");
    row.type = "button";
    row.className = "browser__row browser__row--" + kind;
    row.textContent = " " + label;
    if (kind === "dir") {
      row.addEventListener("click", function () { fetchAndRender(path); });
    } else {
      row.addEventListener("click", function () {
        pickedFile = path;
        Array.prototype.forEach.call(listEl.querySelectorAll(".browser__row"), function (r) {
          r.style.background = "";
        });
        row.style.background = "var(--accent-soft)";
        useBtn.disabled = false;
      });
    }
    return row;
  }

  function commit() {
    var chosen = currentMode === "csv" ? pickedFile : currentPath;
    if (!chosen) return;
    pathInput.value = chosen;
    closeDialog();
  }

  openBtn.addEventListener("click", openDialog);
  document.querySelectorAll("[data-browse-close]").forEach(function (el) {
    el.addEventListener("click", closeDialog);
  });
  useBtn.addEventListener("click", commit);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !dialog.hidden) closeDialog();
  });
})();
