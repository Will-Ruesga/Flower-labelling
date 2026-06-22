"""Cross-module data types. No behavior — only dataclasses and type aliases."""

from dataclasses import dataclass, field
from typing import Any


Row = dict[str, Any]


@dataclass
class ModuleResult:
    """Envelope returned by every module-public function.

    On failure, ``data`` often carries a ``RunViewState`` so the UI can render
    the error state without recomputing.
    """

    ok: bool
    message: str = ""
    data: Any = None


@dataclass
class DatasetManifest:
    """Resolved image list + root directory for a single run."""

    imgs_paths: list[str]
    dataset_root: str


@dataclass
class SaveReport:
    """Outcome of a CSV save."""

    csv_path: str
    rows_saved: int


@dataclass
class SetupState:
    """User-facing setup fields. Frozen while ``locked`` is True."""

    source_type: str = ""
    source_path: str = ""
    pages_to_label: list[int] = field(default_factory=list)
    probed: bool = False
    locked: bool = False


@dataclass
class DatasetState:
    """Resolved dataset info, populated by ``probe_source`` / ``confirm_setup``."""

    imgs_paths: list[str] = field(default_factory=list)
    dataset_root: str = ""
    expected_num_pages: int = 0
    header: list[str] = field(default_factory=list)
    has_subdirs: bool = False


@dataclass
class NavigationState:
    """Pointer into the dataset and into the current image's pages."""

    current_image_idx: int = 0
    current_page_idx: int = 0


@dataclass
class PromptState:
    """SAM3 prompt + mode + confidence threshold currently selected by the user."""

    text: str = ""
    mode: str = "single"
    threshold: float = 0.5


@dataclass
class OutputState:
    """Per-page inference cache for the current image + pending-save queue.

    ``page_outputs`` and ``sam_states`` only hold entries for the *current*
    image; both are wiped on image advance so the next image runs fresh.
    """

    page_outputs: dict[int, dict] = field(default_factory=dict)
    sam_states: dict[int, dict] = field(default_factory=dict)
    pending_rows: list[Row] = field(default_factory=list)
    last_save_report: SaveReport | None = None


@dataclass
class RunContext:
    """Top-level run state; owned by ``run_context``, mutated only by ``application``."""

    setup: SetupState = field(default_factory=SetupState)
    dataset: DatasetState = field(default_factory=DatasetState)
    navigation: NavigationState = field(default_factory=NavigationState)
    prompt: PromptState = field(default_factory=PromptState)
    output: OutputState = field(default_factory=OutputState)
    finished: bool = False
    finished_message: str = ""


@dataclass
class RunViewState:
    """UI-ready snapshot returned by every ``application`` use-case."""

    setup: SetupState
    prompt: PromptState
    navigation: NavigationState
    dataset_size: int
    expected_num_pages: int
    labelled_pages: list[int]
    viewer_image: str | None
    status: str
    message: str
    has_subdirs: bool = False
    current_subfolder: str = ""
    labelled_total_count: int = 0
    finished: bool = False
    finished_message: str = ""
