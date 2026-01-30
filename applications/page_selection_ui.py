import tkinter as tk
from math import ceil

# ---------------------------------------------------------
# Page Selection UI Class
# ---------------------------------------------------------
class PageSelectionUI:
    def __init__(self, max_pages=10):
        """
        Creates a Tkinter popup that allows the user to select
        which pages of a TIFF file to label.

        :param max_pages (int): Maximum number of pages to display for selection
        """
        # Initialize variables
        self.max_pages = max_pages
        self.selected_pages = []
        self.all_selected = False
        self.default_btn_bg: str

        # Create root
        self.root = tk.Tk()
        self.root.title("Select Pages to Label")
        self.root.geometry("400x400")
        self.root.minsize(350, 300)

        # Configure grid (3 rows x 2 columns)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Build UI
        self._page_selection_frame()
        self._error_frame()
        self._action_buttons_frame()

        # Start loop
        self.root.mainloop()


    # ---------------------------------------------------------
    # ROW 0, COL 0-1: Scrollable Page Selection
    # ---------------------------------------------------------
    def _page_selection_frame(self):
        """
        Creates a frame containing page selection checkboxes
        arranged in two columns and centered horizontally.
        """
        # Outer container (fills available space)
        container = tk.Frame(self.root)
        container.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        # Inner frame (actual centered content)
        content = tk.Frame(container)
        content.grid(row=0, column=0)

        # Create BooleanVars
        self.page_vars = [tk.BooleanVar(value=False) for _ in range(self.max_pages)]

        # Create checkboxes in 2 columns
        rows_per_col = ceil(self.max_pages / 2)

        for i in range(self.max_pages):
            col = 0 if i < rows_per_col else 1
            row = i if i < rows_per_col else i - rows_per_col

            tk.Checkbutton(
                content,
                text=f"Page {i + 1}",
                variable=self.page_vars[i],
                font=("Arial", 12)
            ).grid(row=row, column=col, padx=20, pady=5, sticky="w")


    # ---------------------------------------------------------
    # ROW 1, COL 0-1: Error Message
    # ---------------------------------------------------------
    def _error_frame(self):
        """
        Creates an error message label (hidden by default).
        """
        self.error_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 12, "bold"),
            fg="red"
        )
        self.error_label.grid(row=1, column=0, columnspan=2, pady=(0, 5))


    # ---------------------------------------------------------
    # ROW 2, COL 0-1: Action Buttons
    # ---------------------------------------------------------
    def _action_buttons_frame(self):
        """
        Creates Select All toggle button and Submit button.
        """
        # Select All button (COL 0)
        self.select_all_btn = tk.Button(
            self.root,
            text="Select All",
            command=self._toggle_select_all,
            font=("Arial", 12),
            relief="raised"
        )
        self.select_all_btn.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.default_btn_bg = self.select_all_btn.cget("bg")

        # Submit button (COL 1)
        submit_btn = tk.Button(
            self.root,
            text="Submit",
            command=self._on_submit,
            font=("Arial", 12)
        )
        submit_btn.grid(row=2, column=1, padx=10, pady=10, sticky="e")


    # ------------------------------------------------------------------------------------------------ #
    #                                      Button helper functions                                     #
    # ------------------------------------------------------------------------------------------------ #
    def _toggle_select_all(self):
        """
        Toggle button to select or deselect all pages.
        Updates button appearance based on state.
        """
        self.all_selected = not self.all_selected

        for var in self.page_vars:
            var.set(self.all_selected)

        if self.all_selected:
            self.select_all_btn.config(relief="sunken", bg="#d0f0d0")
        else:
            self.select_all_btn.config(relief="raised", bg=self.default_btn_bg)
    
    # ---------------------------------------------------------
    def _on_submit(self):
        """
        Retrieves the indices of selected pages and closes the UI.

        If no pages are selected, displays an error message
        without closing the window.
        """
        self.selected_pages = [i for i, var in enumerate(self.page_vars) if var.get()]

        if not self.selected_pages:
            self.error_label.config(text="Please select at least one page!")
            return

        self.root.destroy()
