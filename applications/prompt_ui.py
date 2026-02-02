import tkinter as tk

from PIL import Image, ImageTk
from tkinter import ttk
from pathlib import Path
from typing import Dict, Optional, Any

from model_manager import ModelManager

# Defines
BTN_WIDTH = 12
BTN_HEIGHT = 1


# ---------------------------------------------------------
# Prompt Class UI
# ---------------------------------------------------------
class PromptUI:
    def __init__(self, processor, imgs_paths: list, header: list[str], pages_to_label: list[int] = [0]):
        """
        :param processor: The model processor
        :param imgs_paths: List of absolute paths of all the iamges
        :param pages_to_label: List of ints with the pages to label set at [0] if no pages
        :param header: List of strings with the header columns of the csv
        """
        # -------------------------------------------------
        # Navigation
        self.current_img_index = 0
        self.current_page_index: int = 0
        
        # -------------------------------------------------
        # Image / page data
        self.imgs_paths = imgs_paths                        # Paths of all the imges
        self.pages_to_label = pages_to_label                # Pages to label
        self.valid_pages_to_label: list[int] = []           # Valid pages to label
        self.current_image: Image.Image | None = None       # PIL Image container
        self.page_outputs: Dict[int, Optional[Any]] = {}    # {page_idx: model_output}

        # -------------------------------------------------
        # Rendering
        self.canvas_img_id = None
        self.tk_img = None

        self.orig_w: Optional[int] = None
        self.orig_h: Optional[int] = None
        self.disp_sx: Optional[float] = None
        self.disp_sy: Optional[float] = None

        # -------------------------------------------------
        # Model
        self.model_manager = ModelManager(processor)

        # -------------------------------------------------
        # Interaction
        self.generation_mode = None # "single" | "multiple"
        self.bbox_enabled = False
        self.bbox = None
        self.bbox_start = None

        # -------------------------------------------------
        # Saving
        self.header = header
        self.expected_num_pages: Optional[int] = None

        # -------------------------------------------------
        # UI widgets
        self.img_canvas: Optional[tk.Canvas] = None
        self.page_info_label: Optional[tk.Label] = None
        self.error_label: Optional[tk.Label] = None
        self.single_mask_tgl: Optional[tk.Button] = None
        self.multiple_mask_tgl: Optional[tk.Button] = None
        self.bbox_btn: Optional[tk.Button] = None

        self.img_frame = None
        self.prompt_text = None
        self.progress = None

        # UI (root)
        self.root = tk.Tk()
        self.root.geometry("1200x1100")
        self.root.title("Prompt Labelling")
        self.root.minsize(1200, 1000)

        # Button configuration
        self.common_kwargs = {
            "font": ("Arial", 14),
            "width": BTN_WIDTH,
            "height": BTN_HEIGHT,
            "relief": "solid",
            "bd": 2,
        }

        self.common_kwargs_small = {
            "font": ("Arial", 8),
            "width": 2,
            "height": BTN_HEIGHT,
            "relief": "solid",
            "bd": 2,
        }
        # Button background
        self.default_btn_bg = tk.Button(self.root).cget("bg")

        # -------------------------------------------------
        # UI Layout
        # Make all 4 columns equal width
        for c in range(3):
            self.root.columnconfigure(c, weight=1, minsize=250)

        # Make image display rows large, control rows smaller
        # Rows 0–7 = big image section
        for r in range(0, 8):
            self.root.rowconfigure(r, weight=6)

        # Rows 8–10 = control panel
        for r in range(8, 11):
            self.root.rowconfigure(r, weight=2)

        # Error row (row 11)
        self.root.rowconfigure(11, weight=1)

        # --- UI Widgets --- #
        # Create image display frame and show first image
        self._build_image_display()

        # Add frames to the UI
        self._build_status_bar()
        self._prompt_text_box()
        self._mask_input_buttons()
        self._mask_generate_buttons()
        self._decision_buttons()
        
        # Show first valid image/page
        self._load_image(self.current_img_index)

        # Start the loop
        self.root.mainloop()


    # ---------------------------------------------------------
    # ROW 0-7, COL 0-3: Image Display
    # ---------------------------------------------------------
    def _build_image_display(self):
        """
        Creates a frame to display the current image and its mask overlay.
        """
        self.img_frame = tk.Frame(self.root, bg="black")
        self.img_frame.grid(row=0, column=0, rowspan=8, columnspan=3, sticky="nsew")

        self.img_canvas = tk.Canvas(self.img_frame, bg="black")
        self.img_canvas.pack(expand=True, fill="both")

        self.img_canvas.bind("<ButtonPress-1>", self._on_bbox_start)
        self.img_canvas.bind("<B1-Motion>", self._on_bbox_drag)
        self.img_canvas.bind("<ButtonRelease-1>", self._on_bbox_end)

    # ---------------------------------------------------------
    def _load_image(self, image_index: int):
        """
        Loads and displays the image at the given index without overlay.

        :param index: Index of the image in self.imgs_paths
        """
        # Check out of bounds (last image)
        if image_index >= len(self.imgs_paths):
            self.root.destroy()
            return

        # Get path and set image container
        img_path = self.imgs_paths[image_index]
        self.current_image = Image.open(img_path)

        # Get valid pages if no valid pages -> skip image
        valid_pages, num_pages = self._prepare_image_pages(self.current_image)
        if self.expected_num_pages is None:
            self.expected_num_pages = num_pages
            self._extend_header_for_pages(num_pages)
        elif num_pages != self.expected_num_pages:
            print(
                f"[WARNING]: Image {img_path} has {num_pages} instead of {self.expected_num_pages} "
                "it might correspond toa different dataset."
            )
            self._load_image(image_index + 1)
            return
        self.num_pages = num_pages
        if not valid_pages:
            self._load_image(image_index + 1)
            return
        
        # Update class values
        self.current_page_index = valid_pages[0]
        self.valid_pages_to_label = valid_pages
        self.current_img_index = image_index

        # Init the output dict 
        self._init_page_outputs(self.valid_pages_to_label)

        
        # Show page 0
        self._show_page()


    # ---------------------------------------------------------
    # ROW 8-9, COL 0: Text Box Prompt
    # ---------------------------------------------------------
    def _prompt_text_box(self):
        """
        Generates a text box where the user can write a small prompt for the model
        """
        # Frame
        prompt_frame = tk.Frame(self.root)
        prompt_frame.grid(row=8, column=0, rowspan=1, columnspan=1, padx=5, pady=5, sticky="nsew")
        prompt_frame.grid_columnconfigure(0, weight=1)

        # Label
        promt_text_label = tk.Label(prompt_frame, text="Enter model prompt:", font=("Arial", 12))
        promt_text_label.grid(row=0, column=0, padx=10, pady=(0, 5), sticky="nw")

        # Page info label
        self.page_info_label = tk.Label(prompt_frame, text="", font=("Arial", 10, "italic"), fg="gray")
        self.page_info_label.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="nw")

        # Inject navigation buttons NEXT to label
        self._build_page_nav_buttons(prompt_frame)

        # Text box 
        self.prompt_text = tk.Text(prompt_frame, height=5,font=("Arial", 14),wrap="word",)
        self.prompt_text.grid(row=2, column=0, columnspan=3,  padx=10, sticky="ew")


    # ---------------------------------------------------------  
    def _build_page_nav_buttons(self, frame: tk.Frame):
        """
        Creates Previous / Next page navigation buttons
        Positioned in the SAME row as the prompt, NEXT column
        """

        # Configure grid in frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        # --- Generate Page Navigation Buttons --- #
        # Previous Page Button
        prev_btn = tk.Button(
            frame, 
            text="<", 
            command=lambda: self._on_change_page(-1), 
            **self.common_kwargs_small
        )
        prev_btn.grid(row=1, column=1, pady=5, sticky="ew")

        # Next Page Button
        next_btn = tk.Button(
            frame, 
            text=">",
            command=lambda: self._on_change_page(1), 
            **self.common_kwargs_small
        )
        next_btn.grid(row=1, column=2, pady=5, sticky="ew")


    # ---------------------------------------------------------
    # ROW 9-10, COL 0: Mask Gen Buttons
    # ---------------------------------------------------------
    def _mask_generate_buttons(self):
        """
        Creates two buttons for mask generation
        - Generate -> Generates the mask or masks in the displayed image
        - Label Rest -> Labels the rest of the images with the specified settings
        """

        # Frame for mask generation buttons
        gen_buttons_frame = tk.LabelFrame(self.root, text="Mask Generation", font=("Arial", 10), relief="solid", bd=1)
        gen_buttons_frame.grid(row=9, column=0, rowspan=1, columnspan=1, padx=5, pady=5, sticky="nsew")
        
        # Add importance to grid frame for good spacing
        for c in range(2):
            gen_buttons_frame.grid_columnconfigure(c, weight=1)

        # --- Generate Buttons --- #
        # Mask button
        multiple_mask_tgl = self._make_button(
            gen_buttons_frame,
            text="Generate",
            command=self._on_generate_mask,
            highlight="black"
        )
        multiple_mask_tgl.grid(row=0, column=0, padx=5)

        # Label Rest button
        label_rest_btn = self._make_button(
            gen_buttons_frame,
            text="Label Rest",
            command=self._on_label_rest,
            highlight="black"
        )
        label_rest_btn.grid(row=0, column=1, padx=15)


    # ---------------------------------------------------------
    # ROW 8-10, COL 1: Mask Input Buttons
    # ---------------------------------------------------------
    def _mask_input_buttons(self):
        """
        Creates two toggle buttons to control the type of output masks
        - Single -> Single mask output
        - Multiple -> Multiple masks output
        """
        # Frame for mask generation buttons
        input_buttons_frame = tk.LabelFrame(self.root, text="Input", font=("Arial", 10), relief="solid", bd=1)
        input_buttons_frame.grid(row=8, column=1, rowspan=3, columnspan=1, padx=5, pady=5, sticky="nsew")
        
        # Add importance to grid frame for good spacing
        input_buttons_frame.grid_columnconfigure(0, weight=1)
        for r in range(4):
            input_buttons_frame.rowconfigure(r, weight=1)

        # --- Toggle Buttons --- #
        # Single Mask Toggle
        self.single_mask_tgl = self._make_button(
            input_buttons_frame, 
            text="Single Mask", 
            command=lambda:self._set_generation_mode("single")
        )
        self.single_mask_tgl.grid(row=0, column=0, pady=5)
        
        # Multiple Masks Toggle
        self.multiple_mask_tgl = self._make_button(
            input_buttons_frame, 
            text="Multiple Masks", 
            command=lambda:self._set_generation_mode("multiple")
        )
        self.multiple_mask_tgl.grid(row=1, column=0, pady=5)

        # Bounding Box Toggle: Click and drag
        self.bbox_btn = self._make_button(
            input_buttons_frame,
            text="BBox: OFF",
            command=self._toggle_bbox_mode
        )
        self.bbox_btn.grid(row=3, column=0, pady=5)


    # ---------------------------------------------------------
    # ROW 8-10, COL 2: Decision Buttons
    # ---------------------------------------------------------
    def _decision_buttons(self):
        """
        Creates 4 buttons in the same column
        - Correct -> Label as correct, save and pass to next image
        - Inorrect -> Label as incorrect, save and pass to next image
        - Discard -> Label as discarded, pass to next image without saving
        """ 
        # Frame for the two buttons
        decision_buttons_frame = tk.LabelFrame(self.root, text="Correctness Desicion", font=("Arial", 10), relief="solid", bd=1)
        decision_buttons_frame.grid(row=8, column=2, rowspan=3, columnspan=1, padx=5, pady=5, sticky="nsew")
        decision_buttons_frame.grid_columnconfigure(0, weight=1)
        # Add importance to grid frame for good spacing
        decision_buttons_frame.grid_columnconfigure(0, weight=1)
        for r in range(4):
            decision_buttons_frame.rowconfigure(r, weight=1)

        # Correct = status "correct"
        correct_btn = self._make_button(
            decision_buttons_frame,
            text="Correct",
            command=lambda: self._on_submit_decision("correct"),
            highlight="green"
        )
        correct_btn.grid(row=0, column=0, pady=5)

        # Incorrect = status "incorrect"
        incorrect_btn = self._make_button(
            decision_buttons_frame, 
            text="Incorrect", 
            command=lambda: self._on_submit_decision("incorrect"),
            highlight="orange"
        )
        incorrect_btn.grid(row=1, column=0, pady=5)

        # Discard = status "discarded"
        discard_btn = self._make_button(
            decision_buttons_frame, 
            text="Discard", 
            command=lambda: self._on_submit_decision("discard"),
            highlight="red"
        )
        discard_btn.grid(row=2, column=0, pady=5)


    # ---------------------------------------------------------
    # ROW 11, COL 0: Error Message
    # ---------------------------------------------------------
    def _build_status_bar(self):
        """
        Builds the bottom status bar containing the error label and progress bar.
        Called once during UI setup.
        """
        self.status_frame = tk.Frame(self.root)
        self.status_frame.grid(row=11, column=0, columnspan=3, sticky="ew", pady=5)

        # Error label
        self.error_label = tk.Label(self.status_frame, text="", font=("Arial", 12, "bold"), fg="red")
        self.error_label.grid(row=0, column=1, sticky="e", padx=20)

        # Progress bar (very small)
        self.progress = ttk.Progressbar(self.status_frame, orient="horizontal", mode="determinate",length=200)
        self.progress.grid(row=0, column=0, sticky="w", padx=10)

        # Set max steps
        self.progress["maximum"] = len(self.imgs_paths) - 1
        self._update_progress()


    # ------------------------------------------------------------------------------------------------ #
    #                                      Button helper functions                                     #
    # ------------------------------------------------------------------------------------------------ #
    def _on_change_page(self, step: int):
        """
        Move to previous or next selected page

        :param step: -1 for previous, +1 for next
        """
        # Local index of the valid pages to label
        valid_pages = self.valid_pages_to_label
        idx = valid_pages.index(self.current_page_index)
        new_idx = idx + step

        if 0 <= new_idx < len(valid_pages):
            self.current_page_index = valid_pages[new_idx]
            self._show_page()

    # ---------------------------------------------------------
    def _on_generate_mask(self):
        """
        Action function called when the button Generate Mask is clicked
        Checks the correctness of the input data given by the user
        - Correct data -> Calls the model to predict masks with the input & prints the mask overlay
        - Incorrect data -> Calls the _print_error function
        """
        assert self.prompt_text is not None

        # Clear errors
        self._update_error()

        # Get data
        prompt = self.prompt_text.get("1.0", "end-1c")
        
        # --- Check data correctness --- #
        # Check prompt is not empty
        if not prompt:
            self._update_error("Prompt cannot be empty!")
            return
        
        # Check a mask has been generated
        if self.generation_mode is None:
            self._update_error("Select an output type! -> ('Single Mask' or 'Multiple Masks')")
            return

        # --- Run model on current image --- #
        if self.current_image is None:
            self._update_error("No image loaded")
            return
        pil_page = self._get_page(self.current_page_index)

        # Save page model output
        output = self.model_manager.run_model(pil_page, prompt, self.generation_mode, bbox=self.bbox)
        self.page_outputs[self.current_page_index] = output

        # Render & Display mask overlay
        self._show_page()

    # ---------------------------------------------------------
    def _on_submit_decision(self, status_value):
        """
        Action function called when the buttons Correct, Incorrect or Discard are clicked
        Checks for a generated mask
        - Mask generated -> Decides what to do with the mask based on status_value and passes to the next image
        - Mask not generated -> Calls the _print_error function

        :param status_value: determines the status os the mask values are -> 'correct', 'incorrect' or 'discarded'
        """
        # Clear errors
        self._update_error()

        # --- Check data correctness --- #
        # Check mask has been generated for every page
        for vp in self.valid_pages_to_label:
            if self.page_outputs[vp] is None and status_value != "discard":
                self._update_error("Please generate the mask in all pages before submitting")
                return

        # --- Save current image (all pages) to CSV ---
        self.model_manager.save_single_image(
            img_path=self.imgs_paths[self.current_img_index],
            num_pages=self.num_pages,
            page_outputs=self.page_outputs,
            correctness_label=status_value,
            header=self.header
        )
        # Clear pages
        self.page_outputs = {p: None for p in self.valid_pages_to_label}

        # --- Pass to next image --- #
        # If this is the last image, close the window
        if self.current_img_index >= len(self.imgs_paths) - 1:
            self._update_error("Finished!")
            self.root.destroy()
            return
         
        self.current_img_index += 1
        self._load_image(self.current_img_index)    # Show next image
        self._update_progress()                     # Update progress bar

    # ---------------------------------------------------------
    def _on_label_rest(self):
        """
        Handle labeling of the remaining images starting from the current index

        Validates the user prompt and generation mode, then runs the model
        in bulk on all remaining images before closing the UI
        """
        assert self.prompt_text is not None

        # Clear errors
        self._update_error()

        # Get data
        prompt = self.prompt_text.get("1.0", "end-1c")

        # --- Check data correctness --- #
        # Check prompt is not empty
        if not prompt:
            self._update_error("Prompt cannot be empty")
            return

        # Check a mask has been generated
        if self.generation_mode is None:
            self._update_error("Select an output type! -> ('Single Mask' or 'Multiple Masks')")
            return
        
        # --- Run model in bulk for the rest of the images --- #
        remaining_paths = self.imgs_paths[self.current_img_index:]
        remaining_paths = self._filter_paths_by_num_pages(remaining_paths)
        if not remaining_paths:
            self.root.destroy()
            return

        rows = self.model_manager.run_model_bulk(remaining_paths, prompt, self.header, self.generation_mode)
        self.model_manager.save_rows_to_csv(rows, self.header)

        # Close UI after processing
        self.root.destroy()


# ------------------------------------------------------------------------------------------------ #
#                                         HELPER FUNCTIONS                                         #
# ------------------------------------------------------------------------------------------------ #
    def _update_error(self, msg=""):
        """
        Updates the error message in the status bar

        :param msg (str): Error message string
        """
        assert self.error_label is not None
        assert self.progress is not None

        self.error_label.config(text=msg)

    # ---------------------------------------------------------
    def _update_progress(self):
        """
        Updates the progress bar based on current image index
        """
        assert self.error_label is not None
        assert self.progress is not None

        self.progress["value"] = self.current_img_index

    def _prepare_image_pages(self, pil_img):
        """
        Given an image prepare the pages
        
        :param pil_img: PIL Image that we are currently on
        :retrun: the valid pages of such image
        """
        num_pages = getattr(pil_img, "n_frames", 1)
        pages = [p for p in self.pages_to_label if 0 <= p < num_pages]
        return pages, num_pages

    def _extend_header_for_pages(self, num_pages: int) -> None:
        """
        Extends the header once with per-page zoom/mask columns.
        """
        base_cols = ["fileName", "status"]
        zoom_cols = []
        mask_cols = []
        for i in range(num_pages):
            zoom_cols.extend([f"ZoomX{i}", f"ZoomY{i}", f"ZoomWidth{i}", f"ZooomHeight{i}"])
            mask_cols.append(f"Mask{i}")
        self.header[:] = base_cols + zoom_cols + mask_cols

    def _filter_paths_by_num_pages(self, paths: list[str]) -> list[str]:
        """
        Keeps only images with the expected number of pages.
        """
        if self.expected_num_pages is None:
            return paths

        filtered = []
        for p in paths:
            try:
                with Image.open(p) as img:
                    num_pages = getattr(img, "n_frames", 1)
            except Exception:
                continue

            if num_pages != self.expected_num_pages:
                print(
                    f"[WARNING]: Image {p} has {num_pages} instead of {self.expected_num_pages} "
                    "it might correspond toa different dataset."
                )
                continue

            filtered.append(p)

        return filtered
    
    def _init_page_outputs(self, valid_pages: list[int]) -> None:
        """
        Initalize the page outputs dict

        :param valid_pages: Valid pages to label of this image
        """
        self.page_outputs = {p: None for p in valid_pages}

    # ---------------------------------------------------------
    def _make_button(self, parent, *, text, command, highlight=None):
        """
        Creates a tkinter button
        
        :param parent: Parent frame to where the button is placed
        :param text: Text of the button
        :param command: Command activated by pressing the button
        :param highlight: Color of the button highlight (default false)
        """
        btn = tk.Button(parent, text=text, command=command, **self.common_kwargs)
        if highlight:
            btn.config(highlightbackground=highlight, highlightthickness=2)
        return btn
    
    # ---------------------------------------------------------
    def _update_generation_toggle(self):
        """
        Update visual state of generation mode toggle buttons
        """
        assert self.single_mask_tgl is not None
        assert self.multiple_mask_tgl is not None

        if self.generation_mode == "single":
            self.single_mask_tgl.config(relief="sunken", bg="#d0f0d0")
            self.multiple_mask_tgl.config(relief="raised", bg=self.default_btn_bg)
        else:
            self.single_mask_tgl.config(relief="raised", bg=self.default_btn_bg)
            self.multiple_mask_tgl.config(relief="sunken", bg="#d0f0d0")

    # ---------------------------------------------------------
    def _set_generation_mode(self, mode: str):
        """
        Set generation mode and update toggle button visuals

        :param mode: "single" or "multiple"
        """
        self.generation_mode = mode
        self._update_generation_toggle()

    # ---------------------------------------------------------
    def _toggle_bbox_mode(self):
        """
        Toggle bounding box drawing mode on or off

        When enabled, users can draw a bounding box on the image canvas.
        Disabling the mode clears any existing bounding box
        """
        assert self.img_canvas is not None
        assert self.bbox_btn is not None

        self.bbox_enabled = not self.bbox_enabled
        self.bbox = None
        self.img_canvas.delete("bbox")

        state = "ON" if self.bbox_enabled else "OFF"
        self.bbox_btn.config(text=f"BBox: {state}")

    # ---------------------------------------------------------
    def _on_bbox_start(self, event):
        """
        Handle the start of a bounding box drawing action

        Stores the initial mouse position when the user starts
        drawing a bounding box on the canvas

        :param event: Tkinter mouse event
        """
        assert self.img_canvas is not None
        assert self.bbox_btn is not None
        if not self.bbox_enabled:
            return
        
        self.bbox_start = (event.x, event.y)
        self.img_canvas.delete("bbox")

    # ---------------------------------------------------------
    def _on_bbox_drag(self, event):
        """
        Handle mouse drag events while drawing a bounding box

        Updates the bounding box rectangle dynamically as the
        mouse is dragged across the canvas

        :param event: Tkinter mouse event
        """
        assert self.img_canvas is not None
        assert self.bbox_btn is not None
        if not self.bbox_enabled or not self.bbox_start:
            return

        x0, y0 = self.bbox_start
        self.img_canvas.delete("bbox")
        self.img_canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="yellow",
            width=2,
            tags="bbox"
        )

    # ---------------------------------------------------------
    def _on_bbox_end(self, event):
        """
        Handle the end of a bounding box drawing action

        Finalizes the bounding box, converts canvas coordinates
        to original image coordinates, and clamps them to valid bounds

        :param event: Tkinter mouse event
        """
        assert self.img_canvas is not None
        assert self.bbox_btn is not None
        assert self.orig_w is not None
        assert self.orig_h is not None
        if not self.bbox_enabled or not self.bbox_start:
            return

        x0, y0 = self.bbox_start
        x1, y1 = event.x, event.y

        # Normalize order
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        # Scale from displayed image coords -> original image coords
        ox0 = int(x0 * self.disp_sx)
        oy0 = int(y0 * self.disp_sy)
        ox1 = int(x1 * self.disp_sx)
        oy1 = int(y1 * self.disp_sy)

        # Clamp to original bounds
        ox0 = max(0, min(self.orig_w - 1, ox0))
        ox1 = max(0, min(self.orig_w - 1, ox1))
        oy0 = max(0, min(self.orig_h - 1, oy0))
        oy1 = max(0, min(self.orig_h - 1, oy1))

        self.bbox = (ox0, oy0, ox1, oy1)
        self.bbox_start = None

    # ---------------------------------------------------------
    def _update_page_label(self):
        """
        Updates the page information label if it exists
        """
        assert self.page_info_label is not None

        # Store values and compute page number and total pages
        img_name = Path(self.imgs_paths[self.current_img_index]).name
        p_idx = self.current_page_index
        valid_pages = self.valid_pages_to_label
        shown_idx = valid_pages.index(p_idx) + 1
        total = len(valid_pages)

        # Display label
        self.page_info_label.config(
            text=f"Displaying selected page {shown_idx} of {total} "
                f"(page {p_idx}) in image {img_name}"
        )
    
    # ---------------------------------------------------------
    def _get_page(self, page_index: int) -> Image.Image:
        """
        Get the page on given index from the image container
        
        :param page_index: Index of the page we are looking for
        :return: The desired page as a PIL Image
        """
        assert self.current_image is not None
        self.current_image.seek(page_index)
        return self.current_image.copy()
    
    # ---------------------------------------------------------
    def _get_overlay_image(self, page_index: int) -> Image.Image:
        """
        Returns a PIL image of the given page with mask overlay applied
        This image is not yet resized for UI display
        
        :param page_index: Index of the page in the TIFF
        """        
        # Get the page
        pil_page = self._get_page(page_index)
        self.orig_w, self.orig_h = pil_page.size

        # Overlay mask if exists
        output = self.page_outputs.get(page_index)
        if output is not None:
            pil_page = self.model_manager.render_image_with_mask(pil_page, output)
        
        return pil_page
    
    # ---------------------------------------------------------
    def _show_page(self):
        """
        Displays the currently selected page of the current image
        """
        assert self.img_canvas is not None
        assert self.page_info_label is not None

        # Get PIL image with overlay
        pil_page = self._get_overlay_image(self.current_page_index)

        # Resize to fit UI canvas
        max_display_size = (900, 900)
        pil_page.thumbnail(max_display_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_page)
        
        # Calculate scaling factors for bbox
        disp_w, disp_h = pil_page.size
        assert self.orig_w is not None
        assert self.orig_h is not None
        self.disp_sx = self.orig_w / disp_w
        self.disp_sy = self.orig_h / disp_h

        # Display in canvas
        self.img_canvas.delete("all")  # Clear previous
        self.canvas_img_id = self.img_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # Reset bbox
        self.bbox = None
        self.bbox_start = None

        # Update page label
        self._update_page_label()

    
    
        
# ---------------------------------------------------------
# Debug grid
# def debug_grid(self):
#     """Overlay colored frames on each grid cell to visualize layout."""
#     import random

#     for r in range(12):
#         for c in range(3):
#             frame = tk.Frame(
#                 self.root,
#                 bg=f"#{random.randint(0, 0xFFFFFF):06x}",
#                 highlightbackground="black",
#                 highlightthickness=1,
#             )
#             frame.grid(row=r, column=c, sticky="nsew")
#             label = tk.Label(frame, text=f"r{r},c{c}", bg=frame["bg"])
#             label.pack()
