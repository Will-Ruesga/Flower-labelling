import tkinter as tk

from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk

from model_manager import ModelManager


BTN_WIDTH = 12
BTN_HEIGHT = 1


########################################
#            Prompt Class UI           #
########################################
class PromptUI:
    def __init__(self, processor, imgs_paths: list, pages_to_label: list[int], header: list[str]):
        """
        :param processor: The model processor
        :param imgs_paths: List of absolute paths of all the iamges
        :param data_type: String with the type of the data -> ("csv", "png")
        :param header: List of strings with the header columns of the csv
        """
        # --- Initlize --- #
        # Inputs
        self.prompt = None
        
        # Model
        self.processor = processor

        # Data
        self.imgs_paths = imgs_paths
        self.selected_pages = sorted(set(pages_to_label))
        self.header = header

        self.current_img_index = 0
        self.current_page_index = 0
        self.num_pages = 1
        self.current_pil_image = None
        
        # States
        self.generation_mode = None # "single" or "multiple"

        # Bbox
        self.bbox_enabled = False
        self.bbox = None
        self.bbox_start = None
        self.bbox_rect_id = None

        # Button configuration
        self.common_kwargs = {
            "font": ("Arial", 14),
            "width": BTN_WIDTH,
            "height": BTN_HEIGHT,
            "relief": "solid",
            "bd": 2,
        }

        # Model Manager
        self.model_manager = ModelManager(self.processor)

        # --- UI (root) --- #
        self.root = tk.Tk()
        self.root.geometry("1200x1100")
        self.root.title("Prompt Labelling")
        self.root.minsize(1200, 1000)

        # Button background
        self.default_btn_bg = tk.Button(self.root).cget("bg")

        # --- UI Layout --- #
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
        self._show_image(self.current_img_index)

        # Add frames to the UI
        self._build_status_bar()
        self._prompt_text_box()
        self._mask_input_buttons()
        self._mask_generate_buttons()
        self._decision_buttons()

        # Start the loop
        self.root.mainloop()


    ########################################
    #    ROW 0-7, COL 0-3: Image Display   #
    ########################################
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

    
    def _show_image(self, image_index: int, page_index: int = 0):
        """
        Loads and displays the image at the given index without overlay.

        :param index: Index of the image in self.imgs_paths
        """
        img_path = self.imgs_paths[image_index]

        # Open image & Get pages
        self.current_pil_image = Image.open(img_path)
        self.num_pages = getattr(self.current_pil_image, "n_frames", 1)

        # Filter selected pages for this image
        self.valid_pages_for_image = [p for p in self.selected_pages if p < self.num_pages]
        self.current_selected_page_idx = 0

        # No valid pages → skip image
        if not self.valid_pages_for_image:
            self._advance_to_next_image()
            return


    ########################################
    #   ROW 8-9, COL 0: Text Box Prompt   #
    ########################################
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
        self.page_info_label = tk.Label(prompt_frame, text="", font=("Arial", 11, "italic"), fg="gray")
        self.page_info_label.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="nw")

        # Text box 
        self.prompt_text = tk.Text(prompt_frame, height=5,font=("Arial", 14),wrap="word",)
        self.prompt_text.grid(row=2, column=0, padx=10, sticky="ew")


    ########################################
    #   ROW 9-10, COL 0: Mask Gen Buttons  #
    ########################################
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

        # --- Generate Mask Button --- #
        multiple_mask_tgl = tk.Button(
            gen_buttons_frame, 
            text="Generate", 
            command=self._on_generate_mask, 
            **self.common_kwargs
        )
        multiple_mask_tgl.grid(row=0, column=0, padx=5)
        multiple_mask_tgl.config(highlightbackground="black", highlightthickness=2)

        # Label Rest button
        label_rest_btn = tk.Button(
            gen_buttons_frame, 
            text="Label Rest", 
            command=self._on_label_rest, 
            **self.common_kwargs
        )
        label_rest_btn.grid(row=0, column=1, padx=15)
        label_rest_btn.config(highlightbackground="black", highlightthickness=2)
    

    ########################################
    #  ROW 8-10, COL 1: Mask Input Buttons #
    ########################################
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
        self.single_mask_tgl = tk.Button(
            input_buttons_frame, 
            text="Single Mask", 
            command=lambda:self._set_generation_mode("single"), 
            **self.common_kwargs
        )
        self.single_mask_tgl.grid(row=0, column=0, pady=5)
        
        # Multiple Masks Toggle
        self.multiple_mask_tgl = tk.Button(
            input_buttons_frame, 
            text="Multiple Masks", 
            command=lambda:self._set_generation_mode("multiple"), 
            **self.common_kwargs
        )
        self.multiple_mask_tgl.grid(row=1, column=0, pady=5)

        # Bounding Box Toggle: Click and drag
        self.bbox_btn = tk.Button(
            input_buttons_frame,
            text="BBox: OFF",
            command=self._toggle_bbox_mode,
            **self.common_kwargs
        )
        self.bbox_btn.grid(row=3, column=0, pady=5)


    ########################################
    #   ROW 8-10, COL 2: Decision Buttons  #
    ########################################
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
        correct_btn = tk.Button(
            decision_buttons_frame,
            text="Correct",
            command=lambda: self._submit_decision("correct"), 
            **self.common_kwargs
        )
        correct_btn.grid(row=0, column=0, pady=5)
        correct_btn.config(highlightbackground="green", highlightthickness=2)

        # Incorrect = status "incorrect"
        incorrect_btn = tk.Button(
            decision_buttons_frame, 
            text="Incorrect", 
            command=lambda: self._submit_decision("incorrect"), 
            **self.common_kwargs
        )
        incorrect_btn.grid(row=1, column=0, pady=5)
        incorrect_btn.config(highlightbackground="orange", highlightthickness=2)

        # Discard = status "discarded"
        discard_btn = tk.Button(
            decision_buttons_frame, 
            text="Discard", 
            command=lambda: self._submit_decision("discarded"), 
            **self.common_kwargs
        )
        discard_btn.grid(row=2, column=0, pady=5)
        discard_btn.config(highlightbackground="red", highlightthickness=2)


    ########################################
    #      ROW 11, COL 0: Error Message    #
    ########################################
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
        self.progress["maximum"] = len(self.imgs_paths)
        self._update_progress()


    # ------------------------------------------------------------------------------------------------ #
    #                                      Button helper functions                                     #
    # ------------------------------------------------------------------------------------------------ #
    def _on_generate_mask(self):
        """
        Action function called when the button Generate Mask is clicked
        Checks the correctness of the input data given by the user
        - Correct data -> Calls the model to predict masks with the input & prints the mask overlay
        - Incorrect data -> Calls the _print_error function
        """
        # Get data
        self.prompt = self.prompt_text.get("1.0", "end-1c")
        
        # --- Check data correctness --- #
        # Check prompt is not empty
        if not self.prompt:
            self._update_error("Prompt cannot be empty!")
            return
        
        # Check a mask has been generated
        if self.generation_mode is None:
            self._update_error("Select an output type! -> ('Single Mask' or 'Multiple Masks')")
            return

        # --- Run model on current image --- #
        image = self.current_pil_image.copy()

        self.model_output = self.model_manager.run_model(image, self.prompt, self.generation_mode, bbox=self.bbox)

        # Render & Display mask overlay instead of the raw image
        overlay_img = self.model_manager.render_image_with_mask(image, self.model_output)
        overlay_img.thumbnail((900, 900))
        self.tk_img = ImageTk.PhotoImage(overlay_img)

        # Update the existing canvas image item
        self.img_canvas.itemconfig(self.canvas_img_id, image=self.tk_img)
        self._img_ref = self.tk_img

        # Update states
        self._update_error("")


    ########################################
    def _submit_decision(self, status_value):
        """
        Action function called when the buttons Correct, Incorrect or Discard are clicked
        Checks for a generated mask
        - Mask generated -> Decides what to do with the mask based on status_value and passes to the next image
        - Mask not generated -> Calls the _print_error function

        :param status_value: determines the status os the mask values are -> 'correct', 'incorrect' or 'discarded'
        """
        # --- Check data correctness --- #
        # Check mask has been generated
        if not hasattr(self, "model_output") or self.model_output is None:
            self._update_error("Please generate the mask before submitting")
            return

        # --- Save current image to CSV ---
        self.model_manager.save_single_image(
            img_path=self.imgs_paths[self.current_img_index],
            output=self.model_output,
            correctness_label=status_value,
            header=self.header
        )

        # Move to next selected page if available
        if self.current_selected_page_idx < len(self.valid_pages_for_image) - 1:
            self.current_selected_page_idx += 1
            self.model_output = None
            self._show_current_selected_page()
            self._update_error("")
            return
        
        # --- Pass to next image --- #
        # If this is the last image, close the window
        if self.current_img_index >= len(self.imgs_paths) - 1:
            self.root.destroy()
            return
         
        self.current_img_index += 1
        self.model_output = None                    # Reset status
        self._show_image(self.current_img_index)    # Show next image
        self._update_progress()                     # Update progress bar
        self._update_error("")                      # Clear errors
        

    ########################################
    def _on_label_rest(self):
        # Get data
        self.prompt = self.prompt_text.get("1.0", "end-1c")

        # --- Check data correctness --- #
        # Check prompt is not empty
        if not self.prompt:
            self._update_error("Prompt cannot be empty")
            return

        # Check a mask has been generated
        if self.generation_mode is None:
            self._update_error("Select an output type! -> ('Single Mask' or 'Multiple Masks')")
            return
        
        # --- Run model in bulk for the rest of the images --- #
        remaining_paths = self.imgs_paths[self.current_img_index:]

        self.model_manager.run_model_bulk(remaining_paths, self.prompt, self.header, self.generation_mode)

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
            self.error_label.config(text=msg)

    def _update_progress(self):
        """
        Updates the progress bar based on current image index
        """
        self.progress["value"] = self.current_img_index

    def _update_generation_toggle(self):
        """
        Update visual state of generation mode toggle buttons
        """
        if self.generation_mode == "single":
            self.single_mask_tgl.config(relief="sunken", bg="#d0f0d0")
            self.multiple_mask_tgl.config(relief="raised", bg=self.default_btn_bg)
        else:
            self.single_mask_tgl.config(relief="raised", bg=self.default_btn_bg)
            self.multiple_mask_tgl.config(relief="sunken", bg="#d0f0d0")

    def _set_generation_mode(self, mode: str):
        """
        Set generation mode and update toggle button visuals

        :param mode: "single" or "multiple"
        """
        self.generation_mode = mode
        self._update_generation_toggle()

    def _toggle_bbox_mode(self):
        self.bbox_enabled = not self.bbox_enabled
        self.bbox = None
        self.img_canvas.delete("bbox")

        state = "ON" if self.bbox_enabled else "OFF"
        self.bbox_btn.config(text=f"BBox: {state}")

    def _on_bbox_start(self, event):
        if not self.bbox_enabled:
            return
        self.bbox_start = (event.x, event.y)
        self.img_canvas.delete("bbox")

    def _on_bbox_drag(self, event):
        if not self.bbox_enabled or not self.bbox_start:
            return

        x0, y0 = self.bbox_start
        self.img_canvas.delete("bbox")
        self.bbox_rect_id = self.img_canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="yellow",
            width=2,
            tags="bbox"
        )

    def _on_bbox_end(self, event):
        if not self.bbox_enabled or not self.bbox_start:
            return

        x0, y0 = self.bbox_start
        x1, y1 = event.x, event.y

        # Normalize order
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        # Scale from displayed image coords -> original image coords
        sx = self.orig_w / max(1, self.disp_w)
        sy = self.orig_h / max(1, self.disp_h)

        ox0 = int(x0 * sx)
        oy0 = int(y0 * sy)
        ox1 = int(x1 * sx)
        oy1 = int(y1 * sy)

        # Clamp to original bounds
        ox0 = max(0, min(self.orig_w - 1, ox0))
        ox1 = max(0, min(self.orig_w - 1, ox1))
        oy0 = max(0, min(self.orig_h - 1, oy0))
        oy1 = max(0, min(self.orig_h - 1, oy1))

        self.bbox = (ox0, oy0, ox1, oy1)
        self.bbox_start = None

    def _update_page_label(self):
        img_name = Path(self.imgs_paths[self.current_img_index]).name
        shown_idx = self.current_selected_page_idx + 1
        total = len(self.valid_pages_for_image)
        page = self.current_page_index

        self.page_info_label.config(
            text=f"Displaying selected page {shown_idx} of {total} "
                f"(page {page}) in image {img_name}")

    def _show_current_selected_page(self):
        # Open the page
        page = self.valid_pages_for_image[self.current_selected_page_idx]
        self.current_page_index = page

        # Save original size for bbox scaling
        self.current_pil_image.seek(page)
        image = self.current_pil_image.copy()
        self.orig_w, self.orig_h = image.size

        # Show image in canvas (resized for UI)
        self.img_canvas.delete("all")
        image.thumbnail((900, 900))
        self.disp_w, self.disp_h = image.size
        self.tk_img = ImageTk.PhotoImage(image)

        # Store the canvas image item id
        self.canvas_img_id = self.img_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self._img_ref = self.tk_img

        # Reset bbox per image
        self.bbox = None
        self.bbox_start = None
        self.img_canvas.delete("bbox")

        # Update page label
        self._update_page_label()

        
########################################
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
