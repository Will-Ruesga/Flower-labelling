
import tkinter as tk
from tkinter import ttk

from model_manager import ModelManager
from utils.image_utils import filter_paths_by_num_pages, init_header_from_first_image
from utils.data_utils import save_rows_to_csv


# =================================================================================================
#                                       Automatic UI Class
# =================================================================================================
class AutomaticUI:
    def __init__(self, processor, imgs_paths: list, header: list[str]):
        """
        :param processor: The model processor
        :param imgs_paths: List of absolute paths of all the iamges
        :param data_type: String with the type of the data -> ("csv", "png")
        :param header: List of strings with the header columns of the csv
        """
        # Initlize varaibles
        self.processor = processor
        self.imgs_paths = imgs_paths
        self.header = header
        self.prompt = None
        self.mask_output_type = "multiple"
        self.expected_num_pages: int | None = None

        # Create root once
        self.root = tk.Tk()
        self.root.geometry("600x500")
        self.root.title("Automatic Labelling")
        self.root.minsize(400, 200)

        # Configure grid to scale properly
        for r in range(4):
            self.root.rowconfigure(r, weight=1)
        for c in range(1):
            self.root.columnconfigure(c, weight=1)

        # Create a model manager to keep UI clean
        self.model_manager = ModelManager(self.processor)

        # Add frames to the UI
        self._prompt_text_box()
        self._mask_output_radio()
        self._sumbit_info_button()

        # Start the loop
        self.root.mainloop()


    # ---------------------------------------------------------
    # ROW 0, COL 0-1: Text Box for Prompt
    # ---------------------------------------------------------
    def _prompt_text_box(self):
        """
        Generates a text box where the user can write a small prompt for the model
        """
        # Frame
        prompt_frame = tk.Frame(self.root)
        prompt_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

        # Make column 0 expand
        prompt_frame.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Label
        promt_text_label = tk.Label(prompt_frame, text="Enter model prompt:", font=("Arial", 12))
        promt_text_label.grid(row=0, column=0, columnspan=1, padx=10, pady=(0, 5), sticky="nw")

        # Text box
        self.prompt_text = tk.Text(prompt_frame, height=5,font=("Arial", 14),wrap="word",)
        self.prompt_text.grid(row=1, column=0, padx=10, sticky="ew")
    

    # ---------------------------------------------------------
    # ROW 1, COL 0-1: Mult Mask Radio Button
    # ---------------------------------------------------------
    def _mask_output_radio(self):
        """
        Creates two radio buttons in a row to control the number of output masks
        - Single mask
        - Multiple masks
        """
        # Frame
        mask_out_radio_frame = tk.Frame(self.root)
        mask_out_radio_frame.grid(row=1, column=0, columnspan=2, pady=10)

        mask_out_label = tk.Label(mask_out_radio_frame, text="Select type of mask output:", font=("Arial", 14))
        mask_out_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Radio selection variable
        self.mask_output_type_var = tk.StringVar(value="")

        # Single mask radio
        single_mask_radio = ttk.Radiobutton(mask_out_radio_frame, text="Single mask", variable=self.mask_output_type_var, value="single")
        single_mask_radio.grid(row=1, column=0, padx=20)

        # Multiple mask radio
        mult_mask_radio = ttk.Radiobutton(mask_out_radio_frame, text="Multiple masks", variable=self.mask_output_type_var, value="multiple")
        mult_mask_radio.grid(row=1, column=1, padx=20)


    # ---------------------------------------------------------
    # ROW 2, COL 0: Error Message
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


    # ---------------------------------------------------------
    # ROW 3, COL 0: Submit Info Button
    # ---------------------------------------------------------
    def _sumbit_info_button(self):
        """
        Creates a button to submit the information
        When clicked check all fields have an answer and close the UI
        """ 
        # Submit button
        submit_btn = tk.Button(self.root, text="Submit", command=self._on_submit, font=("Arial", 14))
        submit_btn.grid(row=3, column=0, columnspan=2, padx=50, pady=20, sticky="s")


# =================================================================================================
#                                      Button helper functions
# =================================================================================================
    def _on_submit(self):
        """
        Action function called when the button Sumbit is clicked
        Checks the correctness of the input data given by the user
        - Correct data -> Closes the window finishing and executes the model
        - Incorrect data -> Calls the _print_error function
        """
        # Get data
        self.prompt = self.prompt_text.get("1.0", "end-1c")
        self.mask_output_type = None
        if hasattr(self, "mask_output_type_var"):
            mask_type_val = self.mask_output_type_var.get()
            if mask_type_val in ("multiple", "single"):
                self.mask_output_type = mask_type_val

        # --- Check data correctness --- #
        # Check prompt is not empty
        if not self.prompt:
            self._update_error("Prompt cannot be empty!")
            return

        # Check radio button selected
        if self.mask_output_type is None:
            self._update_error("Please choose an output type!")
            return

        # Set header once based on the first image
        if not self.imgs_paths:
            self._update_error("No images to process.")
            return
        if self.expected_num_pages is None:
            self.expected_num_pages = init_header_from_first_image(self.imgs_paths, self.header)
        self.imgs_paths = filter_paths_by_num_pages(self.imgs_paths, self.expected_num_pages)
        if not self.imgs_paths:
            self._update_error("No valid images found.")
            return
        
        # --- Run model in bulk for all the images --- #
        rows = self.model_manager.run_model_bulk(
            self.imgs_paths,
            self.prompt,
            self.header,
            self.mask_output_type,
        )
        save_rows_to_csv(rows, self.header)
        
        # Close UI after processing
        self.root.destroy()

    # ---------------------------------------------------------
    def _update_error(self, msg=""):
        """
        Updates the error message in the status bar

        :param msg (str): Error message string
        """
        self.error_label.config(text=msg)