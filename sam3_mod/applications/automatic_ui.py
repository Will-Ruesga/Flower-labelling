
import tkinter as tk
from tkinter import ttk

from img_pred_utils import process_all_images


########################################
#          Automatic Class UI          #
########################################
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
        self.mask_output = False

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

        # Add frames to the UI
        self._prompt_text_box()
        self._mask_output_radio()
        self._sumbit_info_button()

        # Start the loop
        self.root.mainloop()

        # Execute the model wiht the prompt
        process_all_images(processor=processor, imgs_paths=imgs_paths, prompt=self.prompt, header=header)


    ########################################
    #  ROW 0, COL 0-1: Text Box for Prompt #
    ########################################
    def _prompt_text_box(self):
        """
        Generates a text box where the user can write a small promppt for the model
        """
        # Frame
        self.prompt_frame = tk.Frame(self.root)
        self.prompt_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

        # Make column 0 expand
        self.prompt_frame.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Label
        tk.Label(self.prompt_frame, text="Enter prompt", font=("Arial", 12)).grid(
            row=0, column=0, columnspan=1, pady=(0, 5), sticky="nw"
        )

        # Text box
        self.prompt_text = tk.Text(
            self.prompt_frame, 
            height=5,
            font=("Arial", 14),
            wrap="word",
        )
        self.prompt_text.grid(row=1, column=0, padx=10, sticky="ew")
    

    ########################################
    # ROW 1, COL 0-1: Mult Mask Radio Button #
    ########################################
    def _mask_output_radio(self):
        """
        Creates two radio buttons side by side that specify the use of single or multiple masks
        """
        # Frame
        self.radio_frame = tk.Frame(self.root)
        self.radio_frame.grid(row=1, column=0, columnspan=2, pady=10)

        tk.Label(self.radio_frame, text="Output mask type", font=("Arial", 14)).grid(
            row=0, column=0, columnspan=2, pady=(0, 10)
        )

        # Track radio selection
        self.mask_output_var = tk.StringVar(value="")

        # Single mask radio
        single_radio = ttk.Radiobutton(
            self.radio_frame, 
            text="Single", 
            variable=self.mask_output_var, 
            value="single"
        )
        single_radio.grid(row=1, column=0, padx=20)

        # Multiple mask radio
        mult_radio = ttk.Radiobutton(
            self.radio_frame, 
            text="Multiple", 
            variable=self.mask_output_var, 
            value="multiple"
        )
        mult_radio.grid(row=1, column=1, padx=20)


    ########################################
    #     ROW 2, COL 0: Error Message    #
    ########################################
    def _print_error(self, msg="Error in selection!"):
        """
        Creates a small text in red when there has been an error in the selection

        :param msg (str): Error message string
        """
        self.error_frame = tk.Frame(self.root)
        self.error_frame.grid(row=2, column=0, columnspan=2, pady=(5, 5))
        
        tk.Label(
            self.error_frame,
            text=msg,
            font=("Arial", 16, "bold"),
            fg="red"
        ).grid(row=0, column=0, columnspan=1, padx=10, pady=5)


    ########################################
    #   ROW 3, COL 0: Submit Info Button   #
    ########################################
    def _sumbit_info_button(self):
        """
        Creates a button to submit the information
        When clicked check all fields have an answer and close the UI
        """ 
        # Define helper functions
        def _on_submit():
            # Get data
            self.prompt = self.prompt_text.get("1.0", "end-1c")
            if hasattr(self, "mask_output_var"):
                self.mask_output = True if self.mask_output_var.get() == "multiple" else False
            else:
                self.mask_output = None

            # Check data correctness
            if (not self.prompt.strip()) or (self.mask_output is None):
                self._print_error()
            else:
                self.root.destroy()

        # Submit button
        submit_btn = tk.Button(self.root, text="Submit", command=_on_submit, font=("Arial", 14))
        submit_btn.grid(row=3, column=0, columnspan=2, padx=50, pady=20, sticky="s")