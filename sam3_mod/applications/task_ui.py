import tkinter as tk

from tkinter import ttk, filedialog


########################################
#            Choose UI Class           #
########################################
class TaskSelectionUI:
    def __init__(self, behaviors):
        # Initialize variables
        self.behaviors = behaviors
        self.task = None
        self.data_type = None
        self.data_abspath = ""

        # Create root once
        self.root = tk.Tk()
        self.root.geometry("600x500")
        self.root.title("Labeling App")
        self.root.minsize(400, 200)

        # Configure grid to scale properly
        for r in range(6):
            self.root.rowconfigure(r, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Add frames to the UI
        self._select_task_drop()
        self._data_type_radio()
        self._file_browser_frame()
        self._sumbit_info_button()

        # Start the loop
        self.root.mainloop()

    ########################################
    #     ROW 0-1: Task Drop Down Menu     #
    ########################################
    def _select_task_drop(self):
        """
        Creates Tkinter UI with a dropdown menu to select task type
        Once submitted, prints the selection
        """
        # Set value
        self.selected_task = tk.StringVar(value=self.behaviors[0])

        # Label
        tk.Label(self.root, text="Select the task to perform", font=("Arial", 16)).grid(
            row=0, column=0, columnspan=2, pady=20, sticky="n")

        # Dropdown / Combobox
        combo = ttk.Combobox(self.root, textvariable=self.selected_task, values=self.behaviors[1], state="readonly",
                             font=("Arial", 14))
        combo.grid(row=1, column=0, columnspan=2, padx=50, pady=10, sticky="ew")
    

    ########################################
    #     ROW 2: Data Type Radio Button    #
    ########################################
    def _data_type_radio(self):
        """
        Creates two radio buttons side by side that specify the data type -> (.csv or [images])
        """
        # New frame for radio buttons
        self.radio_frame = tk.Frame(self.root)
        self.radio_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Label(self.radio_frame, text="Data type", font=("Arial", 14)).grid(
            row=0, column=0, columnspan=2, pady=(0, 10)
        )

        # Track radio selection
        self.data_type_var = tk.StringVar(value="")

        # CSV radio
        csv_radio = ttk.Radiobutton(
            self.radio_frame, 
            text="CSV Data", 
            variable=self.data_type_var, 
            value="csv"
        )
        csv_radio.grid(row=1, column=0, padx=20)

        # Image radio
        img_radio = ttk.Radiobutton(
            self.radio_frame, 
            text="Image Folder", 
            variable=self.data_type_var, 
            value="png"
        )
        img_radio.grid(row=1, column=1, padx=20)


    ########################################
    #       ROW 3: File Browser Frame      #
    ########################################
    def _browse_for_path(self):
        """
        Opens a file/folder dialog depending on the selected data type
        Updates the text box and stores the path
        """
        # Make sure data_type is selected
        if not hasattr(self, "data_type_var") or self.data_type_var.get() == "":
            self._print_error("Please select a data type first!")
            return

        dtype = self.data_type_var.get()

        if dtype == "csv":
            path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        else:  # images
            path = filedialog.askdirectory(title="Select an Image Folder")

        if path:
            # Store it in the class
            self.data_abspath = path

            # Update the text box
            self.path_entry.config(state="normal")
            self.path_var.set(path)
            self.path_entry.config(state="disabled")


    def _file_browser_frame(self):
        """
        Creates a frame with a display-only text box and a 'Browse' button
        Allows the user to pick a CSV file or a folder depending on the data_type
        """
        # Frame
        self.browser_frame = tk.Frame(self.root)
        self.browser_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

        # Make column 0 expand
        self.browser_frame.columnconfigure(0, weight=1)

        # Label
        tk.Label(self.browser_frame, text="Select data path", font=("Arial", 14)).grid(
            row=0, column=0, columnspan=2, pady=(0, 5)
        )

        # TEXT BOX (disabled, display-only)
        self.path_var = tk.StringVar(value="")
        self.path_entry = tk.Entry(
            self.browser_frame, 
            textvariable=self.path_var, 
            font=("Arial", 12),
            state="disabled"
        )
        self.path_entry.grid(row=1, column=0, padx=10, sticky="ew")

        # Browse button
        browse_btn = tk.Button(self.browser_frame, text="Browse", font=("Arial", 12),
                            command=self._browse_for_path)
        browse_btn.grid(row=1, column=1, padx=10, pady=5)


    ########################################
    #         ROW 4: Error Message         #
    ########################################
    def _print_error(self, msg="Error in selection!"):
        """
        Creates a small text in red when there has been an error in the selection

        :param msg (str): Error message string
        """
        self.error_frame = tk.Frame(self.root)
        self.error_frame.grid(row=4, column=0, columnspan=3, pady=(5, 5))
        
        tk.Label(
            self.error_frame,
            text=msg,
            font=("Arial", 16, "bold"),
            fg="red"
        ).grid(row=0, column=0, padx=10, pady=5)


    ########################################
    #       ROW 5: Submit Info Button      #
    ########################################
    def _sumbit_info_button(self):
        """
        Creates a button to submit the information
        When clicked check all fields have an answer and close the UI
        """ 
        # Define helper functions
        def _on_submit():
            # Get data
            self.task = self.selected_task.get()
            self.data_abspath = self.path_var.get()
            if hasattr(self, "data_type_var"):
                self.data_type = self.data_type_var.get()
            else:
                self.data_type = None

            # Check data correctness
            if (self.task is None) or (self.task == self.behaviors[0]) or (self.data_type is None) or (self.data_abspath == ""):
                self._print_error()
            else:
                self.root.destroy()

        # Submit button
        submit_btn = tk.Button(self.root, text="Submit", command=_on_submit, font=("Arial", 14))
        submit_btn.grid(row=5, column=1, padx=50, pady=20, sticky="se")