# import os
# import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import to_rgb
# from matplotlib.patches import Rectangle

from PIL import Image

from skimage.color import lab2rgb, rgb2lab

from sklearn.cluster import KMeans

# from matplotlib.widgets import Button
# import matplotlib.gridspec as gridspec
# import matplotlib

from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox#, draw_box_on_image, plot_results

class SAM3UI:
    def __init__(self, processor, img_path, prompt, use_bbox_input=False):
        # Model
        self.processor = processor
        self.prompt = prompt
        self.use_bbox_input = use_bbox_input
        
        # Image
        self.image_rgb = Image.open(img_path).convert("RGB")
        self.width, self.height = self.image_rgb.size
        self.img_path = img_path
        
        # States
        self.press_event = None
        self.box_patch = None
        self.box_final = None
        self.box_done = False
        self.points = []

        self.masks = None
        self.scores = None
        self.decision = "n"

        # Variables
        self.current_mask = 0
        self.mask_abspath = 0

    
    ########################################
    #             Execute SAM3             #
    ########################################
    def _prompt_model(self):
        """
        Run the model with a text prompt
        :return: The model output and the inference state
        """
        inference_state = self.processor.set_image(self.image_rgb)
        output = self.processor.set_text_prompt(state=inference_state, prompt=self.prompt)

        # Get the masks, bounding boxes, and scores
        return output, inference_state


    def _box_prompt_model(self, box_input_list, box_labels):
        """
        Run the model with bounding box input(s) and the text prompt
        
        :param box_input_list: Array of bounding box inputs in format xywh
        :param box_labels: Array of labels for each bounding box (inclusive -> [True], exclusive -> [False])
        :return: The model output and the inference state
        """
        box_input_cxcywh = box_xywh_to_cxcywh(torch.tensor(box_input_list).view(-1,4))
        norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, self.width, self.height).tolist()

        for box, label in zip(norm_boxes_cxcywh, box_labels):
            inference_state = self.processor.add_geometric_prompt(
                state=inference_state, box=box, label=label
            )
        output = self.processor.set_text_prompt(state=inference_state, prompt=self.prompt)

        # Get the masks, bounding boxes, and scores
        return output, inference_state
    

    def process_image(self):
        """
        Processes the image with the SAM3 model and returns outputs and plots
        
        :param self: Description
        """
        # --- USE UI LAYOUT WITH SINGLE FIGURE SETTING --- #
        if not self.use_bbox_input:
            # Call the model
            return self._prompt_model()

        # --- USE UI LAYOUT WITH GRIDSPEC SETTING --- #
    

    ########################################
    #             Plot Results             #
    ########################################
    @staticmethod
    def _generate_colors(n_colors=256, n_samples=5000):
        """
        Generates the colors by K-Means
        :param n_colors: Number of colors
        :param n_samples: Number of samples
        """
        # Step 1: Random RGB samples
        np.random.seed(42)
        rgb = np.random.rand(n_samples, 3)
        # Step 2: Convert to LAB for perceptual uniformity
        # print(f"Converting {n_samples} RGB samples to LAB color space...")
        lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
        # print("Conversion to LAB complete.")
        # Step 3: k-means clustering in LAB
        kmeans = KMeans(n_clusters=n_colors, n_init=10)
        # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
        kmeans.fit(lab)
        # print("KMeans fitting complete.")
        centers_lab = kmeans.cluster_centers_
        # Step 4: Convert LAB back to RGB
        colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
        colors_rgb = np.clip(colors_rgb, 0, 1)
        return colors_rgb
    

    @staticmethod
    def _plot_mask(mask, color="r", ax=None):
        """
        Plots the mask(s) given the color
        :param mask: Mask(s) array
        :param color: Color to use for the mask(s)
        :param ax: Axis to use
        """
        mask_h, mask_w = mask.shape
        mask_img = np.zeros((mask_h, mask_w, 4), dtype=np.float32)
        mask_img[..., :3] = to_rgb(color)
        mask_img[..., 3] = mask * 0.5
        # Use the provided ax or the current axis
        if ax is None:
            ax = plt.gca()
        ax.imshow(mask_img)
    

    @staticmethod
    def _plot_bbox(img_height, img_width, box, box_format="XYXY", relative_coords=True,
                  color="r", linestyle="solid", text=None, ax=None,):
        """
        Plot bounding box
        :param img_height: Image height
        :param img_width: Image width
        :param box: Box coordinates
        :param box_format: Format of the box coordinates
        :param relative_coords: Use relative coordinates
        :param color: Color of the box
        :param linestyle: Linestyle of the box
        :param text: Format of the box text
        :param ax: Axis to use
        """
        if box_format == "XYXY":
            x, y, x2, y2 = box
            w = x2 - x
            h = y2 - y
        elif box_format == "XYWH":
            x, y, w, h = box
        elif box_format == "CxCyWH":
            cx, cy, w, h = box
            x = cx - w / 2
            y = cy - h / 2
        else:
            raise RuntimeError(f"Invalid box_format {box_format}")

        if relative_coords:
            x *= img_width
            w *= img_width
            y *= img_height
            h *= img_height

        if ax is None:
            ax = plt.gca()
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color,
                                 facecolor="none", linestyle=linestyle,)
        ax.add_patch(rect)
        if text is not None:
            facecolor = "w"
            ax.text(x, y - 5, text, color=color, weight="bold", fontsize=8,
                bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},)
    

    def plot_objects(self, results):
        """
        Draws masks and waits for keyboard input

        :param fig: Matplotlib figure with the instruction panel information
        :param output: Ouput results of the model processor

        :return: The resulting action of the keyboard -> ['1', '2' or '3']
        """
        # Create figureS
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.imshow(self.image_rgb)
        plt.axis("off")

        # Instruction box
        instructions = ("Keyboard actions:\n"
                        "1 → Correct - Save mask & Next image\n"
                        "2 → Incorrect - Save mask & Next image\n"
                        "3 → Incorrect - Next image")
        ax.text(0.01, 0.99, instructions, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        # Generate colors
        colors = self._generate_colors(n_colors=256, n_samples=5000)

        # Plot all mask objects
        nb_objects = len(results["scores"])
        for i in range(nb_objects):
            color = colors[i % len(colors)]
            self._plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
            prob = results["scores"][i].item()
            self._plot_bbox(
                self.height,
                self.width,
                results["boxes"][i].cpu(),
                text=f"(id={i}, {prob=:.2f})",
                box_format="XYXY",
                color=color,
                relative_coords=False,
            )
        
        # Keyboard handler
        action_result = "3"
        def on_key(event):
            nonlocal action_result
            key = event.key.lower()
            if key in ["1", "2", "3"]:
                action_result = key
                plt.close()
            else:
                print("Wrong key! Press '1', '2', or '3'.")
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        return action_result
    

    ########################################
    #                Misc UI               #
    ########################################
    def close_ui(self, decision):
        self.decision = decision
        plt.close(self.fig)






















































        # else:
        #     fig = plt.figure(figsize=(12, 10))

        #     gs = gridspec.GridSpec(
        #         nrows=2, ncols=1,
        #         height_ratios=[4, 1],
        #         figure=fig
        #     )

        #     # -- Row 1: Image --
        #     ax_img = fig.add_subplot(gs[0])
        #     ax_img.imshow(img)
        #     ax_img.axis("off")

        #     # -- Row 2: Instruction panel --
        #     ax_info = fig.add_subplot(gs[1])
        #     ax_info.axis("off")
        #     ax_info.text(
        #         0.02, 0.98,
        #         "Keyboard actions:\n"
        #         "1 → Correct & Save\n"
        #         "2 → Incorrect & Save\n"
        #         "3 → Incorrect (skip)",
        #         fontsize=14,
        #         verticalalignment='top'
        #     )

        #     return self._plot_objects(fig, results=output)



#     gridspec = self.fig.add_gridspec(
        #         2, 2,
        #         width_ratios=[1, 1.2],
        #         height_ratios=[0.15, 1]
        #     )

        #     # Row 0, Col 0 → Instructions
        #     self.ax_instructions = self.fig.add_subplot(gridspec[0, 0])
        #     self.ax_instructions.axis("off")
        #     self.ax_instructions.text(
        #         0.5, 0.5,
        #         "1 - Drag to draw bounding box where the flower is\n"
        #         "2 - Click twice in different regions of the flower\n",
        #         ha='center', va='center', fontsize=12
        #     )

        #     # Row 0, Col 1 → Buttons placeholder
        #     self.ax_mask_buttons = self.fig.add_subplot(gs[0, 1])
        #     self.ax_mask_buttons.axis("off")

        #     # Row 1, Col 0 → Original image
        #     self.ax_left = self.fig.add_subplot(gs[1, 0])
        #     self.ax_left.imshow(self.img_rgb)
        #     self.ax_left.axis("off")

        #     # Row 1, Col 1 → Mask display
        #     self.ax_right = self.fig.add_subplot(gs[1, 1])
        #     self.ax_right.imshow(self.img_rgb)
        #     self.ax_right.set_title("Mask Result")
        #     self.ax_right.axis("off")
        
        # # event handlers
        # self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        # self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        # self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        # self.cid_click_points = self.fig.canvas.mpl_connect('button_press_event', self.on_click_points)

        # plt.tight_layout()
        # plt.show()



    # #  Drag box
    # def on_press(self, event):
    #     if self.box_done:  
    #         return
    #     if event.inaxes != self.ax_left:
    #         return

    #     self.press_event = event
    #     if self.box_patch is not None:
    #         self.box_patch.remove()

    #     self.box_patch = Rectangle((event.xdata, event.ydata),
    #                                0, 0, fill=False, edgecolor='lime', linewidth=2)
    #     self.ax_left.add_patch(self.box_patch)
    #     self.fig.canvas.draw()

    # def on_drag(self, event):
    #     if not self.press_event or self.box_done:
    #         return
    #     if event.inaxes != self.ax_left:
    #         return

    #     x0, y0 = self.press_event.xdata, self.press_event.ydata
    #     x1, y1 = event.xdata, event.ydata

    #     xmin, xmax = min(x0, x1), max(x0, x1)
    #     ymin, ymax = min(y0, y1), max(y0, y1)

    #     self.box_patch.set_xy((xmin, ymin))
    #     self.box_patch.set_width(xmax - xmin)
    #     self.box_patch.set_height(ymax - ymin)

    #     self.fig.canvas.draw()

    # def on_release(self, event):
    #     if not self.press_event or self.box_done:
    #         return
    #     if event.inaxes != self.ax_left:
    #         return

    #     x0, y0 = self.press_event.xdata, self.press_event.ydata
    #     x1, y1 = event.xdata, event.ydata

    #     xmin, xmax = int(min(x0, x1)), int(max(x0, x1))
    #     ymin, ymax = int(min(y0, y1)), int(max(y0, y1))

    #     self.box_final = np.array([xmin, ymin, xmax, ymax])
    #     self.box_done = True
    #     self.press_event = None

    # # Point selection
    # def on_click_points(self, event):
    #     if not self.box_done:
    #         return
    #     if len(self.points) >= 2:
    #         return
    #     if event.inaxes != self.ax_left:
    #         return

    #     px, py = int(event.xdata), int(event.ydata)

    #     self.points.append([px, py])

    #     self.ax_left.scatter(px, py, color='yellow', marker='*', s=250, edgecolor='black')
    #     self.fig.canvas.draw()

    #     # Once 2 points clicked → run SAM2 and show mask viewer
    #     if len(self.points) == 2:
    #         self.run_sam2_and_show_masks()

    # # SAM2 mask computation + UI
    # def run_sam2_and_show_masks(self):
    #     input_point = np.array(self.points)
    #     input_label = np.array([1, 1])

    #     self.predictor.set_image(self.img_rgb)

    #     masks, scores, _ = self.predictor.predict(
    #         point_coords=input_point,
    #         point_labels=input_label,
    #         box=self.box_final,
    #         multimask_output=True
    #     )
    #     # Sort masks in order of likelihood
    #     sorted_ind = np.argsort(scores)[::-1]
    #     masks_sorted = masks[sorted_ind]
    #     scores_sorted = scores[sorted_ind]

    #     self.masks = masks_sorted
    #     self.scores = scores_sorted

    #     # add mask buttons
    #     self.add_mask_buttons()

    #     # show mask 1 by default
    #     self.display_mask(0)

    #     self.add_decision_buttons()

    # def add_decision_buttons(self):
    #     btn_positions = [
    #         [0.7, 0.05, 0.1, 0.05],  # Correct & Save
    #         [0.82, 0.05, 0.1, 0.05], # Incorrect & Save
    #         [0.94, 0.05, 0.05, 0.05] # Discard
    #     ]
        
    #     if hasattr(self, "decision_buttons"):
    #         for btn in self.decision_buttons:
    #             btn.ax.remove()
    #     self.decision_buttons = []

    #     def correct_save(event):
    #         self.save_mask_to_folder("correct_masks")
    #         print("Labeled as correct and saving")
    #         self.close_ui(1)

    #     def incorrect_save(event):
    #         self.save_mask_to_folder("incorrect_masks")
    #         print("Labeled as incorrect and saving")
    #         self.close_ui(0)

    #     def discard(event):
    #         print("Labeled as incorrect and discarding")
    #         self.close_ui("n")

    #     callbacks = [correct_save, incorrect_save, discard]
    #     labels = ["Correct & Save", "Incorrect & Save", "Discard"]

    #     for pos, cb, label in zip(btn_positions, callbacks, labels):
    #         ax_btn = self.fig.add_axes(pos)
    #         b = Button(ax_btn, label)
    #         b.on_clicked(cb)
    #         self.decision_buttons.append(b)

    #     self.fig.canvas.draw()

    # def save_mask_to_folder(self, folder_name):
    #     """
    #     Save the highest-scoring mask in the specified folder
    #     in the same binary format as your previous save_mask function.
    #     Returns the absolute path of the saved mask.
    #     """
    #     if self.masks is None:
    #         print("No mask to save!")
    #         return None

    #     save_dir = os.path.join(os.path.dirname(self.img_path), folder_name)
    #     os.makedirs(save_dir, exist_ok=True)

    #     mask = self.masks[self.current_mask]  # save current mask
    #     self.mask_abspath = os.path.join(save_dir, os.path.basename(self.img_path))

    #     # Ensure NumPy array
    #     if isinstance(mask, torch.Tensor):
    #         mask_np = mask.detach().cpu().numpy()
    #     elif isinstance(mask, np.ndarray):
    #         mask_np = mask
    #     else:
    #         raise TypeError("Mask must be a PyTorch tensor or NumPy array.")

    #     mask_np = np.squeeze(mask_np)
    #     mask_img = (mask_np > 0).astype(np.uint8) * 255

    #     Image.fromarray(mask_img).save(self.mask_abspath)

    

    # # Buttons for mask selection
    # def add_mask_buttons(self):
    #     self.ax_mask_buttons.clear()
    #     self.ax_mask_buttons.axis("off")

    #     btn_positions = [0.5, 0.65, 0.8]
    #     self.buttons = []

    #     for i in range(3):
    #         ax_btn = self.fig.add_axes([btn_positions[i], 0.93, 0.15, 0.05])
    #         b = Button(ax_btn, f"Mask {i+1}")
    #         b.on_clicked(lambda _, j=i: self.display_mask(j))
    #         self.buttons.append(b)

    #     self.fig.canvas.draw()

    # # Display mask on right panel
    # def display_mask(self, idx):
        
    #     # Update current mask
    #     self.current_mask = idx

    #     # Plot settings
    #     self.ax_right.clear()
    #     self.ax_right.imshow(self.img_rgb)
    #     self.ax_right.set_title(f"Mask {idx+1} (score={self.scores[idx]:.3f})")
    #     self.ax_right.axis("off")

    #     # Mask processing
    #     mask = self.masks[idx].astype(np.uint8)
    #     h, w = mask.shape[-2:]
    #     color = np.array([30/255, 144/255, 255/255, 0.6])
    #     mask_rgba = mask.reshape(h, w, 1) * color.reshape(1, 1, 4)

    #     # Smooth contours
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
    #     mask_rgba = cv2.drawContours(mask_rgba.copy(), contours, -1, (1, 1, 1, 0.5), thickness=2)

    #     self.ax_right.imshow(mask_rgba)
    #     self.fig.canvas.draw()
