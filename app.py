import customtkinter as ctk
from tkinter import filedialog, Canvas
from menu import Menu
from PIL import Image, ImageTk
import cv2
from video_widgets import VideoImport, VideoOutput, CloseOutput  # Assuming video_widgets has these classes
from models import *
from utilities import get_construction_error_from_model, get_transform, get_transform_v2
import torch
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.title("Anomaly Detection")
        self.geometry('1000x600')
        self.resizable(False, False)
        # self.minsize(800, 500)
        self.video_capture = None
        self.video_frame = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ImprovedAutoencoder().to(self.device)
        model_path = "trained_models/autoencoder_model_large_better.pth"
        self.model.load_state_dict(torch.load(model_path))
        self.transform = get_transform()

        self.reconstruction_error = None
        # layout
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=3, uniform="a")
        self.columnconfigure(1, weight=6, uniform="a")

        self.video_import = VideoImport(self, self.import_video)
        self.image_output = VideoOutput(self, self.resize_image)
        self.image_output.grid_forget()  # Hide initially

        self.close_button = CloseOutput(self, self.close_edit)
        self.close_button.place_forget()  # Hide initially

        self.init_parameters()
        self.menu = Menu(self, self.threshold, self.reconstruction_error)

        self.after(20, self.update_video)
        self.mainloop()

    def init_parameters(self):
        self.threshold = ctk.DoubleVar(value=0.005)

    def update_video(self):
        if self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.resize(frame, (800, 800))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(frame)
                self.reconstruction_error = get_construction_error_from_model(self.model, img, self.transform)
                self.menu.update_anomaly_text_and_plot(self.reconstruction_error)
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_frame = img_tk
                self.image_output.delete("all")
                self.image_output.create_image(400, 400, image=img_tk)

        self.after(20, self.update_video)

    def import_video(self, path):
        if path:
            self.video_capture = cv2.VideoCapture(path)
            self.video_import.grid_forget()  # Hide VideoImport
            self.image_output.grid(row=0, column=1, sticky="nsew")  # Show VideoOutput
            self.close_button.place(relx=0.99, rely=0.01, anchor="ne")  # Show Close button
            self.menu.reset_and_show_error_plot()

    def close_edit(self):
        if self.video_capture:
            self.video_capture.release()
        self.video_capture = None
        self.image_output.grid_forget()  # Hide VideoOutput
        self.close_button.place_forget()  # Hide Close button
        self.video_import.grid(column=1, columnspan=1, row=0, sticky="nsew")  # Show VideoImport again
        self.menu.hide_error_plot()
    def resize_image(self, event):
        # If you need to handle resizing, implement logic here
        pass


if __name__ == "__main__":
    App()