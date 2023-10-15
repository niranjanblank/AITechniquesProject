import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

# Placeholder function for processing frames
def process_frame(frame):
    # simulated model output
    return "Normal" if (frame // 60) % 2 == 0 else "Anomalous"


class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player App")

        self.video_path = ""
        self.video_capture = None
        self.frame_width = 640
        self.frame_height = 480

        self.video_frame = tk.Frame(root, width=self.frame_width, height=self.frame_height)
        self.video_frame.pack()

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.label_frame = tk.Frame(root)
        self.label_frame.pack()

        self.status_label = tk.Label(self.label_frame, text="Status Placeholder", font=("Arial", 16), background="black")
        self.status_label.pack()

        self.btn_open = tk.Button(self.label_frame, text="Open Video", command=self.open_file)
        self.btn_open.pack()

        self.btn_play = tk.Button(self.label_frame, text="Play", command=self.play_video)
        self.btn_play.pack()

    def open_file(self):
        self.video_path = filedialog.askopenfilename()

    def process_and_update_status(self, frame_count):
        label = process_frame(frame_count)
        self.status_label.config(text=label)
        if label == "Normal":
            self.status_label.config(fg="green")
        else:
            self.status_label.config(fg="red")

    def play_video(self):
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            frame_count = 0
            while self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret:
                    frame_count += 1
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img_tk = ImageTk.PhotoImage(image=img)
                    self.video_label.img_tk = img_tk
                    self.video_label.config(image=img_tk)
                    self.root.update()
                    self.process_and_update_status(frame_count)
                else:
                    break
            self.video_capture.release()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = VideoPlayerApp(root)
    root.mainloop()