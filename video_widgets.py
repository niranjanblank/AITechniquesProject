import customtkinter as ctk
from tkinter import filedialog, Canvas

class VideoImport(ctk.CTkFrame):
    def __init__(self, parent, import_func):
        super().__init__(master=parent)
        self.grid(column=1, columnspan=1,row=0, sticky="nsew")
        self.import_func = import_func

        ctk.CTkButton(self, text="open video", command=self.open_dialog).pack(expand=True)

    def open_dialog(self):
        path = filedialog.askopenfile().name
        self.import_func(path)

class VideoOutput(Canvas):
    def __init__(self,parent, resize_image):
        super().__init__(master=parent, background="#2B2B2B", bd=0, highlightthickness=0,relief="ridge")
        self.grid(row=0, column=1, sticky="nsew")
        self.bind('<Configure>', resize_image)

class CloseOutput(ctk.CTkButton):
    def __init__(self,parent, close_func):
        super().__init__(master=parent, text="X", text_color="white", fg_color="transparent", width=40, height=40,
                         hover_color="red",
                         command=close_func
                         )
        self.place(relx=0.99, rely=0.01, anchor="ne")