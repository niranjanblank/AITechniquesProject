import customtkinter as ctk
from video_widgets import *
from PIL import Image, ImageTk
from menu import Menu
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.geometry('1000x600')
        self.minsize(800, 500)

        # layout
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=2, uniform="a")
        self.columnconfigure(1, weight=6, uniform="a")

        self.video_import = VideoImport(self, self.import_video)
        self.init_parameters()

        self.menu = Menu(self, self.threshold)
        self.mainloop()

    def init_parameters(self):
        self.threshold = ctk.DoubleVar(value=0.00115)
        # self.threshold.trace('w', self.)
    def import_video(self, path):
        self.image = Image.open(path)
        self.image_tk = ImageTk.PhotoImage(self.image)

        self.video_import.grid_forget()
        self.image_output = VideoOutput(self, self.resize_image)
        self.close_button = CloseOutput(self, self.close_edit)


    def close_edit(self):

        self.image_output.grid_forget()
        self.close_button.place_forget()

        self.video_import = VideoImport(self, self.import_video)

    def resize_image(self, event):

        #place image
        self.image_output.delete('all')
        self.image_output.create_image(event.width/2,event.height/2,image = self.image_tk )

if __name__=="__main__":
    App()
