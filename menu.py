import customtkinter as ctk

class Menu(ctk.CTkFrame):
    def __init__(self, parent, threshold):
        super().__init__(master=parent, fg_color="black")
        self.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self,text="Anomaly Detection in Video", font=("Helvetica", 16, "bold")).pack()
        SliderPanel(self, "Threshold Value", threshold)

class Panel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color="#242424")
        self.pack(fill="x", pady=4, ipady=8, padx=4)

class SliderPanel(Panel):
    def __init__(self, parent, text, threshold):
        super().__init__(parent=parent)

        self.rowconfigure((0,1), weight=1)
        self.columnconfigure((0,1), weight=1)
        ctk.CTkLabel(self, text=text).grid(row=0, column=0, sticky="W", padx=5)
        self.num_label = ctk.CTkLabel(self, text= threshold.get())
        self.num_label.grid(row=0, column=1, sticky="E", padx=5)
        ctk.CTkSlider(self, variable=threshold, from_=0.001, to=0.01,
                      command=self.update_text
                      ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        # ctk.CTkEntry(self, textvariable=threshold).grid(row=1, column=0, columnspan=2, sticky="ew", pady=5, padx=5)

    def update_text(self, value):
        self.num_label.configure(text=f'{round(value,5)}')