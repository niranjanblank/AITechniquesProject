import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class Menu(ctk.CTkFrame):
    def __init__(self, parent, threshold, reconstruction_error):
        super().__init__(master=parent, fg_color="black")
        self.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self,text="Anomaly Detection in Video", font=("Helvetica", 16, "bold")).pack()
        SliderPanel(self, "Threshold Value", threshold)
        self.anomaly_text_panel = AnomalyTextPanel(self, threshold)
        self.error_plot_panel = ErrorPlotPanel(self)
        self.error_plot_panel.pack_forget()  # Hide initially

    def show_error_plot(self):
        self.error_plot_panel.pack()

    def hide_error_plot(self):
        self.error_plot_panel.pack_forget()

    def reset_and_show_error_plot(self):
        self.error_plot_panel.reset_plot()
        self.show_error_plot()

    def update_anomaly_text_and_plot(self, reconstruction_error):
        if reconstruction_error is not None:
            self.anomaly_text_panel.update_text(str(reconstruction_error))
            self.error_plot_panel.update_plot(reconstruction_error)
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


class AnomalyTextPanel(Panel):
    def __init__(self, parent, threshold):
        super().__init__(parent=parent)
        self.threshold = threshold
        self.title_label = ctk.CTkLabel(self, text="Prediction", padx=5)
        self.title_label.grid(row=0, column=0, sticky="W")

        self.value_label = ctk.CTkLabel(self, text="None")
        self.value_label.grid(row=1, column=0, sticky="W", padx=5)
        self.prediction_label = ctk.CTkLabel(self, text="None")
        self.prediction_label.grid(row=2, column=0, sticky="W", padx=5)

    def update_text(self, reconstruction_error):
        if float(reconstruction_error) > self.threshold.get():
            new_text = "Anamolous"
        else:
            new_text = "Non-Anamolous"
        self.prediction_label.configure(text=new_text)
        self.value_label.configure(text=reconstruction_error)


class ErrorPlotPanel(Panel):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.grid_rowconfigure(0, weight=0)  # Configure row for title
        self.grid_rowconfigure(1, weight=1)  # Configure row for plot
        self.grid_columnconfigure(0, weight=1)  # Configure column

        self.title_label = ctk.CTkLabel(self, text="Reconstruction Error")
        self.title_label.grid(row=0, column=0, sticky='w', padx=5)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky='nsew', padx=5)

        self.error_data = []
        plt.tight_layout()

    def update_plot(self, new_error):
        self.error_data.append(new_error)
        self.ax.clear()
        self.ax.plot(self.error_data)
        self.ax.set_ylabel('Reconstruction Error', fontsize=10)
        plt.tight_layout()
        self.canvas.draw()

    def reset_plot(self):
        self.error_data.clear()
        self.ax.clear()
        self.ax.set_ylabel('Reconstruction Error', fontsize=10)
        plt.tight_layout()
        self.canvas.draw()

