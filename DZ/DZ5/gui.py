from ast import literal_eval
from copy import deepcopy
import csv
import numpy as np
import os

from tkinter import Canvas, Menu
from tkinter.ttk import Frame, Label, Button, Menubutton, Notebook

from util import normalize_gesture


class GestureWindow(Frame):
    labels = ["α", "β", "γ", "δ", "ε"]

    def __init__(self, root_window, evaluation_model=None, number_of_reference_points: int = 20):
        super().__init__()

        self.__root_window = root_window
        self.__recorded_gesture_index = 0
        self.__recorded_gesture_label = None

        self.__evaluation_model = evaluation_model
        self.__number_of_reference_points = number_of_reference_points

        self.__classification_outputs = None
        self.__results = [0.] * len(self.labels)

        self.__record_dict =\
            {
                0: list(),
                1: list(),
                2: list(),
                3: list(),
                4: list(),
            }

        if not os.path.exists("data"):
            os.mkdir("data")

        if not os.path.exists("data/dataset.tsv"):
            f = open("data/dataset.tsv", mode="w+")
            f.close()

        self.load_data()

        self.initialize()

    def initialize(self):
        self.master.title("NENR DZ5 - Miljenko Šuflaj")
        self.pack(fill="both", expand=True)

        notebook = Notebook(self)
        notebook.pack()

        record_gestures_frame = Frame(self)
        classify_gestures_frame = Frame(self)

        notebook.add(record_gestures_frame, text="Zabilježi geste")
        notebook.add(classify_gestures_frame, text="Klasificiraj geste")

        notebook.enable_traversal()
        notebook.select(record_gestures_frame)

        gesture_classification_frame = Frame(classify_gestures_frame)
        gesture_record_frame_2 = Frame(classify_gestures_frame)

        gesture_classification_frame.pack(side="top", fill="x", expand=True)
        gesture_record_frame_2.pack(fill="both")

        # region Record gesture tab
        gesture_info_frame = Frame(record_gestures_frame)
        gesture_record_frame = Frame(record_gestures_frame)
        gesture_end_frame = Frame(record_gestures_frame)
        gesture_info_frame.pack(side="top", fill="x")
        gesture_record_frame.pack(fill="both")
        gesture_end_frame.pack(side="bottom", fill="x")

        gesture_name_menu = Menubutton(gesture_info_frame, text="Odaberi gestu za bilježenje...")
        gesture_name_menu.menu = Menu(gesture_name_menu, tearoff=0)
        gesture_name_menu["menu"] = gesture_name_menu.menu

        for i, label in enumerate(GestureWindow.labels):
            gesture_name_menu.menu.add_command(label=label, command=self.switch_record(i))

        self.__recorded_gesture_label = Label(gesture_info_frame)
        gesture_record_canvas = Canvas(gesture_record_frame, width=700, height=475, bg="white")
        gesture_record_canvas.current_coords = None
        gesture_record_canvas.records = list()
        gesture_record_save_button = Button(gesture_end_frame, text="Pohrani u .tsv", command=self.save_data)
        gesture_record_delete_button = Button(gesture_end_frame,
                                              text="Zaboravi učitane podatke", command=self.delete_internal)

        gesture_name_menu.pack(side="left", fill="x", padx=5, pady=5, expand=True)
        self.__recorded_gesture_label.pack(side="left", fill="x", padx=5, pady=5)
        gesture_record_canvas.pack(fill="both", padx=5, pady=5, expand=True)
        gesture_record_delete_button.pack(side="left", fill="x", padx=5, pady=5)
        gesture_record_save_button.pack(side="right", fill="x", padx=5, pady=5)

        self.switch_record(self.__recorded_gesture_index)()
        # endregion

        # region Classify gesture tab
        gesture_classification_frame = Frame(classify_gestures_frame)
        gesture_record_frame_2 = Frame(classify_gestures_frame)

        gesture_classification_frame.pack(side="top", fill="x", expand=True)
        gesture_record_frame_2.pack(fill="both")

        classification_labels = list()
        self.__classification_outputs = list()

        for category in GestureWindow.labels:
            classification_labels.append(Label(gesture_classification_frame, text=category))
            self.__classification_outputs.append(Label(gesture_classification_frame,
                                                       text=f"{0.:.01f}%", font=("helvetica", 8)))
            classification_blank = Label(gesture_classification_frame)

            classification_labels[-1].pack(side="left", fill="x", padx=5, pady=5)
            self.__classification_outputs[-1].pack(side="left", fill="x", padx=5, pady=5)
            classification_blank.pack(side="left", fill="x", padx=50, pady=5)

        gesture_record_canvas_2 = Canvas(gesture_record_frame_2, width=700, height=525, bg="white")
        gesture_record_canvas_2.current_coords = None
        gesture_record_canvas_2.records = list()

        gesture_record_canvas_2.pack(side="left", fill="both", padx=5, pady=5, expand=True)
        # endregion

        # region Functionality
        for record_canvas in [gesture_record_canvas, gesture_record_canvas_2]:
            draw_function = self.get_draw_on_canvas_function(record_canvas)

            record_canvas.bind("<B1-Motion>", draw_function)
            record_canvas.bind("<ButtonRelease-1>", draw_function)

        self.__root_window.bind("<BackSpace>", self.get_delete_currently_drawn(gesture_record_canvas))
        self.__root_window.bind("<Return>", self.get_record_gesture_function(gesture_record_canvas))

        gesture_record_canvas_2.bind("<Leave>", self.get_evaluate_gesture_function(gesture_record_canvas_2))
        self.__root_window.bind("<Delete>", self.get_delete_currently_drawn(gesture_record_canvas_2))
        # endregion

    def load_data(self, path: str = "data/dataset.tsv"):
        with open(path, mode="r", newline="") as file:
            reader = csv.reader(file, delimiter="\t")

            for row in reader:
                self.__record_dict[literal_eval(row[1])].append(literal_eval(row[0]))

    def save_data(self, path: str = "data/dataset.tsv"):
        with open(path, mode="w+", newline="") as file:
            writer = csv.writer(file, delimiter="\t")

            for y, entries in self.__record_dict.items():
                for x in entries:
                    writer.writerow((x, y))

    def delete_internal(self):
        for key in self.__record_dict:
            self.__record_dict[key].clear()

        self.switch_record(self.__recorded_gesture_index)()

    def switch_record(self, record_id: int):
        def _y():
            self.__recorded_gesture_index = record_id
            self.__recorded_gesture_label["text"] = f"Bilježim gestu {self.labels[self.__recorded_gesture_index]} " \
                                                    f"(trenutno zabilježeno njih " \
                                                    f"{len(self.__record_dict[self.__recorded_gesture_index])})"

        return _y

    def switch_model(self, model):
        self.__evaluation_model = model

    def update_results(self):
        maximum_index = np.argmax(self.__results)

        if self.__results[maximum_index] < 1e-6:
            maximum_index = -1

        for i in range(len(self.__results)):
            self.__classification_outputs[i]["text"] = f"{self.__results[i]:.02f}%"
            self.__classification_outputs[i]["font"] = ("helvetica", 8)

            if i == maximum_index:
                self.__classification_outputs[i]["font"] = ("helvetica", 12, "bold")

    @staticmethod
    def get_draw_on_canvas_function(canvas: Canvas):
        def draw_function(event):
            event_type = str(event.type)
            new_coords = event.x, event.y

            if event_type == "Motion":
                canvas.records.append(new_coords)

                if canvas.current_coords is not None and canvas.current_coords != new_coords:
                    canvas.create_line(*new_coords, *canvas.current_coords)

                canvas.current_coords = new_coords
            else:
                canvas.current_coords = None

        return draw_function

    def get_delete_currently_drawn(self, canvas: Canvas):
        def delete_function(event):
            if canvas.records is not None and len(canvas.records) != 0:
                canvas.records.clear()
                canvas.current_coords = None

                canvas.delete("all")

                self.__results = [0.] * len(self.labels)
                self.update_results()

        return delete_function

    def get_record_gesture_function(self, canvas: Canvas):
        def record_function(_):
            if canvas.records is not None and len(canvas.records) != 0:
                self.__record_dict[self.__recorded_gesture_index].append(deepcopy(canvas.records))
                canvas.records.clear()

                self.switch_record(self.__recorded_gesture_index)()

            canvas.delete("all")

        return record_function

    def get_evaluate_gesture_function(self, canvas: Canvas):
        def evaluate_gesture_function(_):
            if canvas.records is not None and len(canvas.records) != 0:
                gesture = normalize_gesture(canvas.records, self.__number_of_reference_points)

                if self.__evaluation_model is None:
                    self.__results = [0.] * len(self.labels)
                else:
                    result = self.__evaluation_model.predict(gesture)
                    self.__results = [round(x * 100, 2) for x in result]

                self.update_results()

        return evaluate_gesture_function
