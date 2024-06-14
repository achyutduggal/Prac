import os
from threading import Thread
from queue import Queue
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import yaml
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from fsspec import Callback
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

############################################### WORKING CODE FOR TAB 1 ########################################


rectangles = []  # Store rectangles as [(x, y, w, h), ...]
current_rectangle = None  # Current rectangle being drawn or resized
selected_rectangle_idx = None  # Index of the selected rectangle
resize_anchor = None  # Corner being used for resizing
image_list = []
image_index = 0
current_image_label = None
prev_button = None
next_button = None
save_directory = ""
global path_folder, train_folder, val_folder
path_folder = ""
train_folder = ""
val_folder = ""
all_labels = set()


def select_path_folder():
    global path_folder
    path_folder = filedialog.askdirectory()
    if path_folder:
        print(f"Path folder set to: {path_folder}")


def select_train_folder():
    global train_folder
    train_folder = filedialog.askdirectory()
    if train_folder:
        print(f"Train folder set to: {train_folder}")


def select_val_folder():
    global val_folder
    val_folder = filedialog.askdirectory()
    if val_folder:
        print(f"Validation folder set to: {val_folder}")


class LiteralString(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


yaml.add_representer(LiteralString, literal_presenter)


def create_yaml():
    global path_folder, train_folder, val_folder
    if not path_folder or not train_folder or not val_folder:
        print("One or more directory paths are not set. Please set all directory paths first.")
        return

    # Get unique and sorted class names
    class_names = sorted(all_labels)

    yaml_data = {
        'path': path_folder,
        'train': train_folder,
        'val': val_folder,
        'nc': len(class_names)
    }

    yaml_file_path = filedialog.asksaveasfilename(
        title="Save YAML configuration",
        filetypes=[("YAML Files", "*.yaml"), ("All Files", "*.*")],
        defaultextension=".yaml"
    )

    if yaml_file_path:
        with open(yaml_file_path, 'w') as file:
            # Dump all keys except 'names'
            for key, value in yaml_data.items():
                file.write(f"{key}: {value}\n")
            # Manually write the 'names' key with the inline list format
            file.write("names: [")
            file.write(", ".join(f"'{name}'" for name in class_names))
            file.write("]\n")
        print(f"YAML configuration saved to {yaml_file_path}")
    else:
        print("Save operation cancelled.")


def select_save_directory():
    global save_directory
    save_directory = filedialog.askdirectory()
    if save_directory:
        print(f"Save directory set to: {save_directory}")


def on_save():
    global img, image_list, image_index, train_folder
    if img is not None and image_list:
        if not train_folder:
            print("Train directory not set. Please set the directory first.")
            return

        # Check and create an 'annotations' subdirectory in the train folder
        annotations_directory = os.path.join(train_folder, "annotations")
        if not os.path.exists(annotations_directory):
            os.makedirs(annotations_directory)
            print(f"Created annotations directory: {annotations_directory}")

        # Continue with saving the annotations as before
        img_height, img_width = img.shape[:2]
        label_id_mapping = create_label_id_mapping(rectangles)
        yolo_data = convert_to_yolo_format(rectangles, img_width, img_height, label_id_mapping)

        # Create a filename based on the current image
        base_filename = os.path.splitext(os.path.basename(image_list[image_index]))[0]
        annotation_filename = f"{base_filename}.txt"

        # Full path for saving the file inside the annotations directory
        full_path = os.path.join(annotations_directory, annotation_filename)
        save_to_file(yolo_data, full_path)


def save_classes():
    global save_directory
    if not save_directory:
        print("Save directory not set. Please set the directory first.")
        return

    classes = sorted(set(label for *_, label in rectangles if label))
    classes_filename = "classes.txt"
    full_path = os.path.join(save_directory, classes_filename)

    try:
        with open(full_path, 'w') as file:
            for cls in classes:
                file.write(cls + "\n")
        print(f"Classes saved to {full_path}")
    except Exception as e:
        print(f"Error saving classes file: {e}")


def save_to_file(yolo_data, full_path):
    try:
        with open(full_path, 'w') as file:
            for data in yolo_data:
                file.write(data + "\n")
        print(f"Annotations saved to {full_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


def start_rectangle(x, y):
    global current_rectangle
    current_rectangle = [x, y, 0, 0]
    rectangles.append(current_rectangle)


def update_rectangle(x, y):
    if current_rectangle:
        current_rectangle[2] = x - current_rectangle[0]
        current_rectangle[3] = y - current_rectangle[1]
        print(f"Updated rectangle: {current_rectangle}")
        update_canvas()


def finish_rectangle():
    global current_rectangle, all_labels
    if current_rectangle:
        existing_labels = list(all_labels)
        existing_labels.sort()

        label_dialog = LabelDialog(root, "Select a label for the rectangle", existing_labels)
        label = label_dialog.result

        if label:
            current_rectangle.append(label)
            all_labels.add(label)
        else:
            rectangles.pop()

        current_rectangle = None
        update_canvas()


def select_rectangle(x, y):
    global selected_rectangle_idx, resize_anchor
    selected_rectangle_idx = None
    resize_anchor = None
    for i, (rx, ry, rw, rh, label) in enumerate(rectangles):
        corners = get_corner_points(rx, ry, rw, rh)
        for j, corner in enumerate(corners):
            if is_click_inside_circle(x, y, corner):
                selected_rectangle_idx = i
                resize_anchor = j
                return


def is_click_inside_circle(x, y, circle_center):
    cx, cy = circle_center
    distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    return distance < 10  # 10 is the radius of the circle


def get_corner_points(x, y, w, h):
    return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]


def select_folder():
    global image_list, image_index
    folder_path = filedialog.askdirectory()
    if folder_path:
        image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                      file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        image_index = 0
        if image_list:
            load_image(image_list[0])
            update_canvas()


def show_next_image():
    global image_index, image_list
    if image_index < len(image_list) - 1:
        image_index += 1
        load_image(image_list[image_index])
        update_canvas()


def show_prev_image():
    global image_index, image_list
    if image_index > 0:
        image_index -= 1
        load_image(image_list[image_index])
        update_canvas()


def load_image(path):
    global img, img_copy, photo, rectangles, current_rectangle, selected_rectangle_idx
    rectangles = []  # Reset rectangles for new image
    current_rectangle = None
    selected_rectangle_idx = None

    img = cv2.imread(path)
    img_copy = img.copy()
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    image_canvas.itemconfig(image_on_canvas, image=photo)


def update_canvas():
    global photo, img_copy
    img_copy = img.copy()

    for rect in rectangles:
        # Check the length of rect and unpack accordingly
        if len(rect) == 5:
            x, y, w, h, label = rect
        elif len(rect) == 4:
            x, y, w, h = rect
            label = ""  # Default label if none is provided

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for corner in get_corner_points(x, y, w, h):
            cv2.circle(img_copy, corner, 10, (0, 0, 255), -1)

    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)))
    image_canvas.itemconfig(image_on_canvas, image=photo)


def on_canvas_click(event):
    global selected_rectangle_idx, current_rectangle
    x, y = event.x, event.y

    select_rectangle(x, y)

    if selected_rectangle_idx is None:
        start_rectangle(x, y)
        image_canvas.bind("<B1-Motion>", on_canvas_drag)
        image_canvas.bind("<ButtonRelease-1>", on_canvas_release)
    else:
        image_canvas.bind("<B1-Motion>", on_resize_drag)
        image_canvas.bind("<ButtonRelease-1>", on_resize_release)

def on_canvas_double_click(event):
    global selected_rectangle_idx, current_rectangle, rectangles, all_labels
    x, y = event.x, event.y
    select_rectangle(x, y)

    if selected_rectangle_idx is not None:
        existing_labels = sorted(list(all_labels))

        # Open the custom dialog
        label_dialog = LabelDialog(root, "Enter label for the rectangle", existing_labels)
        label = label_dialog.result

        if label:
            # Update the label of the selected rectangle
            if current_rectangle is not None:
                current_rectangle.append(label)
                all_labels.add(label)  # Update the global set of labels
            else:
                rectangles[selected_rectangle_idx] = (*rectangles[selected_rectangle_idx][:-1], label)
                all_labels.add(label)
            print("Label change cancelled.")

        update_canvas()


def on_canvas_drag(event):
    print("Dragging...")
    update_rectangle(event.x, event.y)
    update_canvas()


def on_canvas_release(event):
    update_rectangle(event.x, event.y)
    image_canvas.unbind("<B1-Motion>")
    image_canvas.unbind("<ButtonRelease-1>")
    finish_rectangle()


def on_resize_drag(event):
    if selected_rectangle_idx is not None and resize_anchor is not None:
        resize_rectangle(event.x, event.y)


def on_resize_release(event):
    image_canvas.unbind("<B1-Motion>")
    image_canvas.unbind("<ButtonRelease-1>")


def resize_rectangle(x, y):
    rect = rectangles[selected_rectangle_idx]
    if resize_anchor == 0:  # Top-left corner
        dx, dy = rect[0] - x, rect[1] - y
        rect[0], rect[1] = x, y
        rect[2] += dx
        rect[3] += dy
    elif resize_anchor == 1:  # Top-right corner
        dy = rect[1] - y
        rect[1] = y
        rect[2] = x - rect[0]
        rect[3] += dy
    elif resize_anchor == 2:  # Bottom-left corner
        dx = rect[0] - x
        rect[0] = x
        rect[2] += dx
        rect[3] = y - rect[1]
    elif resize_anchor == 3:  # Bottom-right corner
        rect[2] = x - rect[0]
        rect[3] = y - rect[1]
    update_canvas()


def delete_selected_rectangle():
    global selected_rectangle_idx
    if selected_rectangle_idx is not None:
        del rectangles[selected_rectangle_idx]
        selected_rectangle_idx = None  # Reset the selected index
        update_canvas()


def print_rectangle_coordinates():
    for idx, (x, y, w, h, label) in enumerate(rectangles):
        print(f"Rectangle {idx}: (x: {x}, y: {y}, width: {w}, height: {h}, label: {label})")


def on_key_press(event):
    if event.keysym == 'Delete':
        delete_selected_rectangle()


def create_label_id_mapping(rectangles):
    labels = sorted(set(label for *_, label in rectangles))
    return {label: idx for idx, label in enumerate(labels)}


def convert_to_yolo_format(rectangles, img_width, img_height, label_id_mapping):
    yolo_data = []
    for rect in rectangles:
        x, y, w, h, label = rect
        w_new = abs(w)
        h_new = abs(h)
        class_id = label_id_mapping[label]
        x_center = (x + w_new / 2) / img_width
        y_center = (y + h_new / 2) / img_height
        width = w_new / img_width
        height = h_new / img_height
        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return yolo_data


def save_to_file(yolo_data, default_filename):
    # Ask the user to specify the file path for saving
    file_path = filedialog.asksaveasfilename(
        initialfile=default_filename,
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt")],
        title="Save annotations as"
    )

    # Save the annotations if a file path is provided
    if file_path:
        with open(file_path, 'w') as file:
            for data in yolo_data:
                file.write(data + "\n")
        print(f"Annotations saved to {file_path}")
    else:
        print("Save operation cancelled.")


class LabelDialog(tk.Toplevel):
    def __init__(self, parent, title, existing_labels):
        super().__init__(parent)
        self.title(title)
        self.result = None

        self.label_var = tk.StringVar()
        self.combobox = ttk.Combobox(self, textvariable=self.label_var, values=existing_labels)
        self.combobox.pack(padx=10, pady=10)

        ok_button = tk.Button(self, text="OK", command=self.on_ok)
        ok_button.pack(side="left", padx=(10, 5), pady=10)

        cancel_button = tk.Button(self, text="Cancel", command=self.on_cancel)
        cancel_button.pack(side="right", padx=(5, 10), pady=10)

        self.transient(parent)
        self.grab_set()
        self.wait_window()

    def on_ok(self):
        self.result = self.label_var.get()
        self.destroy()

    def on_cancel(self):
        self.destroy()


############################################### WORKING CODE FOR TAB 1 ########################################

############################################### WORKING CODE FOR TAB 2 ########################################

class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)

    def flush(self):
        pass


loss_queue = Queue()


def on_train_batch_end(trainer, losses, loss_queue):
    loss = trainer.loss.detach().cpu().numpy()
    loss_queue.put(loss)

    info = f"Epoch: {trainer.epoch},  Loss: {loss}"
    root.after(0, lambda: text_widget.insert(tk.END, info + "\n"))


def train_model(data, epochs, batch, lr0, imgsz, losses, loss_queue):
    model = YOLO()
    model.add_callback("on_train_epoch_end", lambda trainer: on_train_batch_end(trainer, losses, loss_queue))
    model.train(data=data, epochs=epochs, batch=batch, lr0=lr0, imgsz=imgsz, task='detect')
    model_variant = r"C:\Users\achyu\Downloads\yolov8n.pt"


def start_training_thread(text_widget, root, ax, line, fig, data, epochs, lr, batch, task, imgsz):
    thread = Thread(target=train_model, args=(text_widget, root, ax, line, fig, data, epochs, lr, batch, task, imgsz),
                    daemon=True)
    thread.start()


def start_training():
    data = data_entry.get()
    epochs = int(epochs_entry.get())
    batch = int(batch_entry.get())
    lr0 = float(lr_entry.get())
    imgsz = int(imgsz_entry.get())

    training_thread = Thread(target=train_model, args=(data, epochs, batch, lr0, imgsz, losses, loss_queue),
                             daemon=True)
    training_thread.start()


def update_plot():
    while not loss_queue.empty():
        loss = loss_queue.get()
        losses.append(loss)
        line.set_data(range(len(losses)), losses)
        ax.relim()
        ax.autoscale_view()

    canvas.draw()
    canvas.flush_events()
    root.after(1000, update_plot)


def select_yaml():
    filename = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
    data_entry.delete(0, tk.END)
    data_entry.insert(0, filename)


def choose_file(entry):
    file_path = filedialog.askopenfilename()
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

root = ttk.Window(themename='litera')
root.title("Image Editor")

# Styling configurations
button_style = 'info.TButton'  # You can choose other styles like 'success', 'danger', etc.
tab_style = 'info.TNotebook'  # Style for the tabs
label_style = 'info.TLabel'  # Style for the labels

# Custom style for the tabs on the side
style = ttk.Style()
style.configure(tab_style + '.Tab', tabposition='wn', padding=[200, 30], background='lightblue')

# Create the notebook with styled tabs
notebook = ttk.Notebook(root, style=tab_style)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)

notebook.add(tab1, text='Labeling')
notebook.add(tab2, text='Training')
notebook.add(tab3, text='Testing')
notebook.pack(fill='both', expand=True)

# Common button configurations
button_config = {'width': 20, 'bootstyle': button_style}

# Tab 1 widgets
btn_select_path_folder = ttk.Button(tab1, text="Select Path Folder", **button_config, command=select_path_folder)
btn_select_path_folder.pack()

btn_select_train_folder = ttk.Button(tab1, text="Select Train Folder", **button_config, command=select_train_folder)
btn_select_train_folder.pack()

btn_select_val_folder = ttk.Button(tab1, text="Select Validation Folder", **button_config, command=select_val_folder)
btn_select_val_folder.pack()

btn_print_coords = ttk.Button(tab1, text="Print coords", **button_config, command=print_rectangle_coordinates)
btn_print_coords.pack()

btn_save = ttk.Button(tab1, text="Save Annotations", **button_config, command=on_save)
btn_save.pack()

btn_select_folder = ttk.Button(tab1, text="Select Folder", **button_config, command=select_folder)
btn_select_folder.pack()

btn_prev = ttk.Button(tab1, text="<< Prev", **button_config, command=show_prev_image)
btn_prev.pack(side=tk.LEFT)

btn_next = ttk.Button(tab1, text="Next >>", **button_config, command=show_next_image)
btn_next.pack(side=tk.RIGHT)

btn_select_save_dir = ttk.Button(tab1, text="Select Save Directory", **button_config, command=select_save_directory)
btn_select_save_dir.pack()

btn_save_classes = ttk.Button(tab1, text="Save Classes", **button_config, command=save_classes)
btn_save_classes.pack()

btn_create_yaml = ttk.Button(tab1, text="Create YAML", **button_config, command=create_yaml)
btn_create_yaml.pack()

# Create a canvas for image display in tab1
image_canvas = tk.Canvas(tab1, width=600, height=400)
image_canvas.pack()
image_on_canvas = image_canvas.create_image(0, 0, anchor=tk.NW)

image_canvas.bind("<Button-1>", on_canvas_click)
image_canvas.bind("<Double-Button-1>", on_canvas_double_click)
root.bind("<KeyPress>", on_key_press)

############################################# Tab 2 widgets ##################################
main_frame = ttk.Frame(tab2)
main_frame.pack(fill=tk.BOTH, expand=True)

# Matplotlib plot for loss
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
losses = []


tk.Label(main_frame, text="Data:").grid(row=0, column=0, sticky="e")
data_entry = ttk.Entry(main_frame)
data_entry.grid(row=0, column=1, padx=5, pady=5)
select_yaml_button = ttk.Button(main_frame, text="Browse", command=select_yaml)
select_yaml_button.grid(row=0, column=2, padx=5, pady=5)

text_widget = tk.Text(main_frame, height=10, width=50)
text_widget.grid(row=6, column=0, columnspan=3, padx=10, pady=10)


canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas_widget = canvas.get_tk_widget()

canvas_widget.grid(row=0, column=0, sticky='nsew')

# Configure the grid behavior in the frame
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

data_entry = ttk.Entry(main_frame)
data_entry.grid(row=0, column=1, padx=5, pady=5)
select_yaml_button = ttk.Button(main_frame, text="Browse", command=select_yaml)
select_yaml_button.grid(row=0, column=2, padx=5, pady=5)

ttk.Label(main_frame, text="Epochs:").grid(row=1, column=0, sticky="e")
epochs_entry = ttk.Entry(main_frame)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(main_frame, text="Learning Rate:").grid(row=2, column=0, sticky="e")
lr_entry = ttk.Entry(main_frame)
lr_entry.grid(row=2, column=1, padx=5, pady=5)

ttk.Label(main_frame, text="Batch Size:").grid(row=3, column=0, sticky="e")
batch_entry = ttk.Entry(main_frame)
batch_entry.grid(row=3, column=1, padx=5, pady=5)

ttk.Label(main_frame, text="Task:").grid(row=4, column=0, sticky="e")
task_combobox = ttk.Combobox(main_frame, values=["detect", "classify"])
task_combobox.grid(row=4, column=1, padx=5, pady=5)
task_combobox.set("detect")

ttk.Label(main_frame, text="Image Size:").grid(row=5, column=0, sticky="e")
imgsz_entry = ttk.Entry(main_frame)
imgsz_entry.grid(row=5, column=1, padx=5, pady=5)

# Text widget for logs
text_widget = tk.Text(main_frame, height=10, width=50)
text_widget.grid(row=6, column=0, columnspan=3, padx=10, pady=10)

# Button to start training
root.after(1000, update_plot)

start_button = ttk.Button(main_frame, text="Start Training", command=start_training)

start_button.grid(row=7, column=5, pady=10)

####################################### TAB 3 Widgets ############################################

model_frame = ttk.Frame(tab3)
model_frame.pack(padx=10, pady=10, fill='x')
ttk.Label(model_frame, text="Model File (.pt):").pack(side=tk.LEFT, padx=5)
model_entry = ttk.Entry(model_frame)
model_entry.pack(side=tk.LEFT, expand=True, fill='x', padx=5)
model_button = ttk.Button(model_frame, text="Browse", command=lambda: choose_file(model_entry))
model_button.pack(side=tk.LEFT, padx=5)

# Image file selection
image_frame = ttk.Frame(tab3)
image_frame.pack(padx=10, pady=10, fill='x')
ttk.Label(model_frame, text="Image File:").pack(side=tk.LEFT, padx=5)
image_entry = ttk.Entry(model_frame)
image_entry.pack(side=tk.LEFT, expand=True, fill='x', padx=5)
image_button = ttk.Button(model_frame, text="Browse", command=lambda: choose_file(image_entry))
image_button.pack()

result_label = tk.Label(model_frame, text="Result will be shown here")
result_label.pack()

# Test button
def test_model(model, image):
    model = YOLO(model)
    outs = model.predict(image, save=True)
    result_label.config(outs)


test_button = ttk.Button(model_frame, text="Test Model", bootstyle=PRIMARY,
                          command=lambda: test_model(model_entry.get(), image_entry.get()))
test_button.pack(padx=10, pady=10)

# Run the application
root.mainloop()
