from hyformulas import *
from tkinter import * 
import tkinter as tk
from tkinter import ttk
import webbrowser
import tkinter.messagebox as mb
import time
import tkinter.simpledialog as sd
import yaml

# Initialize coordinates dictionary
coordinates = {}

root = tk.Tk()
root.title("Yardage Book Generator")

window = tk.Frame(master=root)
window.pack(fill=tk.BOTH, expand=True)

frm_title = tk.Frame(master=window)
frm_title.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

frm_coords = tk.Frame(master=window)
frm_coords.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

frm_dropdown = tk.Frame(master=frm_title)
frm_dropdown.grid(row=1, column=0, sticky="w")

frm_options = tk.Frame(master=window, height=100)
frm_options.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

frm_options.columnconfigure(0, weight=1, minsize=200)
frm_options.columnconfigure(1, weight=1, minsize=200)
frm_options.rowconfigure(0, weight=1, minsize=200)

frm_button = tk.Frame(master=window, height=20)
frm_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Load coordinates from YAML file
def load_coordinates():
    try:
        with open("coordinates.yml", "r") as file:
            coordinates.update(yaml.safe_load(file))
    except FileNotFoundError:
        pass

# Load coordinates when the app starts
load_coordinates()

# Dropdown list
courses = list(coordinates.keys())  # Use the existing coordinates as the course names
selected_course = tk.StringVar()
selected_course.set(courses[0])  # Set the default value



def callback(url):  # for opening hyperlink
    webbrowser.open_new(url)


def loadingWindow():
    win = tk.Tk()
    win.title('Generating')
    message = "Generating yardage book..."
    tk.Label(win, text=message).pack(padx=20, pady=20)
    return win


def run_program():
    # get the coordinates, colors, and other options from the GUI
    try:
        lonmin = float(ent_minlon.get())
        latmax = float(ent_maxlat.get())
        latmin = float(ent_minlat.get())
        lonmax = float(ent_maxlon.get())
    except ValueError:
        tk.messagebox.showerror(title="Error", message="Please make sure all coordinates are formatted properly.")
        return False

    try:
        colors = {
            "fairways": hexToBGR(color_entries["Fairways"].get()),
            "tee boxes": hexToBGR(color_entries["Tee Boxes"].get()),
            "greens": hexToBGR(color_entries["Greens"].get()),
            "background": hexToBGR(color_entries["Background"].get()),
            "trees": hexToBGR(color_entries["Trees"].get()),
            "water": hexToBGR(color_entries["Water"].get()),
            "sand": hexToBGR(color_entries["Sand"].get()),
            "text": hexToBGR(color_entries["Text & Labels"].get()),
            "woods": hexToBGR(color_entries["Trees"].get())
        }
    except ValueError:
        tk.messagebox.showerror(title="Error", message="Please make sure all colors are properly entered in hex format.")
        return False

    replace_existing = overwrite_var.get()
    measure_in_meters = metersSetting.get()  # print in either meters or yards (default)

    small_width = ent_sm_scale.get()

    try:
        hole_width = int(ent_width.get())
    except ValueError:
        tk.messagebox.showerror(title="Error", message="Please enter a whole number as the hole width.")
        return False

    small_scale = small_width / 100

    med_scale = (small_scale + 1) / 2
    
    #get the selected tbox
    try:
        chosen_tbox = get_selected_tbox()
    except ValueError:
        tk.messagebox.showerror(title="Error", message="Please select a tbox.")
        return False

    # generate the yardage book
    try:
        loading = loadingWindow()
        time.sleep(1)
        book = generateYardageBook(latmin, lonmin, latmax, lonmax, replace_existing, colors, chosen_tbox,filter_width=hole_width,
                                   short_factor=small_scale, med_factor=med_scale)
        loading.destroy()
    except ValueError:
        mb.showerror(title="Error",
                     message="Error: unable to look up coordinates in OSM. Please make sure all coordinates are formatted properly.")
        return False

    # open the output folder for the user
    folderpath = os.path.dirname(os.path.abspath(__file__))
    webbrowser.open('file:///' + folderpath + "/output")

    return True

#read from yaml and update the UI
def update_fields(event):
    selected = course_dropdown.get()
    if selected in coordinates:
        values = coordinates[selected]
        ent_minlon.delete(0, tk.END)
        ent_minlon.insert(0, values["minlon"])
        ent_maxlat.delete(0, tk.END)
        ent_maxlat.insert(0, values["maxlat"])
        ent_minlat.delete(0, tk.END)
        ent_minlat.insert(0, values["minlat"])
        ent_maxlon.delete(0, tk.END)
        ent_maxlon.insert(0, values["maxlon"])


btn_confirm = tk.Button(text="Generate Yardages", master=frm_button, command=run_program, width=20)
btn_confirm.pack()

link1 = tk.Label(master=frm_button, text="Support Hacker Yardage by buying me a coffee", fg="blue")
link1.pack(pady=10)
link1.bind("<Button-1>", lambda e: callback("https://www.buymeacoffee.com/elementninety3"))

lbl_coords = tk.Label(master=frm_title,
                      text="Enter the coordinates from OSM for your course or select course from database:")
lbl_coords.grid(row=0, column=0, sticky="w")

frm_minlon = tk.Frame(master=frm_coords)
frm_minlon.grid(row=1, column=0, padx=5, pady=0)

frm_maxlat = tk.Frame(master=frm_coords)
frm_maxlat.grid(row=0, column=1, padx=5, pady=0)

frm_minlat = tk.Frame(master=frm_coords)
frm_minlat.grid(row=2, column=1, padx=5, pady=0)

frm_maxlon = tk.Frame(master=frm_coords)
frm_maxlon.grid(row=1, column=2, padx=5, pady=0)

frm_coords.columnconfigure(0, weight=1, minsize=150)
frm_coords.rowconfigure(0, weight=1, minsize=50)
frm_coords.columnconfigure(1, weight=1, minsize=150)
frm_coords.rowconfigure(1, weight=1, minsize=50)
frm_coords.columnconfigure(2, weight=1, minsize=150)
frm_coords.rowconfigure(2, weight=1, minsize=50)

lbl_minlon = tk.Label(master=frm_minlon, text="West")
ent_minlon = tk.Entry(master=frm_minlon)
ent_minlon.insert(0, "103.79119")
lbl_minlon.pack()
ent_minlon.pack()

lbl_maxlat = tk.Label(master=frm_maxlat, text="North")
ent_maxlat = tk.Entry(master=frm_maxlat)
ent_maxlat.insert(0, "1.34914")
lbl_maxlat.pack()
ent_maxlat.pack()

lbl_minlat = tk.Label(master=frm_minlat, text="South")
ent_minlat = tk.Entry(master=frm_minlat)
ent_minlat.insert(0, "1.34328")
lbl_minlat.pack()
ent_minlat.pack()

lbl_maxlon = tk.Label(master=frm_maxlon, text="East")
ent_maxlon = tk.Entry(master=frm_maxlon)
ent_maxlon.insert(0, "103.80034")
lbl_maxlon.pack()
ent_maxlon.pack()

frm_colors = tk.Frame(master=frm_options)
frm_colors.grid(row=0, column=0, padx=5, pady=5)

frm_colorslabel = tk.Frame(master=frm_colors)
frm_colorslabel.grid(row=0, column=0, columnspan=2, pady=5)

lbl_colorslabel = tk.Label(master=frm_colorslabel, text="Customize colors:")
lbl_colorslabel.pack()

default_colors = {"Fairways": "#7A7A7D", "Tee Boxes": "#85D87E", "Greens": "#4B8552", "Background": "#FFFFFF",
                  "Trees": "#99bfb0", "Water": "#BAFBEB", "Sand": "#EFE9BF", "Text & Labels": "#000000"}

color_entries = {}

for i, color in enumerate(default_colors):
    lbl_frame = tk.Frame(master=frm_colors)
    ent_frame = tk.Frame(master=frm_colors)
    label_text = color + ": "

    lbl_color = tk.Label(master=lbl_frame, text=label_text, anchor="e", width=10)
    ent_color = tk.Entry(master=ent_frame, width=15)
    ent_color.insert(0, default_colors[color])

    lbl_color.pack(side="right")
    ent_color.pack(side="left")

    lbl_frame.grid(row=(i + 1), column=0, pady=5)
    ent_frame.grid(row=(i + 1), column=1, pady=5)

    color_entries[color] = ent_color

frm_others = tk.Frame(master=frm_options)
frm_others.grid(row=0, column=1, padx=5, pady=5)

frm_otherslabel = tk.Frame(master=frm_others)
frm_otherslabel.grid(row=0, column=1, pady=5)

lbl_otherslabel = tk.Label(master=frm_otherslabel, text="Other options:")
lbl_otherslabel.pack()

frm_width = tk.Frame(master=frm_others)
lbl_frame = tk.Frame(master=frm_width)
ent_frame = tk.Frame(master=frm_width)
label_text = "Hole width (yards):"
lbl_width = tk.Label(master=lbl_frame, text=label_text, anchor="e", width=13)
ent_width = tk.Entry(master=ent_frame, width=5)
ent_width.insert(0, "50")
lbl_width.pack(side="right")
ent_width.pack(side="left")
lbl_frame.grid(row=0, column=0, pady=5)
ent_frame.grid(row=0, column=1, pady=5)
frm_width.grid(row=1, column=1, padx=5,)

frm_sm_scale = tk.Frame(master=frm_others)
lbl_sm_scale = tk.Label(master=frm_sm_scale, text="Hole width at tees (percent):")
lbl_sm_scale.grid(row=0, column=0)
ent_sm_scale = tk.Scale(master=frm_sm_scale, from_=20, to=100, orient="horizontal")
ent_sm_scale.set(100)
ent_sm_scale.grid(row=0, column=1)
frm_sm_scale.grid(row=2, column=1, padx=5, pady=5)

overwrite_var = tk.IntVar()
metersSetting = tk.IntVar()

ent_overwrite = tk.Checkbutton(master=frm_others, text="Overwrite existing files?", variable=overwrite_var)
ent_overwrite.grid(row=3, column=1, padx=5, pady=10)

ent_meters = tk.Checkbutton(master=frm_others, text="Meters?", variable=metersSetting)
ent_meters.grid(row=4, column=1, padx=5, pady=10)

# Dropdown list to select saved courses
course_dropdown = ttk.Combobox(master=frm_dropdown, values=courses, textvariable=selected_course)
course_dropdown.pack(side="left")  
course_dropdown.bind("<<ComboboxSelected>>", update_fields)  # Bind the update_fields function to the selection event

# Dropdown list to select T boxes
tboxes=["all","red", "white", "blue", "black"]
selected_tbox = tk.StringVar()
tbox_dropdown = ttk.Combobox(master=frm_dropdown, values=tboxes,textvariable=selected_tbox)
tbox_dropdown.current(0)
tbox_dropdown.pack(side="right")



def get_selected_tbox():
    chosen_tbox = selected_tbox.get()  # Retrieve the selected value
    print("Selected tbox:", chosen_tbox)  # You can use chosen_course as needed


def dropdown_callback(event):
    # Handle selection change
    selected = course_dropdown.get()
    # Do something with the selected course
    print("Selected course:", selected)

course_dropdown.bind("<<ComboboxSelected>>", update_fields)

#save coordinates to a file (yaml)
def save_coordinates():
    course_name = sd.askstring("Save Course", "Enter a name for the course:")
    if course_name:
        # Save the coordinates using the provided course name
        coordinates[course_name] = {
            "minlon": ent_minlon.get(),
            "maxlat": ent_maxlat.get(),
            "minlat": ent_minlat.get(),
            "maxlon": ent_maxlon.get()
        }
        with open("coordinates.yml", "w") as file:
            yaml.dump(coordinates, file)
        course_dropdown["values"] = list(coordinates.keys())
        course_dropdown.current(len(coordinates) - 1)  # Select the newly added course
        print("Coordinates saved for course:", course_name)
        mb.showinfo("Save Successful", "Coordinates saved successfully!")
    else:
        mb.showwarning("Save Failed", "Course name cannot be empty!")  


frm_buttons = tk.Frame(master=frm_title)
frm_buttons.grid(row=2, column=0, pady=10)

btn_save = tk.Button(master=frm_buttons, text="Save", command=save_coordinates)
btn_save.pack(side="left")


root.mainloop()
