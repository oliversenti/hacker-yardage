import tkinter as tk
from tkinter import messagebox

class Coordinate:
    def __init__(self, name, north, east, south, west):
        self.name = name
        self.north = north
        self.east = east
        self.south = south
        self.west = west

class CoordinateApp:
    def __init__(self, root):
        self.root = root
        self.coordinates = []
        self.load_coordinates()

        self.name_label = tk.Label(root, text="Name:")
        self.name_label.pack()
        self.name_entry = tk.Entry(root)
        self.name_entry.pack()

        self.north_label = tk.Label(root, text="North:")
        self.north_label.pack()
        self.north_entry = tk.Entry(root)
        self.north_entry.pack()

        self.east_label = tk.Label(root, text="East:")
        self.east_label.pack()
        self.east_entry = tk.Entry(root)
        self.east_entry.pack()

        self.south_label = tk.Label(root, text="South:")
        self.south_label.pack()
        self.south_entry = tk.Entry(root)
        self.south_entry.pack()

        self.west_label = tk.Label(root, text="West:")
        self.west_label.pack()
        self.west_entry = tk.Entry(root)
        self.west_entry.pack()

        self.save_button = tk.Button(root, text="Save", command=self.save_coordinates)
        self.save_button.pack()

        self.coordinates_dropdown = tk.StringVar(root)
        self.coordinates_dropdown.trace("w", self.on_dropdown_select)  # Bind callback function to dropdown selection
        self.coordinates_menu = tk.OptionMenu(root, self.coordinates_dropdown, ())
        self.coordinates_menu.pack()

        self.update_dropdown_options()

    def save_coordinates(self):
        name = self.name_entry.get()
        north = self.north_entry.get()
        east = self.east_entry.get()
        south = self.south_entry.get()
        west = self.west_entry.get()

        if not name or not north or not east or not south or not west:
            messagebox.showwarning("Error", "Please fill in all the fields")
            return

        coordinate = Coordinate(name, north, east, south, west)
        self.coordinates.append(coordinate)
        self.update_dropdown_options()

        self.name_entry.delete(0, tk.END)
        self.north_entry.delete(0, tk.END)
        self.east_entry.delete(0, tk.END)
        self.south_entry.delete(0, tk.END)
        self.west_entry.delete(0, tk.END)

        self.save_coordinates_to_file()

    def update_dropdown_options(self, *args):
        menu = self.coordinates_menu["menu"]
        menu.delete(0, "end")

        for coordinate in self.coordinates:
            menu.add_command(label=coordinate.name, command=lambda name=coordinate.name: self.coordinates_dropdown.set(name))

    def on_dropdown_select(self, *args):
        selected_name = self.coordinates_dropdown.get()
        if selected_name:
            for coordinate in self.coordinates:
                if coordinate.name == selected_name:
                    self.name_entry.delete(0, tk.END)
                    self.name_entry.insert(0, coordinate.name)
                    self.north_entry.delete(0, tk.END)
                    self.north_entry.insert(0, coordinate.north)
                    self.east_entry.delete(0, tk.END)
                    self.east_entry.insert(0, coordinate.east)
                    self.south_entry.delete(0, tk.END)
                    self.south_entry.insert(0, coordinate.south)
                    self.west_entry.delete(0, tk.END)
                    self.west_entry.insert(0, coordinate.west)
                    break

    def save_coordinates_to_file(self):
        with open("coordinates.txt", "w") as file:
            for coordinate in self.coordinates:
                file.write(f"{coordinate.name},{coordinate.north},{coordinate.east},{coordinate.south},{coordinate.west}\n")

    def load_coordinates(self):
        try:
            with open("coordinates.txt", "r") as file:
                for line in file:
                    values = line.strip().split(",")
                    if len(values) == 5:
                        name, north, east, south, west = values
                        coordinate = Coordinate(name, north, east, south, west)
                        self.coordinates.append(coordinate)
        except FileNotFoundError:
            pass

root = tk.Tk()
app = CoordinateApp(root)
root.mainloop()
