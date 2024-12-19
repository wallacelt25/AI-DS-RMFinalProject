import tkinter as tk
from tkinter import ttk
from WarehouseAdmin import WarehouseAdmin

class WarehouseGUI:
    def __init__(self, warehouse_admin):
        self.warehouse_admin = warehouse_admin
        self.root = tk.Tk()
        self.root.title("Advanced Warehouse Management System")
        self.create_widgets()

    def create_widgets(self):
        # Build GUI components here: buttons, labels, entry fields, etc.
        # For example, a listbox to display items and buttons to process orders

     def run(self):
        self.root.mainloop()

    def update_inventory_display(self):
        """
        Updates the display of inventory items.
        """
        self.item_listbox.delete(0, 'end')
        for item_id, item in self.warehouse_admin.layout.items.items():
            self.item_listbox.insert('end', f"{item_id} - {item.location} - {item.weight}kg")

    def visualize_order_processing(self, order):
        """
        Visualizes the processing of an order.
        Placeholder logic: simply list the items in the order.
        """
        print("Processing Order:")
        for item in order.items:
            print(f"Item ID: {item.product_id}, Location: {item.location}")
