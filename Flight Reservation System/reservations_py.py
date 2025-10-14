import tkinter as tk
from tkinter import ttk, messagebox

class ReservationsPage:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        self.tree = None
        
    def create_page(self):
        """Create the reservations page UI"""
        if self.frame:
            self.frame.destroy()
            
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            self.frame,
            text="Flight Reservations",
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Button frame (top)
        top_button_frame = ttk.Frame(self.frame)
        top_button_frame.pack(pady=(0, 10))
        
        # Refresh button
        refresh_btn = ttk.Button(
            top_button_frame,
            text="🔄 Refresh",
            command=self.load_reservations,
            width=15
        )
        refresh_btn.pack(side='left', padx=5)
        
        # Back button
        back_btn = ttk.Button(
            top_button_frame,
            text="🔙 Back to Home",
            command=self.app.show_home_page,
            width=15
        )
        back_btn.pack(side='left', padx=5)
        
        # Table frame
        table_frame = ttk.LabelFrame(self.frame, text="Reservations List", padding=10)
        table_frame.pack(fill='both', expand=True, pady=10)
        
        # Create Treeview
        columns = ('ID', 'Name', 'Flight Number', 'Departure', 'Destination', 'Date', 'Seat')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        self.tree.heading('ID', text='ID')
        self.tree.heading('Name', text='Passenger Name')
        self.tree.heading('Flight Number', text='Flight Number')
        self.tree.heading('Departure', text='Departure')
        self.tree.heading('Destination', text='Destination')
        self.tree.heading('Date', text='Date')
        self.tree.heading('Seat', text='Seat')
        
        # Configure column widths
        self.tree.column('ID', width=50, anchor='center')
        self.tree.column('Name', width=150)
        self.tree.column('Flight Number', width=100, anchor='center')
        self.tree.column('Departure', width=120)
        self.tree.column('Destination', width=120)
        self.tree.column('Date', width=100, anchor='center')
        self.tree.column('Seat', width=80, anchor='center')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Action buttons frame
        action_frame = ttk.Frame(self.frame)
        action_frame.pack(pady=10)
        
        # Edit button
        edit_btn = ttk.Button(
            action_frame,
            text="✏️ Edit Selected",
            command=self.edit_selected,
            style='Warning.TButton',
            width=20
        )
        edit_btn.pack(side='left', padx=5)
        
        # Delete button
        delete_btn = ttk.Button(
            action_frame,
            text="🗑️ Delete Selected",
            command=self.delete_selected,
            style='Danger.TButton',
            width=20
        )
        delete_btn.pack(side='left', padx=5)
        
        # Info label
        self.info_label = ttk.Label(
            self.frame,
            text="Select a reservation to edit or delete",
            font=('Arial', 10),
            foreground='gray'
        )
        self.info_label.pack(pady=10)
        
        # Bind double-click to edit
        self.tree.bind('<Double-1>', lambda e: self.edit_selected())
        
    def load_reservations(self):
        """Load and display all reservations"""
        if not self.tree:
            return
            
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            reservations = self.app.db.get_all_reservations()
            
            if not reservations:
                self.info_label.config(text="No reservations found. Book a flight to get started!")
                return
            
            # Insert reservations into tree
            for reservation in reservations:
                self.tree.insert('', 'end', values=reservation)
            
            self.info_label.config(
                text=f"Total reservations: {len(reservations)} | Select a reservation to edit or delete"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reservations: {str(e)}")
            self.info_label.config(text="Error loading reservations")
    
    def get_selected_reservation_id(self):
        """Get the ID of the selected reservation"""
        selection = self.tree.selection()
        if not selection:
            return None
            
        item = self.tree.item(selection[0])
        return item['values'][0] if item['values'] else None
    
    def edit_selected(self):
        """Edit the selected reservation"""
        reservation_id = self.get_selected_reservation_id()
        if not reservation_id:
            messagebox.showwarning("No Selection", "Please select a reservation to edit.")
            return
        
        self.app.show_edit_page(reservation_id)
    
    def delete_selected(self):
        """Delete the selected reservation"""
        reservation_id = self.get_selected_reservation_id()
        if not reservation_id:
            messagebox.showwarning("No Selection", "Please select a reservation to delete.")
            return
        
        # Confirm deletion
        selection = self.tree.selection()
        item = self.tree.item(selection[0])
        passenger_name = item['values'][1]
        flight_number = item['values'][2]
        
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this reservation?\n\n"
            f"Passenger: {passenger_name}\n"
            f"Flight: {flight_number}\n\n"
            f"This action cannot be undone."
        )
        
        if confirm:
            try:
                success = self.app.db.delete_reservation(reservation_id)
                if success:
                    messagebox.showinfo("Success", "Reservation deleted successfully!")
                    self.load_reservations()
                else:
                    messagebox.showerror("Error", "Failed to delete reservation.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def show(self):
        """Show the reservations page"""
        self.create_page()
        self.load_reservations()
        if self.frame:
            self.frame.tkraise()