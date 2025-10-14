import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

class EditReservationPage:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        self.entries = {}
        self.reservation_id = None
        self.reservation_data = None
        
    def create_page(self, reservation_id):
        """Create the edit reservation page UI"""
        self.reservation_id = reservation_id
        
        if self.frame:
            self.frame.destroy()
            
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Load reservation data
        self.load_reservation_data()
        
        if not self.reservation_data:
            messagebox.showerror("Error", "Reservation not found!")
            self.app.show_reservations_page()
            return
        
        # Title
        title_label = ttk.Label(
            self.frame,
            text=f"Edit Reservation (ID: {self.reservation_id})",
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Main form frame
        form_frame = ttk.LabelFrame(self.frame, text="Update Flight Details", padding=20)
        form_frame.pack(fill='both', expand=True, pady=10)
        
        # Form fields
        fields = [
            ('Name', 'name', self.reservation_data[1]),
            ('Flight Number', 'flight_number', self.reservation_data[2]),
            ('Departure', 'departure', self.reservation_data[3]),
            ('Destination', 'destination', self.reservation_data[4]),
            ('Date (YYYY-MM-DD)', 'date', self.reservation_data[5]),
            ('Seat Number', 'seat_number', self.reservation_data[6])
        ]
        
        self.entries = {}
        
        for i, (label_text, field_name, current_value) in enumerate(fields):
            # Label
            label = ttk.Label(form_frame, text=label_text + ":")
            label.grid(row=i, column=0, sticky='w', pady=5, padx=(0, 10))
            
            # Entry
            entry = ttk.Entry(form_frame, width=30, font=('Arial', 10))
            entry.grid(row=i, column=1, sticky='ew', pady=5)
            entry.insert(0, current_value)
            self.entries[field_name] = entry
            
        # Configure column weights
        form_frame.columnconfigure(1, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=20)
        
        # Update button
        update_btn = ttk.Button(
            button_frame,
            text="💾 Update Reservation",
            command=self.update_reservation,
            style='Success.TButton',
            width=20
        )
        update_btn.pack(side='left', padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(
            button_frame,
            text="❌ Cancel",
            command=self.app.show_reservations_page,
            width=20
        )
        cancel_btn.pack(side='left', padx=5)
        
        # Delete button
        delete_btn = ttk.Button(
            button_frame,
            text="🗑️ Delete Reservation",
            command=self.delete_reservation,
            style='Danger.TButton',
            width=20
        )
        delete_btn.pack(side='left', padx=5)
        
        # Reset button
        reset_btn = ttk.Button(
            button_frame,
            text="🔄 Reset Form",
            command=self.reset_form,
            width=20
        )
        reset_btn.pack(side='left', padx=5)
        
    def load_reservation_data(self):
        """Load the reservation data from database"""
        try:
            self.reservation_data = self.app.db.get_reservation_by_id(self.reservation_id)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reservation: {str(e)}")
            self.reservation_data = None
    
    def validate_form(self):
        """Validate form inputs"""
        errors = []
        
        # Check if all fields are filled
        for field_name, entry in self.entries.items():
            if not entry.get().strip():
                field_label = field_name.replace('_', ' ').title()
                errors.append(f"{field_label} is required")
        
        # Validate date format
        date_text = self.entries['date'].get().strip()
        if date_text:
            try:
                datetime.strptime(date_text, '%Y-%m-%d')
            except ValueError:
                errors.append("Date must be in YYYY-MM-DD format")
        
        return errors
    
    def update_reservation(self):
        """Handle reservation update"""
        errors = self.validate_form()
        
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return
        
        try:
            # Get form data
            name = self.entries['name'].get().strip()
            flight_number = self.entries['flight_number'].get().strip()
            departure = self.entries['departure'].get().strip()
            destination = self.entries['destination'].get().strip()
            date = self.entries['date'].get().strip()
            seat_number = self.entries['seat_number'].get().strip()
            
            # Update in database
            success = self.app.db.update_reservation(
                self.reservation_id, name, flight_number, departure, destination, date, seat_number
            )
            
            if success:
                messagebox.showinfo("Success", "Reservation updated successfully!")
                self.app.show_reservations_page()
            else:
                messagebox.showerror("Error", "Failed to update reservation. Please try again.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def delete_reservation(self):
        """Handle reservation deletion"""
        if not self.reservation_data:
            return
            
        passenger_name = self.reservation_data[1]
        flight_number = self.reservation_data[2]
        
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this reservation?\n\n"
            f"Passenger: {passenger_name}\n"
            f"Flight: {flight_number}\n\n"
            f"This action cannot be undone."
        )
        
        if confirm:
            try:
                success = self.app.db.delete_reservation(self.reservation_id)
                if success:
                    messagebox.showinfo("Success", "Reservation deleted successfully!")
                    self.app.show_reservations_page()
                else:
                    messagebox.showerror("Error", "Failed to delete reservation.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def reset_form(self):
        """Reset form to original values"""
        if not self.reservation_data:
            return
            
        values = [
            self.reservation_data[1],  # name
            self.reservation_data[2],  # flight_number
            self.reservation_data[3],  # departure
            self.reservation_data[4],  # destination
            self.reservation_data[5],  # date
            self.reservation_data[6]   # seat_number
        ]
        
        for i, (field_name, entry) in enumerate(self.entries.items()):
            entry.delete(0, tk.END)
            entry.insert(0, values[i])
    
    def show(self, reservation_id):
        """Show the edit reservation page"""
        self.create_page(reservation_id)
        if self.frame:
            self.frame.tkraise()