import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

class BookingPage:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        self.entries = {}
        
    def create_page(self):
        """Create the booking page UI"""
        if self.frame:
            self.frame.destroy()
            
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            self.frame,
            text="Book New Flight",
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Main form frame
        form_frame = ttk.LabelFrame(self.frame, text="Flight Details", padding=20)
        form_frame.pack(fill='both', expand=True, pady=10)
        
        # Form fields
        fields = [
            ('Name', 'name'),
            ('Flight Number', 'flight_number'),
            ('Departure', 'departure'),
            ('Destination', 'destination'),
            ('Date (YYYY-MM-DD)', 'date'),
            ('Seat Number', 'seat_number')
        ]
        
        for i, (label_text, field_name) in enumerate(fields):
            # Label
            label = ttk.Label(form_frame, text=label_text + ":")
            label.grid(row=i, column=0, sticky='w', pady=5, padx=(0, 10))
            
            # Entry
            entry = ttk.Entry(form_frame, width=30, font=('Arial', 10))
            entry.grid(row=i, column=1, sticky='ew', pady=5)
            self.entries[field_name] = entry
            
        # Configure column weights
        form_frame.columnconfigure(1, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=20)
        
        # Submit button
        submit_btn = ttk.Button(
            button_frame,
            text="✈️ Book Flight",
            command=self.book_flight,
            style='Success.TButton',
            width=20
        )
        submit_btn.pack(side='left', padx=5)
        
        # Back button
        back_btn = ttk.Button(
            button_frame,
            text="🔙 Back to Home",
            command=self.app.show_home_page,
            width=20
        )
        back_btn.pack(side='left', padx=5)
        
        # Clear button
        clear_btn = ttk.Button(
            button_frame,
            text="🗑️ Clear Form",
            command=self.clear_form,
            width=20
        )
        clear_btn.pack(side='left', padx=5)
        
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
    
    def book_flight(self):
        """Handle flight booking"""
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
            
            # Save to database
            reservation_id = self.app.db.create_reservation(
                name, flight_number, departure, destination, date, seat_number
            )
            
            if reservation_id:
                messagebox.showinfo(
                    "Success", 
                    f"Flight booked successfully!\nReservation ID: {reservation_id}"
                )
                self.clear_form()
                self.app.show_reservations_page()
            else:
                messagebox.showerror("Error", "Failed to book flight. Please try again.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def clear_form(self):
        """Clear all form fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
    
    def show(self):
        """Show the booking page"""
        self.create_page()
        if self.frame:
            self.frame.tkraise()