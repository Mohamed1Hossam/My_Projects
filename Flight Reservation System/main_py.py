import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Import our modules
from database_py import Database
from home_py import HomePage
from booking_py import BookingPage
from reservations_py import ReservationsPage
from edit_reservation_py import EditReservationPage

class FlightReservationApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Flight Reservation System")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Initialize database
        self.db = Database()
        
        # Configure styles
        self.setup_styles()
        
        # Initialize pages
        self.home_page = HomePage(self.root, self)
        self.booking_page = BookingPage(self.root, self)
        self.reservations_page = ReservationsPage(self.root, self)
        self.edit_page = EditReservationPage(self.root, self)
        
        # Start with home page
        self.show_home_page()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
    
    def setup_styles(self):
        """Setup custom styles for the application"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure(
            'Large.TButton',
            font=('Arial', 12),
            padding=(10, 8)
        )
        
        style.configure(
            'Success.TButton',
            font=('Arial', 10, 'bold')
        )
        
        style.configure(
            'Warning.TButton',
            font=('Arial', 10)
        )
        
        style.configure(
            'Danger.TButton',
            font=('Arial', 10)
        )
        
        style.configure(
            'Exit.TButton',
            font=('Arial', 10)
        )
        
        # Try to set theme
        try:
            style.theme_use('clam')
        except:
            pass
    
    def show_home_page(self):
        """Show the home page"""
        self.home_page.show()
    
    def show_booking_page(self):
        """Show the booking page"""
        self.booking_page.show()
    
    def show_reservations_page(self):
        """Show the reservations page"""
        self.reservations_page.show()
    
    def show_edit_page(self, reservation_id):
        """Show the edit reservation page"""
        self.edit_page.show(reservation_id)
    
    def quit_app(self):
        """Handle application exit"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit_app()

def main():
    """Main function"""
    try:
        app = FlightReservationApp()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()