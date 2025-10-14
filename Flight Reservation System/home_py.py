import tkinter as tk
from tkinter import ttk

class HomePage:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        
    def create_page(self):
        """Create the home page UI"""
        if self.frame:
            self.frame.destroy()
            
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            self.frame, 
            text="Flight Reservation System",
            font=('Arial', 24, 'bold')
        )
        title_label.pack(pady=(0, 30))
        
        # Subtitle
        subtitle_label = ttk.Label(
            self.frame,
            text="Welcome! Choose an option below:",
            font=('Arial', 12)
        )
        subtitle_label.pack(pady=(0, 40))
        
        # Button frame
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=20)
        
        # Book Flight Button
        book_btn = ttk.Button(
            button_frame,
            text="📝 Book Flight",
            command=self.app.show_booking_page,
            style='Large.TButton',
            width=25
        )
        book_btn.pack(pady=10)
        
        # View Reservations Button
        view_btn = ttk.Button(
            button_frame,
            text="📋 View Reservations",
            command=self.app.show_reservations_page,
            style='Large.TButton',
            width=25
        )
        view_btn.pack(pady=10)
        
        # Exit Button
        exit_btn = ttk.Button(
            button_frame,
            text="❌ Exit",
            command=self.app.quit_app,
            style='Exit.TButton',
            width=25
        )
        exit_btn.pack(pady=(20, 10))
        
        # Footer
        footer_label = ttk.Label(
            self.frame,
            text="Flight Reservation System v1.0",
            font=('Arial', 8),
            foreground='gray'
        )
        footer_label.pack(side='bottom', pady=(40, 0))
        
    def show(self):
        """Show the home page"""
        self.create_page()
        if self.frame:
            self.frame.tkraise()