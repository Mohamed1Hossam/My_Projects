import sqlite3
import os

class Database:
    def __init__(self, db_name="flights.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create tables if they don't exist"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create reservations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                flight_number TEXT NOT NULL,
                departure TEXT NOT NULL,
                destination TEXT NOT NULL,
                date TEXT NOT NULL,
                seat_number TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_name)
    
    def create_reservation(self, name, flight_number, departure, destination, date, seat_number):
        """Create a new reservation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reservations (name, flight_number, departure, destination, date, seat_number)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, flight_number, departure, destination, date, seat_number))
        
        conn.commit()
        reservation_id = cursor.lastrowid
        conn.close()
        return reservation_id
    
    def get_all_reservations(self):
        """Get all reservations"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM reservations ORDER BY id DESC')
        reservations = cursor.fetchall()
        
        conn.close()
        return reservations
    
    def get_reservation_by_id(self, reservation_id):
        """Get a specific reservation by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM reservations WHERE id = ?', (reservation_id,))
        reservation = cursor.fetchone()
        
        conn.close()
        return reservation
    
    def update_reservation(self, reservation_id, name, flight_number, departure, destination, date, seat_number):
        """Update an existing reservation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE reservations 
            SET name = ?, flight_number = ?, departure = ?, destination = ?, date = ?, seat_number = ?
            WHERE id = ?
        ''', (name, flight_number, departure, destination, date, seat_number, reservation_id))
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        return rows_affected > 0
    
    def delete_reservation(self, reservation_id):
        """Delete a reservation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM reservations WHERE id = ?', (reservation_id,))
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        return rows_affected > 0