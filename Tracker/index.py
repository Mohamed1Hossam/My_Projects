import tkinter as tk
from tkinter import messagebox
import requests
import folium
import webbrowser


def get_phone_location():
    phone_number = entry.get()
    if not phone_number:
        messagebox.showerror("Error", "Please enter a phone number.")
        return

    API_KEY = "your_numverify_api_key"  # Replace with your API key
    url = f"http://apilayer.net/api/validate?access_key={API_KEY}&number={phone_number}&country_code=&format=1"

    try:
        response = requests.get(url)
        data = response.json()

        if not data.get("valid", False):
            messagebox.showerror("Error", "Invalid phone number.")
            return

        country = data.get("country_name", "Unknown")
        location = data.get("location", "Unknown")
        carrier = data.get("carrier", "Unknown")
        lat, lon = 0, 0  # Approximate coordinates can be fetched via another API if needed

        # Display results in the GUI
        result_label.config(text=f"Country: {country}\nLocation: {location}\nCarrier: {carrier}")

        # Create a map and open it in a browser
        map = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker([lat, lon], popup=f"{location}, {country}").add_to(map)
        map.save("location_map.html")
        webbrowser.open("location_map.html")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data: {e}")


# Create GUI window
root = tk.Tk()
root.title("Phone Number Locator")
root.geometry("400x300")

tk.Label(root, text="Enter Phone Number:").pack(pady=5)
entry = tk.Entry(root, width=30)
entry.pack(pady=5)

tk.Button(root, text="Get Location", command=get_phone_location).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
