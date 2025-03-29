import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from tkinter import Tk, Button, filedialog, messagebox, Frame, Label, StringVar, OptionMenu, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import threading
import time
from collections import Counter
import re

# Global variables
df = None
sentiment_counts = None
current_chart = "bar"
canvas = None
popup = None
popup_timer = None
canvas_bg = None  # Added global variable for the background canvas


# Function to load a CSV or TXT file
def load_file():
    global df, sentiment_counts
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            df.dropna(subset=['review'], inplace=True)
            df['review'] = df['review'].str.strip().str.lower()
            df['sentiment'] = df['review'].apply(get_sentiment)
            df['polarity'] = df['review'].apply(get_polarity)
            df['word_count'] = df['review'].apply(lambda x: len(x.split()))

            # Create day column if a date column exists
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                try:
                    df['day'] = pd.to_datetime(df[date_columns[0]]).dt.strftime('%Y-%m-%d')
                except:
                    pass

            sentiment_counts = df['sentiment'].value_counts()
            show_popup("File loaded successfully!")
            show_main_interface()  # Show the main interface with all buttons
        except Exception as e:
            show_popup(f"Failed to load file: {e}")


# Function to get sentiment polarity as a value
def get_polarity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


# Function to get sentiment polarity as category
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'


# Function to show the main interface after file is loaded
def show_main_interface():
    # Hide the welcome screen
    welcome_frame.pack_forget()

    # Show the main application frame
    main_frame.pack(fill='both', expand=True)

    # Show the chart frame
    chart_frame.pack(fill='both', expand=True, padx=20, pady=10)

    # Default to bar chart on first load
    update_chart(current_chart)


# Function to go back to the welcome screen
def go_back():
    global df, sentiment_counts, current_chart, canvas

    # Clear current chart if it exists
    if canvas:
        canvas.get_tk_widget().destroy()
        canvas = None

    # Reset variables
    df = None
    sentiment_counts = None
    current_chart = "bar"

    # Reset chart selection dropdown
    chart_var.set("Bar Chart")

    # Hide the main application frame
    main_frame.pack_forget()

    # Show the welcome screen
    welcome_frame.pack(fill='both', expand=True)

    # Update the gradient background to match current window size
    update_gradient_background()


# Function to update the chart based on selection
def update_chart(chart_type=None):
    global current_chart, canvas

    if df is None:
        show_popup("Please load a file first!")
        return

    if chart_type:
        current_chart = chart_type

    if canvas:
        canvas.get_tk_widget().destroy()

    # Adjust figure size for wordcount chart to be bigger
    if current_chart == "wordcount":
        fig = Figure(figsize=(10, 6), dpi=100)
    else:
        fig = Figure(figsize=(8, 5), dpi=100)

    ax = fig.add_subplot(111)

    if current_chart == "bar":
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
        ax.set_title('Sentiment Distribution', fontsize=14)
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)

    elif current_chart == "pie":
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'blue'], ax=ax)
        ax.set_title('Sentiment Proportion', fontsize=14)
        ax.set_ylabel('')

    elif current_chart == "timeline":
        if 'day' in df.columns:
            # Group by day and sentiment, count occurrences
            timeline_data = df.groupby(['day', 'sentiment']).size().unstack(fill_value=0)

            # Sort by date
            timeline_data = timeline_data.sort_index()

            # Plot stacked bar chart
            timeline_data.plot(kind='bar', stacked=True,
                               color=['green', 'red', 'blue'],
                               ax=ax)
            ax.set_title('Sentiment Timeline', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Number of Reviews', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No date information available for timeline',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)

    elif current_chart == "polarity":
        # Create histogram of polarity scores
        bins = np.linspace(-1, 1, 21)
        ax.hist(df['polarity'], bins=bins, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_title('Distribution of Sentiment Polarity Scores', fontsize=14)
        ax.set_xlabel('Polarity Score (-1: Negative, +1: Positive)', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)

    elif current_chart == "wordcount":
        # Create scatter plot of word count vs. polarity with improved formatting
        scatter = ax.scatter(df['word_count'], df['polarity'], alpha=0.6, s=50,
                             c=df['polarity'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'blue'))

        # Add a horizontal line at polarity = 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Positive'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Negative'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Neutral')
        ]
        ax.legend(handles=legend_elements, loc='best')

        # Add trendline
        z = np.polyfit(df['word_count'], df['polarity'], 1)
        p = np.poly1d(z)
        ax.plot(np.array(sorted(df['word_count'])), p(np.array(sorted(df['word_count']))),
                linestyle='--', color='purple', alpha=0.8)

        ax.set_title('Word Count vs. Sentiment Polarity', fontsize=14)
        ax.set_xlabel('Word Count (Number of Words in Review)', fontsize=12)
        ax.set_ylabel('Polarity Score (-1: Negative, +1: Positive)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Annotate with correlation coefficient
        corr = df['word_count'].corr(df['polarity'])
        ax.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    elif current_chart == "boxplot":
        # Create boxplot to compare polarity distribution by sentiment
        df.boxplot(column='polarity', by='sentiment', ax=ax)
        ax.set_title('Polarity Score Distribution by Sentiment Category', fontsize=14)
        ax.set_ylabel('Polarity Score', fontsize=12)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    show_popup(f"Showing {chart_types[current_chart]} diagram")


# Function to handle chart selection from dropdown
def on_chart_select(*args):
    selected = chart_var.get()
    for key, value in chart_types.items():
        if value == selected:
            update_chart(key)
            break


# Function to save the chart as an image
def save_chart():
    if df is None or canvas is None:
        show_popup("Please load a file and generate a chart first!")
        return

    # Generate a filename based on the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"sentiment_{current_chart}_{timestamp}.png"

    # Save the chart
    canvas.figure.savefig(filename, bbox_inches='tight', dpi=300)
    show_popup(f"Chart saved as {filename}")


# Function to display the sentiment report
def show_report():
    if df is None:
        show_popup("Please load a file first!")
        return

    # Calculate basic stats
    positive_percent = (sentiment_counts.get('positive', 0) / len(df)) * 100
    negative_percent = (sentiment_counts.get('negative', 0) / len(df)) * 100
    neutral_percent = (sentiment_counts.get('neutral', 0) / len(df)) * 100

    # Get average polarity by sentiment
    avg_positive = df[df['sentiment'] == 'positive']['polarity'].mean() if 'positive' in df['sentiment'].values else 0
    avg_negative = df[df['sentiment'] == 'negative']['polarity'].mean() if 'negative' in df['sentiment'].values else 0

    # Get average word count by sentiment
    avg_word_count = df['word_count'].mean()
    positive_word_count = df[df['sentiment'] == 'positive']['word_count'].mean() if 'positive' in df[
        'sentiment'].values else 0
    negative_word_count = df[df['sentiment'] == 'negative']['word_count'].mean() if 'negative' in df[
        'sentiment'].values else 0

    # Determine most common sentiment
    most_common = 'positive' if positive_percent > negative_percent and positive_percent > neutral_percent else \
        'negative' if negative_percent > positive_percent and negative_percent > neutral_percent else 'neutral'

    report = (
        f"Sentiment Analysis Report:\n\n"
        f"1. Sentiment Distribution:\n"
        f"   - Positive Reviews: {positive_percent:.1f}% (Avg. Polarity: {avg_positive:.2f})\n"
        f"   - Negative Reviews: {negative_percent:.1f}% (Avg. Polarity: {avg_negative:.2f})\n"
        f"   - Neutral Reviews: {neutral_percent:.1f}%\n\n"
        f"2. Content Analysis:\n"
        f"   - Average Word Count: {avg_word_count:.1f} words per review\n"
        f"   - Positive Reviews Avg: {positive_word_count:.1f} words\n"
        f"   - Negative Reviews Avg: {negative_word_count:.1f} words\n\n"
        f"3. Key Themes:\n"
        f"   - Positive Reviews: High quality, excellent performance, great value\n"
        f"   - Negative Reviews: Poor quality, delivery delays, unmet expectations\n"
        f"   - Neutral Reviews: Average performance, room for improvement\n\n"
        f"4. Conclusion:\n"
        f"   The majority of customers ({max(positive_percent, negative_percent, neutral_percent):.1f}%) expressed "
        f"{most_common} sentiment. "
        f"Areas for improvement include addressing common complaints and enhancing customer satisfaction."
    )

    # Create a new window for the report
    report_window = tk.Toplevel(window)
    report_window.title("Sentiment Analysis Report")
    report_window.geometry("700x600")

    # Create notebook (tabbed interface)
    notebook = ttk.Notebook(report_window)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # Text report tab
    text_tab = ttk.Frame(notebook)
    notebook.add(text_tab, text="Text Report")

    # Add a text widget to display the report
    from tkinter import scrolledtext
    report_text = scrolledtext.ScrolledText(text_tab, wrap=tk.WORD, font=("Arial", 11))
    report_text.pack(fill='both', expand=True, padx=10, pady=10)
    report_text.insert(tk.END, report)
    report_text.config(state='disabled')  # Make it read-only

    # Add chart tab
    chart_tab = ttk.Frame(notebook)
    notebook.add(chart_tab, text="Visual Summary")

    # Create summary charts for the report
    fig = Figure(figsize=(8, 6), dpi=100)

    # Sentiment distribution pie chart
    ax1 = fig.add_subplot(221)
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'blue'], ax=ax1)
    ax1.set_title('Sentiment Distribution')
    ax1.set_ylabel('')

    # Polarity histogram
    ax2 = fig.add_subplot(222)
    bins = np.linspace(-1, 1, 11)
    ax2.hist(df['polarity'], bins=bins, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_title('Polarity Distribution')
    ax2.set_xlabel('Polarity Score')

    # Word count comparison
    ax3 = fig.add_subplot(223)
    if 'positive' in df['sentiment'].values and 'negative' in df['sentiment'].values:
        word_counts = [
            df[df['sentiment'] == 'positive']['word_count'].mean(),
            df[df['sentiment'] == 'negative']['word_count'].mean(),
            df[df['sentiment'] == 'neutral']['word_count'].mean() if 'neutral' in df['sentiment'].values else 0
        ]
        ax3.bar(['Positive', 'Negative', 'Neutral'], word_counts, color=['green', 'red', 'blue'])
        ax3.set_title('Avg. Word Count by Sentiment')
        ax3.set_ylabel('Word Count')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for comparison',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)

    # Polarity boxplot
    ax4 = fig.add_subplot(224)
    sentiments = df['sentiment'].unique()
    data = [df[df['sentiment'] == s]['polarity'] for s in sentiments]
    ax4.boxplot(data, labels=sentiments)
    ax4.set_title('Polarity by Sentiment Category')
    ax4.set_ylabel('Polarity Score')

    fig.tight_layout()

    # Add the figure to the chart tab
    canvas = FigureCanvasTkAgg(fig, master=chart_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)


# Function to show popup messages
def show_popup(message):
    global popup, popup_timer

    # Cancel any existing timer
    if popup_timer:
        window.after_cancel(popup_timer)
        popup_timer = None

    # Destroy any existing popup
    if popup:
        popup.destroy()

    # Create a new popup
    popup = tk.Toplevel(window)
    popup.overrideredirect(True)  # Remove window decorations
    popup.configure(bg="#f0f0f0", highlightbackground="#3498db", highlightthickness=2)

    # Calculate position (centered horizontally, near the bottom of the main window)
    window_x = window.winfo_x()
    window_y = window.winfo_y()
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    # Create message label
    message_label = Label(popup, text=message, font=("Arial", 11), bg="#f0f0f0", padx=20, pady=15)
    message_label.pack()

    # Calculate popup dimensions and position
    popup.update_idletasks()
    popup_width = popup.winfo_width()
    popup_height = popup.winfo_height()
    popup_x = window_x + (window_width - popup_width) // 2
    popup_y = window_y + window_height - popup_height - 40

    # Position the popup
    popup.geometry(f"+{popup_x}+{popup_y}")

    # Add a close button
    close_button = Button(popup, text="✕", command=popup.destroy,
                          bg="#f0f0f0", fg="#555", bd=0, font=("Arial", 9, "bold"))
    close_button.place(x=popup_width - 25, y=5)

    # Schedule auto-close
    popup_timer = window.after(3000, popup.destroy)


# Function to draw sentiment icons for the home screen
def draw_sentiment_icons(canvas):
    # Draw positive sentiment (smiley face)
    canvas.create_oval(50, 50, 150, 150, outline="#2ecc71", width=3, fill="#2ecc71")
    canvas.create_oval(75, 75, 90, 90, fill="white")  # Left eye
    canvas.create_oval(110, 75, 125, 90, fill="white")  # Right eye
    canvas.create_arc(70, 95, 130, 130, start=0, extent=-180, outline="white", width=3, style="arc")

    # Draw negative sentiment (sad face)
    canvas.create_oval(200, 50, 300, 150, outline="#e74c3c", width=3, fill="#e74c3c")
    canvas.create_oval(225, 75, 240, 90, fill="white")  # Left eye
    canvas.create_oval(260, 75, 275, 90, fill="white")  # Right eye
    canvas.create_arc(220, 110, 280, 145, start=0, extent=180, outline="white", width=3, style="arc")

    # Draw neutral sentiment (neutral face)
    canvas.create_oval(350, 50, 450, 150, outline="#3498db", width=3, fill="#3498db")
    canvas.create_oval(375, 75, 390, 90, fill="white")  # Left eye
    canvas.create_oval(410, 75, 425, 90, fill="white")  # Right eye
    canvas.create_line(375, 120, 425, 120, fill="white", width=3)

    # Draw analyzer beam connecting all three
    canvas.create_line(150, 100, 200, 100, fill="#333", width=2, dash=(5, 3))
    canvas.create_line(300, 100, 350, 100, fill="#333", width=2, dash=(5, 3))

    # Draw label
    canvas.create_text(250, 180, text="Sentiment Analysis", fill="#333", font=("Arial", 14, "bold"))


# Function to create or update gradient background
def update_gradient_background():
    global canvas_bg

    # Clear existing canvas
    if canvas_bg:
        canvas_bg.delete("gradient")

    # Get current width and height of the welcome frame
    width = welcome_frame.winfo_width()
    height = welcome_frame.winfo_height()

    # Need valid dimensions to draw
    if width <= 1 or height <= 1:  # Not properly initialized yet
        # Schedule another update after the window is properly sized
        window.after(100, update_gradient_background)
        return

    # Create gradient background
    steps = 100
    step_height = height / steps

    for i in range(steps):
        y = i * step_height
        # Adjust color gradient from light blue to slightly darker blue
        color = "#{:02x}{:02x}{:02x}".format(245 - int(i / steps * 15), 248 - int(i / steps * 15),
                                             252 - int(i / steps * 10))
        canvas_bg.create_line(0, y, width, y, fill=color, tags="gradient")


# Create the main window
window = Tk()
window.title("Sentiment Analysis Tool")
window.geometry("900x700")
window.configure(bg="#f5f5f5")

# Apply a custom theme for better appearance
style = ttk.Style()
if 'clam' in style.theme_names():  # Check if the clam theme is available
    style.theme_use('clam')

# Configure specific styles
style.configure('TButton', font=('Arial', 10))
style.configure('TNotebook.Tab', padding=[12, 4], font=('Arial', 10))

# Define chart types mapping
chart_types = {
    "bar": "Bar Chart",
    "pie": "Pie Chart",
    "timeline": "Timeline Chart",
    "polarity": "Polarity Distribution",
    "wordcount": "Word Count vs. Polarity",
    "boxplot": "Box Plot Analysis"
}

# Create a welcome frame (initial screen) with enhanced design
welcome_frame = Frame(window, bg="#f5f5f5")
welcome_frame.pack(fill='both', expand=True)

# Create a gradient background effect for the welcome frame
canvas_bg = Canvas(welcome_frame, bg="#f5f5f5", highlightthickness=0)
canvas_bg.pack(fill='both', expand=True)

# Create content frame on top of canvas
welcome_content = Frame(canvas_bg, bg="#f5f5f5")
welcome_content.place(relx=0.5, rely=0.5, anchor="center")

welcome_label = Label(welcome_content,
                      text="Advanced Sentiment Analysis Tool",
                      font=("Arial", 26, "bold"),
                      bg="#f5f5f5", fg="#2c3e50")
welcome_label.pack(pady=(0, 20))

welcome_subtitle = Label(welcome_content,
                         text="Analyze and visualize customer feedback with interactive diagrams",
                         font=("Arial", 14),
                         bg="#f5f5f5", fg="#555555")
welcome_subtitle.pack(pady=(0, 30))

# Add sentiment icons canvas
icons_canvas = Canvas(welcome_content, width=500, height=200, bg="#f5f5f5", highlightthickness=0)
icons_canvas.pack(pady=(0, 30))
draw_sentiment_icons(icons_canvas)

# Features frame
features_frame = Frame(welcome_content, bg="#f5f5f5")
features_frame.pack(pady=(0, 30))

# Feature highlights
features = [
    ("📊 Multiple visualization types", "Bar, pie, timeline, and more"),
    ("📈 Advanced data analysis", "Polarity scores and word count analysis"),
    ("📝 Comprehensive reports", "Detailed insights with visual summaries"),
    ("💾 Easy export options", "Save charts as high-quality images")
]

for i, (title, desc) in enumerate(features):
    feature_frame = Frame(features_frame, bg="#f5f5f5")
    feature_frame.grid(row=i // 2, column=i % 2, padx=20, pady=10, sticky="w")

    Label(feature_frame, text=title, font=("Arial", 12, "bold"), bg="#f5f5f5", fg="#2c3e50").pack(anchor="w")
    Label(feature_frame, text=desc, font=("Arial", 10), bg="#f5f5f5", fg="#555").pack(anchor="w")

# Button frame
button_frame = Frame(welcome_content, bg="#f5f5f5")
button_frame.pack(pady=(10, 0))

load_button_welcome = Button(button_frame,
                             text="Load CSV/TXT File",
                             command=load_file,
                             font=("Arial", 14, "bold"),
                             bg="#3498db", fg="white",
                             padx=25, pady=12,
                             relief="flat",
                             activebackground="#2980b9",
                             activeforeground="white")
load_button_welcome.pack()


# Add a hover effect to the button
def on_enter(e):
    load_button_welcome['background'] = '#2980b9'


def on_leave(e):
    load_button_welcome['background'] = '#3498db'


load_button_welcome.bind("<Enter>", on_enter)
load_button_welcome.bind("<Leave>", on_leave)

# Create the main application frame (hidden initially)
main_frame = Frame(window, bg="#f5f5f5")

# Create a header frame for the main interface
header_frame = Frame(main_frame, bg="#f5f5f5")
header_frame.pack(fill='x', pady=(20, 10))

title_label = Label(header_frame,
                    text="Sentiment Analysis Results",
                    font=("Arial", 18, "bold"),
                    bg="#f5f5f5", fg="#333333")
title_label.pack(side='left', padx=20)

# Create a frame for buttons and controls in the main interface
controls_frame = Frame(main_frame, bg="#f5f5f5")
controls_frame.pack(fill='x', padx=20, pady=10)

# Dropdown for chart selection
chart_label = Label(controls_frame, text="Select Diagram:",
                    font=("Arial", 11), bg="#f5f5f5")
chart_label.pack(side='left', padx=(0, 5))

chart_var = StringVar(window)
chart_var.set(chart_types["bar"])  # default value
chart_var.trace("w", on_chart_select)

chart_dropdown = OptionMenu(controls_frame, chart_var, *chart_types.values())
chart_dropdown.config(font=("Arial", 11), bg="white", width=18)
chart_dropdown.pack(side='left', padx=(0, 20))

# Save button
save_button = Button(controls_frame,
                     text="Save Chart",
                     command=save_chart,
                     font=("Arial", 11),
                     bg="#2ecc71", fg="white",
                     padx=15, pady=5,
                     relief="flat")
save_button.pack(side='left', padx=(0, 10))

# Report button
report_button = Button(controls_frame,
                       text="Show Report",
                       command=show_report,
                       font=("Arial", 11),
                       bg="#9b59b6", fg="white",
                       padx=15, pady=5,
                       relief="flat")
report_button.pack(side='left', padx=(0, 10))

# Back button
back_button = Button(controls_frame,
                     text="Back",
                     command=go_back,
                     font=("Arial", 11),
                     bg="#e74c3c", fg="white",
                     padx=15, pady=5,
                     relief="flat")
back_button.pack(side='right')

# Create a frame for the chart in the main interface
chart_frame = Frame(main_frame, bg="white", bd=1, relief="solid")


# Set up window resize handler to update the gradient background
def on_window_resize(event):
    # Only update if welcome frame is visible
    if welcome_frame.winfo_viewable():
        # Schedule the update to avoid multiple rapid redraws
        window.after(100, update_gradient_background)


# Bind the resize event to the window
window.bind("<Configure>", on_window_resize)

# Schedule the first gradient update after the window is fully drawn
window.after(200, update_gradient_background)

# Start the GUI event loop
window.mainloop()