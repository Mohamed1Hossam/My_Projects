import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, simpledialog, ttk
import re

MAX_INPUT_SIZE = 50000

# Define colors for light and night modes
light_mode_colors = {
    "bg": "#F5F5F5",
    "fg": "#333333",
    "text_bg": "#FFFFFF",
    "text_fg": "#000000",
    "button_bg": "#E0E0E0",
    "button_fg": "#000000",
    "status_bg": "#EEEEEE",
    "highlight_bg": "#E8F0FE",
    "line_number_bg": "#F0F0F0",
    "line_number_fg": "#808080",
    "hover_bg": "#E0E0E0"
}

night_mode_colors = {
    "bg": "#2E2E2E",
    "fg": "#FFFFFF",
    "text_bg": "#3E3E3E",
    "text_fg": "#FFFFFF",
    "button_bg": "#4E4E4E",
    "button_fg": "#FFFFFF",
    "status_bg": "#1E1E1E",
    "highlight_bg": "#364554",
    "line_number_bg": "#2A2A2A",
    "line_number_fg": "#A0A0A0",
    "hover_bg": "#404040"
}


class LineNumberedText(scrolledtext.ScrolledText):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a canvas for line numbers
        self.line_numbers = tk.Canvas(
            self.master,
            width=30,
            highlightthickness=0
        )
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)

        # Bind events
        self.bind('<<Modified>>', self.update_line_numbers)
        self.bind('<Configure>', self.update_line_numbers)
        self.bind('<MouseWheel>', self.on_mousewheel)
        self.line_numbers.bind('<Button-1>', self.on_line_number_click)
        self.line_numbers.bind('<Motion>', self.on_line_number_hover)
        self.line_numbers.bind('<Leave>', self.on_line_number_leave)

        # Track last typed position for green text
        self.last_change_start = "1.0"
        self.last_change_end = "1.0"

        # Store line number tags
        self.line_tags = {}
        self.current_hover_tag = None

    def update_line_numbers(self, event=None):
        self.line_numbers.delete("all")
        self.line_tags.clear()

        # Get visible text area
        first_index = self.index("@0,0")
        last_index = self.index(f"@0,{self.winfo_height()}")

        first_line = int(float(first_index))
        last_line = int(float(last_index)) + 1

        # Draw line numbers
        for line_num in range(first_line, last_line):
            y_coord = self.dlineinfo(f"{line_num}.0")
            if y_coord is not None:
                # Create text with right alignment
                tag_name = f"line_{line_num}"
                text_item = self.line_numbers.create_text(
                    25,
                    y_coord[1] + y_coord[3] // 2,
                    text=str(line_num),
                    anchor="e",
                    fill=night_mode_colors["line_number_fg"] if self.night_mode else light_mode_colors["line_number_fg"]
                )
                self.line_tags[tag_name] = {
                    'item': text_item,
                    'line': line_num,
                    'bbox': self.line_numbers.bbox(text_item)
                }

        self.edit_modified(False)

    def on_line_number_click(self, event):
        for tag_info in self.line_tags.values():
            bbox = tag_info['bbox']
            if bbox and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                line_num = tag_info['line']
                self.highlight_line(line_num)
                break

    def highlight_line(self, line_num):
        self.tag_remove("line_highlight", "1.0", tk.END)
        start = f"{line_num}.0"
        end = f"{line_num + 1}.0"
        self.tag_add("line_highlight", start, end)
        self.tag_config("line_highlight",
                        background=night_mode_colors["hover_bg"] if self.night_mode else light_mode_colors["hover_bg"])
        self.see(start)

    def on_line_number_hover(self, event):
        if self.current_hover_tag:
            self.line_numbers.itemconfig(
                self.line_tags[self.current_hover_tag]['item'],
                fill=night_mode_colors["line_number_fg"] if self.night_mode else light_mode_colors["line_number_fg"]
            )
            self.current_hover_tag = None

        for tag_name, tag_info in self.line_tags.items():
            bbox = tag_info['bbox']
            if bbox and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                self.line_numbers.itemconfig(
                    tag_info['item'],
                    fill="#FFD700"
                )
                self.current_hover_tag = tag_name
                self.highlight_line(tag_info['line'])
                break

    def on_line_number_leave(self, event):
        if self.current_hover_tag:
            self.line_numbers.itemconfig(
                self.line_tags[self.current_hover_tag]['item'],
                fill=night_mode_colors["line_number_fg"] if self.night_mode else light_mode_colors["line_number_fg"]
            )
            self.current_hover_tag = None
            self.tag_remove("line_highlight", "1.0", tk.END)

    def on_mousewheel(self, event):
        self.update_line_numbers()


class TextFormatter:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Text Formatter")
        self.night_mode = False

        # Configure main window
        self.root.geometry("1000x800")
        self.setup_gui()
        self.apply_theme()

    def setup_gui(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.create_toolbar()

        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Input section
        self.input_frame = ttk.LabelFrame(self.paned_window, text="Input Text")
        self.paned_window.add(self.input_frame, weight=1)

        # Output section
        self.output_frame = ttk.LabelFrame(self.paned_window, text="Processed Output")
        self.paned_window.add(self.output_frame, weight=1)

        self.create_text_areas()
        self.create_status_bar()

    def create_text_areas(self):
        self.input_text_box = scrolledtext.ScrolledText(
            self.input_frame,
            wrap=tk.WORD,
            undo=True,
            height=10
        )
        self.input_text_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.output_text_box = LineNumberedText(
            self.output_frame,
            wrap=tk.WORD,
            undo=True,
            height=10
        )
        self.output_text_box.pack(fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        self.output_text_box.bind("<KeyRelease>", self.on_key_release)
        self.input_text_box.bind("<Control-f>", lambda e: self.find_text())
        self.output_text_box.bind("<Control-f>", lambda e: self.find_text())

    def create_toolbar(self):
        self.toolbar = ttk.Frame(self.main_container)
        self.toolbar.pack(fill=tk.X, pady=(0, 5))

        # File operations
        self.file_frame = ttk.LabelFrame(self.toolbar, text="File")
        self.file_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.file_frame, text="Open", command=self.insert_text).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.file_frame, text="Save", command=self.save_text).pack(side=tk.LEFT, padx=2)

        # Edit operations
        self.edit_frame = ttk.LabelFrame(self.toolbar, text="Edit")
        self.edit_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.edit_frame, text="Process", command=self.process_text).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.edit_frame, text="Find", command=self.find_text).pack(side=tk.LEFT, padx=2)

        # View operations
        self.view_frame = ttk.LabelFrame(self.toolbar, text="View")
        self.view_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.view_frame, text="Toggle Theme", command=self.toggle_night_mode).pack(side=tk.LEFT, padx=2)

    def create_status_bar(self):
        self.status_bar = ttk.Frame(self.main_container)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)

        self.line_count_label = ttk.Label(self.status_bar, text="Lines: 0")
        self.line_count_label.pack(side=tk.RIGHT)

    def update_line_count(self):
        text = self.output_text_box.get("1.0", tk.END)
        line_count = len(text.splitlines())
        self.line_count_label.config(text=f"Lines: {line_count}")

    def process_text(self):
        input_text = self.input_text_box.get("1.0", tk.END)
        if len(input_text) > MAX_INPUT_SIZE:
            self.output_text_box.delete("1.0", tk.END)
            self.output_text_box.insert(tk.END, "Input exceeds the maximum limit of 50,000 characters.")
            self.status_label.config(text="Error: Input too long")
        else:
            formatted_text = self.format_input(input_text)
            self.output_text_box.delete("1.0", tk.END)
            self.output_text_box.insert(tk.END, formatted_text)
            self.highlight_syntax(self.output_text_box)
            self.update_line_count()
            self.output_text_box.update_line_numbers()
            self.status_label.config(text="Text processed successfully")

    def format_input(self, input_text):
        patterns = [
            (r'(<DATA LABEL="[^"]+">.*?</DATA>)', "\n\\1\n"),
            (r'(<F.{3}>.*?</F.{3}>)', "\n\\1\n"),
            (r'(<PCM_MODULE>)', "\n\n\\1\n"),
            (r'(<BCE_MODULE>)', "\n\n\\1\n"),
            (r'(</NODEID><NODEID>[^<]+)', "\n\\1\n"),
            (r'(<NODEID>)', "\n\n\\1"),
            (r'</CODE><CODE>', '</CODE> <CODE>')
        ]
        for pattern, replacement in patterns:
            input_text = re.sub(pattern, replacement, input_text)
        return input_text

    def highlight_syntax(self, text_widget):
        text_widget.tag_remove("red_tag", "1.0", tk.END)

        patterns = {
            r"(?<=<CODE>)(.*?)(?=</CODE>)": "red_tag",
            r'(?<=<DATA LABEL=")([^"]+)(?=")': "red_tag",
            r"(?<=<F.{3}>)(.*?)(?=</F.{3}>)": "red_tag"
        }

        for pattern, tag in patterns.items():
            for match in re.finditer(pattern, text_widget.get("1.0", tk.END)):
                start_index = text_widget.index(f"1.0+{match.start()}c")
                end_index = text_widget.index(f"1.0+{match.end()}c")
                text_widget.tag_add(tag, start_index, end_index)

        text_widget.tag_config("red_tag", foreground="red")

    def apply_theme(self):
        colors = night_mode_colors if self.night_mode else light_mode_colors
        self.root.configure(bg=colors["bg"])

        self.output_text_box.line_numbers.configure(
            bg=colors["line_number_bg"]
        )
        self.output_text_box.night_mode = self.night_mode

        style = ttk.Style()
        style.configure("TFrame", background=colors["bg"])
        style.configure("TLabelframe", background=colors["bg"])
        style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["fg"])
        style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
        style.configure("TButton", background=colors["button_bg"])

        for text_widget in (self.input_text_box, self.output_text_box):
            text_widget.configure(
                bg=colors["text_bg"],
                fg=colors["text_fg"],
                insertbackground=colors["fg"]
            )

    def toggle_night_mode(self):
        self.night_mode = not self.night_mode
        self.apply_theme()
        self.output_text_box.update_line_numbers()

    def insert_text(self):
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, "rb") as file:
                    file_content = file.read()
                text_content = file_content.decode("utf-8", errors="replace")
                self.input_text_box.delete("1.0", tk.END)
                self.input_text_box.insert(tk.END, text_content)
                self.status_label.config(text=f"Loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_text(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write(self.output_text_box.get("1.0", tk.END))
                self.status_label.config(text=f"Saved: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    def find_text(self):
        search_text = simpledialog.askstring("Find Text", "Enter text to find:")
        if search_text:
            self.output_text_box.tag_remove("found", "1.0", tk.END)
            start_pos = "1.0"
            count = 0

            while True:
                start_pos = self.output_text_box.search(search_text, start_pos, stopindex=tk.END)
                if not start_pos:
                    break
                count += 1
                end_pos = f"{start_pos}+{len(search_text)}c"
                self.output_text_box.tag_add("found", start_pos, end_pos)
                self.output_text_box.see(start_pos)
                start_pos = end_pos

            self.output_text_box.tag_config("found", background="yellow", foreground="black")
            self.status_label.config(text=f"Found {count} matches")

    def on_key_release(self, event):
        self.highlight_syntax(self.output_text_box)
        self.update_line_count()
        self.output_text_box.update_line_numbers()

        if self.output_text_box.edit_modified():
            # Get the last insertion position
            cursor_position = self.output_text_box.index(tk.INSERT)

            # Remove previous green tags
            self.output_text_box.tag_remove("green_tag", "1.0", tk.END)

            # Add green tag to the newly inserted text
            start_index = f"{cursor_position}-1c"
            self.output_text_box.tag_add("green_tag", start_index, cursor_position)

            # Configure the green color
            self.output_text_box.tag_config("green_tag", foreground="green")
            self.output_text_box.edit_modified(False)


if __name__ == "__main__":
    root = tk.Tk()
    app = TextFormatter(root)
    root.mainloop()