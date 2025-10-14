import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def round_robin(process_names, burst_times, quantum):
    n = len(process_names)
    remaining = burst_times[:]
    arrival_times = [0] * n
    current_time = 0
    wt, tat = [0] * n, [0] * n
    chart = []

    appearances = {i: [] for i in range(n)}

    while any(t > 0 for t in remaining):
        for i in range(n):
            if remaining[i] > 0:
                time_slice = min(quantum, remaining[i])

                chart.append((process_names[i], current_time, current_time + time_slice))
                appearances[i].append(current_time)

                current_time += time_slice
                remaining[i] -= time_slice

                if remaining[i] == 0:
                    tat[i] = current_time
                    wt[i] = tat[i] - burst_times[i]

    rt = []
    for i in range(n):
        if burst_times[i] > quantum:
            if len(appearances[i]) >= 2:
                rt.append(appearances[i][1] - quantum)
            else:
                rt.append(appearances[i][0])
        else:
            rt.append(appearances[i][0])

    avg_wt = sum(wt) / n
    avg_tat = sum(tat) / n
    avg_rt = sum(rt) / n
    return wt, tat, rt, avg_wt, avg_tat, avg_rt, chart


def draw_chart(frame, chart):
    for widget in frame.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, chart[-1][2] + 1)
    ax.set_ylim(0, 30)
    ax.axis('off')
    for name, start, end in chart:
        ax.broken_barh([(start, end - start)], (10, 10), facecolors='skyblue')
        ax.text((start + end) / 2, 15, name, ha='center', va='center')
        ax.text(start, 8, str(start), ha='center', fontsize=8)
    ax.text(chart[-1][2], 8, str(chart[-1][2]), ha='center', fontsize=8)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


def run():
    try:
        num = int(entry_count.get())
        times = list(map(int, entry_bursts.get().split()))
        quantum = int(entry_quantum.get())
        if len(times) != num or any(t < 0 for t in times):
            raise ValueError
        names = [f"P{i + 1}" for i in range(num)]
        wt, tat, rt, awt, atat, art, chart = round_robin(names, times, quantum)

        result_text = "Process\tWT\tTAT\tRT\n"
        for i in range(num):
            result_text += f"{names[i]}\t{wt[i]}\t{tat[i]}\t{rt[i]}\n"
        result_text += f"\nAverage WT: {awt:.2f}\nAverage TAT: {atat:.2f}\nAverage RT: {art:.2f}"

        results.set(result_text)
        draw_chart(chart_frame, chart)
    except Exception as e:
        messagebox.showerror("Error", f"Check Inputs.\n\n{e}")


root = tk.Tk()
root.title("Round Robin Scheduler")

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

tk.Label(input_frame, text="Number of Processes:").grid(row=0, column=0, padx=5, pady=5)
entry_count = tk.Entry(input_frame, width=20)
entry_count.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Burst Times (space separated):").grid(row=1, column=0, padx=5, pady=5)
entry_bursts = tk.Entry(input_frame, width=20)
entry_bursts.grid(row=1, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Quantum:").grid(row=2, column=0, padx=5, pady=5)
entry_quantum = tk.Entry(input_frame, width=20)
entry_quantum.grid(row=2, column=1, padx=5, pady=5)

tk.Button(input_frame, text="Run", command=run).grid(row=3, column=0, columnspan=2, pady=10)

results = tk.StringVar()
tk.Label(root, textvariable=results, font=("Arial", 12), justify="left").pack()

chart_frame = tk.Frame(root, bd=2, relief="sunken", padx=10, pady=10)
chart_frame.pack(pady=10)

root.mainloop()