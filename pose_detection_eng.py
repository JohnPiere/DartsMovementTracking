import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# List to store balance points
balance_points = []

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (vertex)
    c = np.array(c)  # Last point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def analyze_video(file_path, speed, export_path, progress_var, status_label):
    global balance_points
    balance_points = []
    cap = cv2.VideoCapture(file_path)

    # Get the original dimensions and frame rate of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust the frame rate based on the speed input (percentage to multiplier)
    adjusted_fps = original_fps * (speed / 100.0)

    # Create the output video file path
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    if not export_path:
        export_path = os.path.dirname(file_path)
    output_path = os.path.join(export_path, f"{name}_tracked{ext}")

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_path, fourcc, adjusted_fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert the image back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw the pose annotation on the image
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extracting specific landmarks
            landmarks = results.pose_landmarks.landmark

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate the midpoint between left hip and right hip (stomach point)
            stomach_x = (left_hip[0] + right_hip[0]) / 2
            stomach_y = (left_hip[1] + right_hip[1]) / 2

            # Convert to image coordinates
            balance_point = (int(stomach_x * width), int(stomach_y * height))
            balance_points.append(balance_point)

            # Draw a circle at the stomach point
            cv2.circle(image, balance_point, 5, (0, 0, 255), -1)

            # Draw the movement of the balance point
            for i in range(1, len(balance_points)):
                cv2.line(image, balance_points[i - 1], balance_points[i], (255, 0, 0), 2)

        # Write the frame to the output video
        out.write(image)

        # Update progress bar
        frame_count += 1
        progress_var.set((frame_count / total_frames) * 100)
        root.update_idletasks()

    # Release the capture and video writer
    cap.release()
    out.release()

    # Show completion dialog
    result = messagebox.askquestion("Tracking Complete", f"Tracking complete. The video is saved as: {output_path}\n\nWould you like to open it now?")
    if result == 'yes':
        # Open the video file in the default video player
        subprocess.run(['start', output_path], check=True, shell=True)
    
    # Reset application state
    file_path_label.config(text="No file selected")
    speed_entry.delete(0, tk.END)
    speed_entry.insert(0, "100")
    export_path_entry.delete(0, tk.END)
    progress_var.set(0)
    status_label.config(text="Select a video to start analysis")

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        display_text = f"{folder_name}/{file_name}"
        selected_file.set(file_path)
        file_path_label.config(text=display_text)

def select_export_path():
    path = filedialog.askdirectory()
    export_path_entry.delete(0, tk.END)
    export_path_entry.insert(0, path)

def start_analysis():
    file_path = selected_file.get()
    if file_path and file_path != "No file selected":
        try:
            speed = float(speed_entry.get())
            if speed <= 0:
                raise ValueError("Speed must be greater than 0")
        except ValueError as e:
            messagebox.showerror("Invalid Speed", "Please enter a valid speed as a percentage (greater than 0).")
            return
        export_path = export_path_entry.get()
        # Reset progress bar and status label
        progress_var.set(0)
        status_label.config(text="Rendering...")
        analyze_video(file_path, speed, export_path, progress_var, status_label)

# Create the main window
root = tk.Tk()
root.title("Darts Position Analyzer")
root.resizable(False, False)

# Variables
selected_file = tk.StringVar(value="No file selected")

# Set styles for ttk
style = ttk.Style(root)
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TButton', font=('Helvetica', 12), padding=6)
style.configure('TEntry', font=('Helvetica', 12))

# Create and place the buttons and input fields
main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill=tk.BOTH, expand=True)

# Center content using grid layout
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_rowconfigure(0, weight=1)

select_button = ttk.Button(main_frame, text="Select Video", command=select_file)
select_button.grid(row=0, column=0, pady=(0, 5), sticky='ew')

file_path_label = ttk.Label(main_frame, textvariable=selected_file, font=('Helvetica', 7, 'italic'), foreground="grey", wraplength=600, justify="left")
file_path_label.grid(row=1, column=0, pady=(0, 20), sticky='ew')

speed_label = ttk.Label(main_frame, text="Speed (%):", anchor='center')
speed_label.grid(row=2, column=0, pady=(0, 5), sticky='ew')

speed_entry = ttk.Entry(main_frame, width=5)
speed_entry.grid(row=3, column=0, pady=(0, 20), sticky='ew')
speed_entry.insert(0, "100")

export_path_button = ttk.Button(main_frame, text="Select Export Path", command=select_export_path)
export_path_button.grid(row=4, column=0, pady=(0, 5), sticky='ew')

export_path_entry = ttk.Entry(main_frame, width=40)
export_path_entry.grid(row=5, column=0, pady=(0, 20), sticky='ew')

track_button = ttk.Button(main_frame, text="Let's Track", command=start_analysis)
track_button.grid(row=6, column=0, pady=(0, 20), sticky='ew')

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100)
progress_bar.grid(row=7, column=0, pady=(0, 20), sticky='ew')

status_label = ttk.Label(main_frame, text="Select a video to start analysis", font=('Helvetica', 12))
status_label.grid(row=8, column=0, pady=(0, 20), sticky='ew')
status_label.configure(anchor="center")

# Adjust window size to fit content
root.update_idletasks()
root.geometry(f"{main_frame.winfo_reqwidth() + 40}x{main_frame.winfo_reqheight() + 40}")
root.eval('tk::PlaceWindow . center')

# Add a label for the text
made_by_label = ttk.Label(main_frame, text="Made by John-Piere Kumagai", font=('Helvetica', 8))
made_by_label.grid(row=9, column=0, pady=(20, 0), sticky='ew')
made_by_label.configure(anchor="center")

# Run the GUI main loop
root.mainloop()
