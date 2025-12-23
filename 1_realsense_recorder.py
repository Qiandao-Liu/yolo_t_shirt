"""
RealSense D435
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import time
from datetime import datetime
import threading


class RealSenseRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense D435")
        self.root.geometry("1200x800")
        
        self.is_running = False
        self.is_recording = False
        self.pipeline = None
        self.config = None
        self.frame_count = 0
        self.capture_count = 0
        self.capture_interval = tk.IntVar(value=30)  # in 1/30 fps
        self.save_path = tk.StringVar(value="./captured_images")
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.LabelFrame(main_frame, text="preview", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(left_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        camera_frame = ttk.LabelFrame(right_frame, text="cam contorl", padding="10")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(camera_frame, text="start cam", command=self.toggle_camera)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        settings_frame = ttk.LabelFrame(right_frame, text="settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="capture interval:").pack(anchor=tk.W)
        interval_frame = ttk.Frame(settings_frame)
        interval_frame.pack(fill=tk.X, pady=2)
        
        self.interval_scale = ttk.Scale(interval_frame, from_=1, to=120, 
                                         variable=self.capture_interval, orient=tk.HORIZONTAL)
        self.interval_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.interval_label = ttk.Label(interval_frame, text="30", width=4)
        self.interval_label.pack(side=tk.RIGHT)
        self.interval_scale.configure(command=self.update_interval_label)
        
        ttk.Label(settings_frame, text="save path:").pack(anchor=tk.W, pady=(10, 0))
        path_frame = ttk.Frame(settings_frame)
        path_frame.pack(fill=tk.X, pady=2)
        
        ttk.Entry(path_frame, textvariable=self.save_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="preview", command=self.browse_path, width=6).pack(side=tk.RIGHT)
        
        record_frame = ttk.LabelFrame(right_frame, text="recording control", padding="10")
        record_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.record_btn = ttk.Button(record_frame, text="Start Recording", command=self.toggle_recording)
        self.record_btn.pack(fill=tk.X, pady=2)
        self.record_btn.state(['disabled'])
        
        ttk.Button(record_frame, text="Manual Capture", command=self.manual_capture).pack(fill=tk.X, pady=2)
        
        # Status info
        status_frame = ttk.LabelFrame(right_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_labels = {}
        status_items = [
            ("camera", "Camera:", "Disconnected"),
            ("recording", "Recording:", "Stopped"),
            ("frames", "Frames:", "0"),
            ("captured", "Captured:", "0"),
            ("fps", "FPS:", "0 FPS")
        ]
        
        for key, label, default in status_items:
            frame = ttk.Frame(status_frame)
            frame.pack(fill=tk.X, pady=1)
            ttk.Label(frame, text=label, width=10).pack(side=tk.LEFT)
            self.status_labels[key] = ttk.Label(frame, text=default)
            self.status_labels[key].pack(side=tk.LEFT)
        
        # Recent capture preview
        preview_frame = ttk.LabelFrame(right_frame, text="Recent Capture", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_label = ttk.Label(preview_frame, text="No Image")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Footer info
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(info_frame, text="Tip: Auto capture during recording by interval\nManual capture saves the current frame anytime",
                  font=('TkDefaultFont', 9), foreground='gray').pack()
        
    def update_interval_label(self, value):
        self.interval_label.config(text=str(int(float(value))))
        
    def browse_path(self):
        path = filedialog.askdirectory(title="Select Save Directory")
        if path:
            self.save_path.set(path)
            
    def toggle_camera(self):
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure RGB stream
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            # Optional: configure depth stream
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            
            self.pipeline.start(self.config)
            self.is_running = True
            self.start_btn.config(text="Stop Camera")
            self.record_btn.state(['!disabled'])
            self.status_labels["camera"].config(text="Connected", foreground='green')
            
            # Start update thread
            self.update_thread = threading.Thread(target=self.update_frame, daemon=True)
            self.update_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
            
    def stop_camera(self):
        self.is_running = False
        self.is_recording = False
        if self.pipeline:
            self.pipeline.stop()
        self.start_btn.config(text="Start Camera")
        self.record_btn.config(text="Start Recording")
        self.record_btn.state(['disabled'])
        self.status_labels["camera"].config(text="Disconnected", foreground='black')
        self.status_labels["recording"].config(text="Stopped", foreground='black')
        
    def toggle_recording(self):
        if not self.is_recording:
            # Ensure save directory exists
            save_dir = self.save_path.get()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            self.status_labels["recording"].config(text="Recording...", foreground='red')
        else:
            self.is_recording = False
            self.record_btn.config(text="Start Recording")
            self.status_labels["recording"].config(text="Stopped", foreground='black')
            
    def update_frame(self):
        fps_counter = 0
        fps_time = time.time()
        
        while self.is_running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                self.current_frame = color_image.copy()
                
                self.frame_count += 1
                fps_counter += 1
                
                # Calculate FPS
                if time.time() - fps_time >= 1.0:
                    self.root.after(0, lambda fps=fps_counter: 
                                    self.status_labels["fps"].config(text=f"{fps} FPS"))
                    fps_counter = 0
                    fps_time = time.time()
                
                # Auto capture
                if self.is_recording and self.frame_count % self.capture_interval.get() == 0:
                    self.save_image(color_image)
                
                # Update display
                display_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                display_image = cv2.resize(display_image, (800, 600))
                
                # Add overlay info
                if self.is_recording:
                    cv2.circle(display_image, (30, 30), 15, (255, 0, 0), -1)
                    cv2.putText(display_image, "REC", (50, 38), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                cv2.putText(display_image, f"Frame: {self.frame_count}", (10, 580),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                photo = ImageTk.PhotoImage(Image.fromarray(display_image))
                
                self.root.after(0, self.update_video_label, photo)
                self.root.after(0, lambda: self.status_labels["frames"].config(text=str(self.frame_count)))
                
            except Exception as e:
                print(f"Frame update error: {e}")
                break
                
    def update_video_label(self, photo):
        self.video_label.config(image=photo)
        self.video_label.image = photo
        
    def save_image(self, image):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(self.save_path.get(), filename)
        
        cv2.imwrite(filepath, image)
        self.capture_count += 1
        
        self.root.after(0, lambda: self.status_labels["captured"].config(
            text=str(self.capture_count)))
        
        # Update preview
        preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preview = cv2.resize(preview, (200, 150))
        preview_photo = ImageTk.PhotoImage(Image.fromarray(preview))
        self.root.after(0, self.update_preview, preview_photo)
        
    def update_preview(self, photo):
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo
        
    def manual_capture(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            save_dir = self.save_path.get()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_image(self.current_frame)
            messagebox.showinfo("Success", "Image saved!")
        else:
            messagebox.showwarning("Warning", "Please start the camera first!")
            
    def on_closing(self):
        self.is_running = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = RealSenseRecorder(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
