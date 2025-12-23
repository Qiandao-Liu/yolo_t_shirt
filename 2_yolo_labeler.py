"""
YOLO image labeling tool
Features: add t-shirt and sleeve labels with bounding box annotation
Output: YOLO-format label files
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import json
from pathlib import Path


class YOLOLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Labeler - T-Shirt & Sleeve")
        self.root.geometry("1400x900")
        
        # Class definitions
        self.classes = {
            0: {"name": "t-shirt", "color": "#FF6B6B"},
            1: {"name": "sleeve", "color": "#4ECDC4"}
        }
        
        # State variables
        self.image_list = []
        self.current_index = 0
        self.current_image = None
        self.current_photo = None
        self.image_path = None
        self.annotations = []  # [(class_id, x_center, y_center, width, height)]
        self.current_class = tk.IntVar(value=0)
        
        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(toolbar, text="📁 Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Button(toolbar, text="⬅ Previous (A)", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Next (D) ➡", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Button(toolbar, text="💾 Save (S)", command=self.save_annotations).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🗑 Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="↩ Undo (Z)", command=self.undo_last).pack(side=tk.LEFT, padx=2)
        
        # Progress display
        self.progress_label = ttk.Label(toolbar, text="0/0")
        self.progress_label.pack(side=tk.RIGHT, padx=10)
        
        # Left - canvas area
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas
        canvas_frame = ttk.LabelFrame(left_frame, text="Annotation Area", padding="5")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right - control panel
        right_frame = ttk.Frame(main_frame, width=280)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Class selection
        class_frame = ttk.LabelFrame(right_frame, text="Class Selection", padding="10")
        class_frame.pack(fill=tk.X, pady=(0, 10))
        
        for class_id, class_info in self.classes.items():
            frame = ttk.Frame(class_frame)
            frame.pack(fill=tk.X, pady=2)
            
            rb = ttk.Radiobutton(frame, text=f"{class_id}: {class_info['name']}", 
                                  variable=self.current_class, value=class_id)
            rb.pack(side=tk.LEFT)
            
            color_label = tk.Label(frame, text="  ██  ", fg=class_info['color'], 
                                   font=('TkDefaultFont', 12))
            color_label.pack(side=tk.LEFT)
            
            # Shortcut hint
            ttk.Label(frame, text=f"[{class_id+1}]", foreground='gray').pack(side=tk.RIGHT)
        
        # Current image annotations
        annotations_frame = ttk.LabelFrame(right_frame, text="Current Image Annotations", padding="10")
        annotations_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create listbox and scrollbar
        list_frame = ttk.Frame(annotations_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.anno_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                        font=('Consolas', 10))
        self.anno_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.anno_listbox.yview)
        
        ttk.Button(annotations_frame, text="Delete Selected",
                   command=self.delete_selected).pack(fill=tk.X, pady=(5, 0))
        
        # Stats
        stats_frame = ttk.LabelFrame(right_frame, text="Stats", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_labels = {}
        stats_items = [
            ("total", "Total Images:", "0"),
            ("labeled", "Labeled:", "0"),
            ("tshirt", "T-shirt Labels:", "0"),
            ("sleeve", "Sleeve Labels:", "0")
        ]
        
        for key, label, default in stats_items:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=1)
            ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
            self.stats_labels[key] = ttk.Label(frame, text=default)
            self.stats_labels[key].pack(side=tk.LEFT)
        
        # Shortcuts
        help_frame = ttk.LabelFrame(right_frame, text="Shortcuts", padding="10")
        help_frame.pack(fill=tk.X)
        
        shortcuts = [
            "A / ←  : Previous",
            "D / →  : Next",
            "S      : Save",
            "Z      : Undo",
            "1      : T-shirt class",
            "2      : Sleeve class",
            "Delete : Delete selected",
            "Mouse drag: Draw box"
        ]
        
        for shortcut in shortcuts:
            ttk.Label(help_frame, text=shortcut, font=('Consolas', 9)).pack(anchor=tk.W)
            
    def setup_bindings(self):
        # Canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("a", lambda e: self.prev_image())
        self.root.bind("d", lambda e: self.next_image())
        self.root.bind("s", lambda e: self.save_annotations())
        self.root.bind("z", lambda e: self.undo_last())
        self.root.bind("1", lambda e: self.current_class.set(0))
        self.root.bind("2", lambda e: self.current_class.set(1))
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.load_images(folder)
            
    def load_images(self, folder):
        self.image_folder = folder
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_list = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ])
        
        if not self.image_list:
            messagebox.showwarning("Warning", "No images found in the folder!")
            return
            
        # Create labels folder
        self.labels_folder = os.path.join(folder, "labels")
        os.makedirs(self.labels_folder, exist_ok=True)
        
        # Create classes.txt
        classes_file = os.path.join(folder, "classes.txt")
        with open(classes_file, 'w') as f:
            for class_id in sorted(self.classes.keys()):
                f.write(f"{self.classes[class_id]['name']}\n")
        
        self.current_index = 0
        self.update_stats()
        self.load_current_image()
        
    def load_current_image(self):
        if not self.image_list:
            return
            
        self.image_path = self.image_list[self.current_index]
        self.current_image = Image.open(self.image_path)
        
        # Load existing annotations
        self.load_annotations()
        
        # Display image
        self.display_image()
        self.update_progress()
        self.update_annotation_list()
        
    def display_image(self):
        if self.current_image is None:
            return
            
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
            
        # Compute scale factor
        img_width, img_height = self.current_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y)
        
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        # Compute offset (centered)
        self.offset_x = (canvas_width - new_width) // 2
        self.offset_y = (canvas_height - new_height) // 2
        
        # Resize image
        display_img = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(display_img)
        
        # Clear canvas and draw
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, 
                                  anchor=tk.NW, image=self.current_photo)
        
        # Draw existing annotations
        self.draw_annotations()
        
    def draw_annotations(self):
        if self.current_image is None:
            return
            
        img_width, img_height = self.current_image.size
        
        for i, (class_id, x_center, y_center, width, height) in enumerate(self.annotations):
            # Convert YOLO format to pixel coords
            x1 = (x_center - width/2) * img_width
            y1 = (y_center - height/2) * img_height
            x2 = (x_center + width/2) * img_width
            y2 = (y_center + height/2) * img_height
            
            # Convert to canvas coords
            canvas_x1 = x1 * self.scale_factor + self.offset_x
            canvas_y1 = y1 * self.scale_factor + self.offset_y
            canvas_x2 = x2 * self.scale_factor + self.offset_x
            canvas_y2 = y2 * self.scale_factor + self.offset_y
            
            color = self.classes[class_id]['color']
            
            # Draw bounding box
            self.canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                         outline=color, width=2, tags=f"anno_{i}")
            
            # Draw label
            label = f"{self.classes[class_id]['name']} ({i})"
            self.canvas.create_text(canvas_x1 + 3, canvas_y1 - 10,
                                    text=label, fill=color, anchor=tk.W,
                                    font=('Arial', 10, 'bold'))
                                    
    def on_mouse_down(self, event):
        if self.current_image is None:
            return
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        
    def on_mouse_drag(self, event):
        if not self.drawing or self.current_image is None:
            return
            
        # Remove temporary rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            
        color = self.classes[self.current_class.get()]['color']
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline=color, width=2, dash=(4, 4)
        )
        
    def on_mouse_up(self, event):
        if not self.drawing or self.current_image is None:
            return
            
        self.drawing = False
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            
        # Compute bounding box (pixel coords)
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Check minimum size
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return
            
        # Convert to image coords
        img_x1 = (x1 - self.offset_x) / self.scale_factor
        img_y1 = (y1 - self.offset_y) / self.scale_factor
        img_x2 = (x2 - self.offset_x) / self.scale_factor
        img_y2 = (y2 - self.offset_y) / self.scale_factor
        
        # Clamp to image bounds
        img_width, img_height = self.current_image.size
        img_x1 = max(0, min(img_x1, img_width))
        img_y1 = max(0, min(img_y1, img_height))
        img_x2 = max(0, min(img_x2, img_width))
        img_y2 = max(0, min(img_y2, img_height))
        
        # Convert to YOLO format (normalized x_center, y_center, width, height)
        x_center = ((img_x1 + img_x2) / 2) / img_width
        y_center = ((img_y1 + img_y2) / 2) / img_height
        width = (img_x2 - img_x1) / img_width
        height = (img_y2 - img_y1) / img_height
        
        # Add annotation
        class_id = self.current_class.get()
        self.annotations.append((class_id, x_center, y_center, width, height))
        
        # Update display
        self.display_image()
        self.update_annotation_list()
        
    def on_canvas_resize(self, event):
        self.display_image()
        
    def load_annotations(self):
        self.annotations = []
        
        if self.image_path is None:
            return
            
        # Find corresponding label file
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        label_path = os.path.join(self.labels_folder, f"{base_name}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        self.annotations.append((class_id, x_center, y_center, width, height))
                        
    def save_annotations(self):
        if self.image_path is None:
            return
            
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        label_path = os.path.join(self.labels_folder, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            for class_id, x_center, y_center, width, height in self.annotations:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
        self.update_stats()
        messagebox.showinfo("Success", f"Annotations saved to:\n{label_path}")
        
    def update_annotation_list(self):
        self.anno_listbox.delete(0, tk.END)
        
        for i, (class_id, x_center, y_center, width, height) in enumerate(self.annotations):
            class_name = self.classes[class_id]['name']
            self.anno_listbox.insert(tk.END, f"[{i}] {class_name}: ({x_center:.3f}, {y_center:.3f})")
            
    def delete_selected(self):
        selection = self.anno_listbox.curselection()
        if selection:
            index = selection[0]
            del self.annotations[index]
            self.display_image()
            self.update_annotation_list()
            
    def undo_last(self):
        if self.annotations:
            self.annotations.pop()
            self.display_image()
            self.update_annotation_list()
            
    def clear_all(self):
        if self.annotations:
            if messagebox.askyesno("Confirm", "Clear all annotations?"):
                self.annotations = []
                self.display_image()
                self.update_annotation_list()
                
    def prev_image(self):
        if self.image_list and self.current_index > 0:
            self.auto_save()
            self.current_index -= 1
            self.load_current_image()
            
    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.auto_save()
            self.current_index += 1
            self.load_current_image()
            
    def auto_save(self):
        """Auto-save current annotations"""
        if self.image_path is None or not self.annotations:
            return
            
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        label_path = os.path.join(self.labels_folder, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            for class_id, x_center, y_center, width, height in self.annotations:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
    def update_progress(self):
        total = len(self.image_list)
        current = self.current_index + 1 if total > 0 else 0
        self.progress_label.config(text=f"{current}/{total}")
        
    def update_stats(self):
        total = len(self.image_list)
        self.stats_labels["total"].config(text=str(total))
        
        if hasattr(self, 'labels_folder') and os.path.exists(self.labels_folder):
            labeled = len([f for f in os.listdir(self.labels_folder) if f.endswith('.txt')])
            self.stats_labels["labeled"].config(text=str(labeled))
            
            # Count annotations per class
            tshirt_count = 0
            sleeve_count = 0
            
            for label_file in os.listdir(self.labels_folder):
                if label_file.endswith('.txt'):
                    with open(os.path.join(self.labels_folder, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id == 0:
                                    tshirt_count += 1
                                elif class_id == 1:
                                    sleeve_count += 1
                                    
            self.stats_labels["tshirt"].config(text=str(tshirt_count))
            self.stats_labels["sleeve"].config(text=str(sleeve_count))


def main():
    root = tk.Tk()
    app = YOLOLabeler(root)
    root.mainloop()


if __name__ == "__main__":
    main()
