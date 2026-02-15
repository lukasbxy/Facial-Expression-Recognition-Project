r"""
Facial Expression Recognition GUI for live model inference

---How to use---

1. Change into project directory

2. Create virtual environment
macOS / Linux: python3 -m venv venv
Windows: python -m venv venv

3. Activate virtual environment
macOS / Linux: source venv/bin/activate      
Windows: venv\Scripts\activate      

4. Install requirements
pip: pip install -r requirements.txt

5. Start GUI
python demo/demo_gui.py

6. Press Escape to leave Full-Screen Mode
"""
from pathlib import Path
import sys, threading
import cv2, torch, numpy as np
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models import ResNet18_SE_Variant 
EMOTIONS = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]
COLORS = {"high": "#2d5a27", "mid": "#b58e24", "low": "#942121", "bg": "#1a1a1a"}
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GradCAM:
    """GradCAM script for visualizing saliency maps"""
    def __init__(self, model, layer):
        self.model, self.grads, self.acts = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'acts', o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def __call__(self, x):
        out = self.model(x)
        idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, idx].backward(retain_graph=True)
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate((cam - cam.min()) / (cam.max() + 1e-8), (64, 64), mode='bilinear', align_corners=False)
        return (cam.squeeze().cpu().numpy() * 255).astype(np.uint8), idx, F.softmax(out, 1)[0, idx].item()


class App:
    """
    Main Application used for building the GUI and then doing live inference with selected checkpoints.
    """
    
    def __init__(self):
        #Initialize main window
        self.root = ctk.CTk()
        if sys.platform.startswith("win"):
            self.root.tk.call("tk", "scaling", 1.0)
        self.root.title("Facial Expression AI")
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        self.root.update_idletasks()
        self.root.update()
        
        self.W = (self.root.winfo_screenwidth() - 80) // 2
        self.H = self.root.winfo_screenheight() - 220
        self.model = None
        self.gradcam = None 
        self.cap = None 
        self.video_path = None
        self.running = False 
        self.preview_running = False
        self.frame_count = 0
        self.last_result = ("-", 0, None, None, np.zeros(6))
        self.smooth_probs = np.zeros(6)
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self._build_ui()
        self._load_model()

    def _get_checkpoints(self):
        """
        Get checkpoints from runs/ResNet18_SE_Variant/timestamp/checkpoints/*
        """
        pts = []
        
        runs_dir = PROJECT_ROOT / "runs" / "ResNet18_SE_Variant"
        if runs_dir.exists():
            timestamp_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
            for timestamp_dir in timestamp_dirs:
                checkpoints_dir = timestamp_dir / "checkpoints"
                if checkpoints_dir.exists():
                    for p in checkpoints_dir.glob("*.pt"):
                        pts.append(str(p.relative_to(PROJECT_ROOT)))
        
        return pts if pts else []

    def _load_model(self, _=None):
        path = PROJECT_ROOT / self.ckpt_var.get()
        if not path.exists(): return
        self.model = ResNet18_SE_Variant(num_classes=6)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False)['model_state'])
        self.model.to(self.device).eval()
        self.gradcam = GradCAM(self.model, self.model.layer4)
        self.status.configure(text=f"Model ready: {self.ckpt_var.get()}")

    # building GUI
    def _build_ui(self):
        # Header
        head = ctk.CTkFrame(self.root, corner_radius=0)
        head.pack(side = tk.TOP, fill=tk.X)

        head.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(head, text="FACIAL EXPRESSION MODEL DEMO", font=("Helvetica", 24, "bold"), anchor = "center")
        title.grid(row=0, column=0, sticky="ew", padx=20, pady=(18,8))

        controls = ctk.CTkFrame(head, fg_color="transparent")
        controls.grid(row=1,column=0, sticky="ew", padx= 20, pady=(0,12))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=0)

        pts = self._get_checkpoints()
        if not pts:
            messagebox.showwarning("Error", "No checkpoints found. Please train a model first.")
        
        self.ckpt_var = ctk.StringVar(value=pts[0] if pts else "")
        self.ckpt_menu = ctk.CTkOptionMenu(controls, variable=self.ckpt_var, values=pts if pts else ["No checkpoints"], command=self._load_model, width=300)
        self.ckpt_menu.grid(row=0, column=0, sticky="ew", padx=(10,20))
        
        # Buttons
        btn_frame = ctk.CTkFrame(controls, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e", padx=(0,10))

        buttons = [("Import", self._import, None), ("Export", self._export, COLORS["high"]), 
                ("Webcam", self._webcam, None), ("Stop", self._stop, COLORS["low"]), ("Exit", self.root.quit, "#444")]
        
        self.btns= []
        for i, (t,c,fg) in enumerate (buttons):
            b = ctk.CTkButton(btn_frame, text=t, command=c, width=100, fg_color=fg)
            b.grid(row=0, column=i, padx=5)
            self.btns.append(b)


        # main area with 3 columns
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1, uniform='cols')
        main.grid_columnconfigure(1, weight=0)
        main.grid_columnconfigure(2, weight=1, uniform='cols')

        # Left column with original
        f1 = ctk.CTkFrame(main)
        f1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(f1, text="Original & Emotion", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.c1 = tk.Canvas(f1, bg=COLORS["bg"], highlightthickness=0)
        self.c1.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Middle column with stats
        stats_outer = ctk.CTkFrame(main, fg_color="transparent", width=220)
        stats_outer.grid(row=0, column=1, sticky="ns", padx=10, pady=5)

        stats = ctk.CTkFrame(stats_outer, width=200)
        stats.pack(expand=True)
        ctk.CTkLabel(stats, text="Stats", font=("Helvetica", 14, "bold")).pack(pady=(15, 10))
        
        self.bars = {}
        for emo in EMOTIONS:
            row = ctk.CTkFrame(stats, fg_color="transparent")
            row.pack(fill=tk.X, padx=15, pady=6)
            ctk.CTkLabel(row, text=emo, font=("Helvetica", 11)).pack(anchor="w")
            bar = ctk.CTkProgressBar(row, height=8)
            bar.pack(fill=tk.X, pady=2)
            bar.set(0)
            self.bars[emo] = bar

        # Right column with saliency
        f2 = ctk.CTkFrame(main)
        f2.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(f2, text="Saliency Map (GradCAM)", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.c2 = tk.Canvas(f2, bg=COLORS["bg"], highlightthickness=0)
        self.c2.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Status
        bottom = ctk.CTkFrame(self.root, fg_color="transparent")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0,10))
        bottom.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bottom, text="System ready | Press Escape for Exit", font=("Helvetica", 14), anchor='w')
        self.status.grid(row=0, column=0, sticky="ew", pady=(0,6))

        self.export_bar = ctk.CTkProgressBar(bottom)
        self.export_bar.set(0)
        self.export_bar.grid(row=1, column=0, sticky ="ew")
        self.export_bar.grid_remove()

    def _lock_ui(self, lock=True):
        state = "disabled" if lock else "normal"
        for b in self.btns: b.configure(state=state)
        self.ckpt_menu.configure(state=state)

    # processing
    def _process(self, frame, skip=False, draw_stats = False):
        if not self.model: return frame, frame, "-", 0, [0]*6
        h, w = frame.shape[:2]
        orig, sal = frame.copy(), frame.copy()
        
        if skip and self.last_result[2]:
            emo, conf, probs = self.last_result[0], self.last_result[1], self.last_result[4]
        else:
            faces = self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))
            if len(faces) == 0: return orig, sal, "-", 0, [0]*6
            
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            p = int(0.1 * min(fw, fh))
            y1, y2, x1, x2 = max(0, y-p), min(h, y+fh+p), max(0, x-p), min(w, x+fw+p)
            roi = frame[y1:y2, x1:x2]
            
            t = torch.from_numpy(cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (64, 64))).float().permute(2, 0, 1).unsqueeze(0) / 255
            with torch.no_grad():
                probs = F.softmax(self.model(t.to(self.device)), 1)[0].cpu().numpy()
            
            hm, idx, conf = self.gradcam(t.to(self.device))
            emo = EMOTIONS[idx]
            self.last_result = (emo, conf, (x, y, fw, fh, x1, y1, x2, y2), hm, probs)
        
        self.smooth_probs = 0.3 * probs + 0.7 * self.smooth_probs
        smooth_idx = int(self.smooth_probs.argmax())
        emo = EMOTIONS[smooth_idx]
        conf = float(self.smooth_probs[smooth_idx])
        for i, e in enumerate(EMOTIONS):
            p = self.smooth_probs[i]
            self.bars[e].configure(progress_color=COLORS["high"] if p > 0.6 else COLORS["mid"] if p > 0.3 else COLORS["low"])
            self.bars[e].set(p)
            
        x, y, fw, fh, x1, y1, x2, y2 = self.last_result[2]
        hm = self.last_result[3]
        

        fs, th = max(0.6, min(fw, fh) / 100), max(1, int(max(0.6, min(fw, fh) / 100) * 2))
        lbl = f"  {emo}: {conf*100:.0f}%  "
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        lx = max(5, min(x, w - lw - 5))
        ly = max(lh + 10, min(y - 15 if y - 15 >= lh + 20 else y + fh + lh + 20, h - 10))
        
        overlay = orig.copy()
        cv2.rectangle(overlay, (lx, ly - lh - 10), (lx + lw, ly + 10), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, orig, 0.3, 0, orig)
        cv2.rectangle(orig, (x, y), (x+fw, y+fh), (76, 175, 80), 2)
        cv2.putText(orig, lbl, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        
        # stats only in export
        if draw_stats:
            tw = 180
            cv2.rectangle(orig, (w - tw, 0), (w, 220), (30, 30, 30), -1)
            cv2.putText(orig, "STATS", (w - tw + 10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)
            for i, (e, prob) in enumerate(zip(EMOTIONS, probs)):
                yy = 55 + i * 28
                cv2.putText(orig, f"{e[:4]}:", (w - tw + 10, yy), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
                cv2.rectangle(orig, (w - tw + 60, yy - 8), (w - 15, yy + 2), (60, 60, 60), -1)
                cv2.rectangle(orig, (w - tw + 60, yy - 8), (w - tw + 60 + int(105 * prob), yy + 2), (76, 175, 80), -1)


        hmc = cv2.applyColorMap(cv2.resize(hm, (x2-x1, y2-y1)), cv2.COLORMAP_JET)
        sal[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, hmc, 0.5, 0)
        cv2.rectangle(sal, (x, y), (x+fw, y+fh), (255, 255, 255), 1)
        return orig, sal, emo, conf, probs

    def _show(self, left, right):
        for canvas, img in [(self.c1, left), (self.c2, right)]:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cw = max(1, canvas.winfo_width())
            ch = max(1, canvas.winfo_height())
            s = min(cw / rgb.shape[1], ch / rgb.shape[0])
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))))
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=photo)
            canvas.image = photo

    # actions for buttons
    def _import(self):
        if not self.model: return
        self._stop()
        if path := filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")]):
            self.video_path = path
            threading.Thread(target=self._preview, args=(path,), daemon=True).start()

    def _preview(self, path):
        cap = cv2.VideoCapture(path)
        self.preview_running = True
        for n in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            if self.preview_running == False:
                break
            ret, f = cap.read()
            if not ret: break
            l, r, e, c, _ = self._process(f, skip=(n % 3 != 0), draw_stats= False)
            if n % 2 == 0:
                self.root.after(0, lambda l=l, r=r, e=e, c=c, n=n: (
                self._show(l, r),
                self.status.configure(text=f"Preview: {e} ({c*100:.1f}%) | Frame {n}")
                ))
        cap.release()
        self.status.configure(text="Preview stopped. Ready for export.")

    def _export(self):
        if not self.video_path: return messagebox.showwarning("Warning", "Please import video.")
        if out := filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")]):
            threading.Thread(target=self._run_export, args=(out,), daemon=True).start()

    def _run_export(self, out_path):
        self.preview_running = False
        self._lock_ui(True)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))
        self.root.after(0, lambda: (self.export_bar.set(0), self.export_bar.grid()))

        for n in range(total):
            ret, f = cap.read()
            if not ret: break
            l, r, _, _, _ = self._process(f, skip=(n % 5 != 0), draw_stats=True)
            out.write(np.hstack([l, r]))
            if n % 10 == 0:
                p = n/max(1,total)
                self.root.after(0, lambda p=p, n=n, total=total : (
                self.export_bar.set(p),
                self.status.configure(text=f"EXPORT: {n}/{total} ({int(p*100)}%)")
                ))
        
        cap.release()
        out.release()
        self.root.after(0, lambda: self.export_bar.grid_remove())
        self._lock_ui(False)
        self.status.configure(text=f"Export finished: {out_path}")
        messagebox.showinfo("Success", f"Video saved at:\n{out_path}")

    def _webcam(self):
        if not self.model or self.running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return messagebox.showerror("Error", "No Webcam!")
        self.running, self.frame_count = True, 0
        self._loop()

    def _loop(self):
        if not self.running: return
        ret, f = self.cap.read()
        if ret:
            l, r, e, c, _ = self._process(f, skip=(self.frame_count % 3 != 0), draw_stats=False)
            self._show(l, r)
            self.status.configure(text=f"Live: {e} ({c*100:.1f}%)")
            self.frame_count += 1
        self.root.after(20, self._loop)

    def _stop(self):
        self.running = False
        self.preview_running = False
        if self.cap: self.cap.release(); self.cap = None
        for b in self.bars.values(): b.set(0)
        self.c1.delete("all"); self.c2.delete("all")
        self.status.configure(text="Stopped")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
