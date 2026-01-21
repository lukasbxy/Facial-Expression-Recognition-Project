"""
Facial Expression Recognition GUI für unsere Modell-Demonstration

=== QUICKSTART ===

# 1. Ins Projektverzeichnis navigieren

# 2. Virtuelle Umgebung erstellen
python3 -m venv venv

# 3. Virtuelle Umgebung aktivieren
source venv/bin/activate      # für macOS/Linux
# venv\\Scripts\\activate      # für Windows

# 4. Abhängigkeiten installieren
pip install -r requirements.txt

# 5. GUI starten
python demo/demo_gui.py

# Es kann ESC gedrückt werden um den Fullscreen-Modus zu verlassen
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

# === KONFIGURATION ===
EMOTIONS = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]
COLORS = {"high": "#2d5a27", "mid": "#b58e24", "low": "#942121", "bg": "#1a1a1a"}
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GradCAM:
    """GradCAM für Saliency Map Visualisierung"""
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
    """Hauptanwendung"""
    
    def __init__(self):
        # === FENSTER ===
        self.root = ctk.CTk()
        self.root.title("Facial Expression AI")
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        
        # === STATE ===
        self.W = (self.root.winfo_screenwidth() - 80) // 2
        self.H = self.root.winfo_screenheight() - 220
        self.model = self.gradcam = self.cap = self.video_path = None
        self.running, self.frame_count = False, 0
        self.last_result = ("-", 0, None, None, np.zeros(6))
        self.smooth_probs = np.zeros(6)
        
        # === RESOURCES ===
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self._build_ui()
        self._load_model()

    # === MODELL ===
    def _get_checkpoints(self):
        """Get checkpoints from runs/ResNet18_SE_Variant"""
        pts = []
        
        # Search runs/ResNet18_SE_Variant/*/checkpoints/*.pt (most recent first)
        runs_dir = PROJECT_ROOT / "runs" / "ResNet18_SE_Variant"
        if runs_dir.exists():
            # Get all timestamp directories, sort by name (newest first)
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
        self.status.configure(text=f"Modell bereit: {self.ckpt_var.get()}")

    # === UI ===
    def _build_ui(self):
        # Header
        head = ctk.CTkFrame(self.root, corner_radius=0, height=80)
        head.pack(fill=tk.X)
        ctk.CTkLabel(head, text="FACIAL EXPRESSION MODEL DEMO", font=("Helvetica", 24, "bold")).pack(side=tk.LEFT, padx=30, pady=20)
        
        pts = self._get_checkpoints()
        if not pts:
            messagebox.showwarning("Warnung", "Keine Checkpoints gefunden! Bitte zuerst ein Modell trainieren.")
        self.ckpt_var = ctk.StringVar(value=pts[0] if pts else "")
        self.ckpt_menu = ctk.CTkOptionMenu(head, variable=self.ckpt_var, values=pts if pts else ["Keine Checkpoints"], command=self._load_model, width=300)
        self.ckpt_menu.pack(side=tk.LEFT, padx=20)
        
        # Buttons
        btn_frame = ctk.CTkFrame(head, fg_color="transparent")
        btn_frame.pack(side=tk.RIGHT, padx=30)
        btns = [("Import", self._import, None), ("Export", self._export, COLORS["high"]), 
                ("Webcam", self._webcam, None), ("Stop", self._stop, COLORS["low"]), ("Beenden", self.root.quit, "#444")]
        self.btns = [ctk.CTkButton(btn_frame, text=t, command=c, width=100, fg_color=fg) for t, c, fg in btns]
        for b in self.btns: b.pack(side=tk.LEFT, padx=5)

        # Main Area
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Canvas Original
        f1 = ctk.CTkFrame(main)
        f1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        ctk.CTkLabel(f1, text="Original & Emotion", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.c1 = tk.Canvas(f1, width=int(self.W*0.75), height=self.H, bg=COLORS["bg"], highlightthickness=0)
        self.c1.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Stats Panel
        stats_outer = ctk.CTkFrame(main, fg_color="transparent", width=220)
        stats_outer.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        stats = ctk.CTkFrame(stats_outer, width=200)
        stats.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(stats, text="STATISTIKEN", font=("Helvetica", 14, "bold")).pack(pady=(15, 10))
        
        self.bars = {}
        for emo in EMOTIONS:
            f = ctk.CTkFrame(stats, fg_color="transparent")
            f.pack(fill=tk.X, padx=15, pady=6)
            ctk.CTkLabel(f, text=emo, font=("Helvetica", 11)).pack(anchor="w")
            bar = ctk.CTkProgressBar(f, height=8)
            bar.pack(fill=tk.X, pady=2)
            bar.set(0)
            self.bars[emo] = bar

        # Canvas Saliency
        f2 = ctk.CTkFrame(main)
        f2.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        ctk.CTkLabel(f2, text="Saliency Map (GradCAM)", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.c2 = tk.Canvas(f2, width=int(self.W*0.75), height=self.H, bg=COLORS["bg"], highlightthickness=0)
        self.c2.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Status
        self.status = ctk.CTkLabel(self.root, text="System bereit | 'ESC' für Fenstermodus", font=("Helvetica", 14))
        self.status.pack(pady=10)

    def _lock_ui(self, lock=True):
        state = "disabled" if lock else "normal"
        for b in self.btns: b.configure(state=state)
        self.ckpt_menu.configure(state=state)

    # === VERARBEITUNGSCODE ===
    def _process(self, frame, skip=False):
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
        
        # Statistik-Bars aktualisieren
        self.smooth_probs = 0.3 * probs + 0.7 * self.smooth_probs
        for i, e in enumerate(EMOTIONS):
            p = self.smooth_probs[i]
            self.bars[e].configure(progress_color=COLORS["high"] if p > 0.6 else COLORS["mid"] if p > 0.3 else COLORS["low"])
            self.bars[e].set(p)
            
        x, y, fw, fh, x1, y1, x2, y2 = self.last_result[2]
        hm = self.last_result[3]
        
        # Label zeichnen
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
        
        # Stats im Bild (gilt nur für Export)
        tw = 180
        cv2.rectangle(orig, (w - tw, 0), (w, 220), (30, 30, 30), -1)
        cv2.putText(orig, "STATS", (w - tw + 10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)
        for i, (e, prob) in enumerate(zip(EMOTIONS, probs)):
            yy = 55 + i * 28
            cv2.putText(orig, f"{e[:4]}:", (w - tw + 10, yy), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(orig, (w - tw + 60, yy - 8), (w - 15, yy + 2), (60, 60, 60), -1)
            cv2.rectangle(orig, (w - tw + 60, yy - 8), (w - tw + 60 + int(105 * prob), yy + 2), (76, 175, 80), -1)

        # Saliency Map
        hmc = cv2.applyColorMap(cv2.resize(hm, (x2-x1, y2-y1)), cv2.COLORMAP_JET)
        sal[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, hmc, 0.5, 0)
        cv2.rectangle(sal, (x, y), (x+fw, y+fh), (255, 255, 255), 1)
        return orig, sal, emo, conf, probs

    def _show(self, left, right):
        for canvas, img in [(self.c1, left), (self.c2, right)]:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            s = min(self.W / rgb.shape[1], self.H / rgb.shape[0])
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))))
            canvas.delete("all")
            canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=photo)
            canvas.image = photo

    # === AKTIONEN ===
    def _import(self):
        if not self.model: return
        self._stop()
        if path := filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")]):
            self.video_path = path
            threading.Thread(target=self._preview, args=(path,), daemon=True).start()

    def _preview(self, path):
        cap = cv2.VideoCapture(path)
        for n in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, f = cap.read()
            if not ret: break
            l, r, e, c, _ = self._process(f, skip=(n % 3 != 0))
            if n % 2 == 0:
                self._show(l, r)
                self.status.configure(text=f"Vorschau: {e} ({c*100:.1f}%) | Frame {n}")
                self.root.update()
        cap.release()
        self.status.configure(text="Vorschau beendet. Export bereit.")

    def _export(self):
        if not self.video_path: return messagebox.showwarning("Warnung", "Bitte Video importieren.")
        if out := filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")]):
            threading.Thread(target=self._run_export, args=(out,), daemon=True).start()

    def _run_export(self, out_path):
        self._lock_ui(True)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))
        
        for n in range(total):
            ret, f = cap.read()
            if not ret: break
            l, r, _, _, _ = self._process(f, skip=(n % 5 != 0))
            out.write(np.hstack([l, r]))
            if n % 10 == 0:
                self.status.configure(text=f"EXPORT: {n}/{total} ({n*100//total}%)")
                self.root.update()
        
        cap.release()
        out.release()
        self._lock_ui(False)
        self.status.configure(text=f"Export fertig: {out_path}")
        messagebox.showinfo("Erfolg", f"Video gespeichert:\n{out_path}")

    def _webcam(self):
        if not self.model or self.running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return messagebox.showerror("Fehler", "Keine Webcam!")
        self.running, self.frame_count = True, 0
        self._loop()

    def _loop(self):
        if not self.running: return
        ret, f = self.cap.read()
        if ret:
            l, r, e, c, _ = self._process(f, skip=(self.frame_count % 3 != 0))
            self._show(l, r)
            self.status.configure(text=f"Live: {e} ({c*100:.1f}%)")
            self.frame_count += 1
        self.root.after(20, self._loop)

    def _stop(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None
        for b in self.bars.values(): b.set(0)
        self.c1.delete("all"); self.c2.delete("all")
        self.status.configure(text="Gestoppt")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
