import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys    
import threading
import joblib
from PIL import Image, ImageTk
from collections import deque
import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)

from feature_extractor import extract_static_features, extract_dynamic_features

# ============================================================
# LABELS  -  must match exactly what train_dynamic.py printed
# ============================================================
STATIC_LABELS = [
    'A','B','C','D','E','F','G','I','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

# After running train_dynamic.py, verify this order matches
# what it printed as "Label order: [...]"
DYNAMIC_LABELS = sorted([
    'H','J','hello','bye','namaste','practice','thank_you','sorry'
])

SEQUENCE_LEN        = 30
STATIC_CONF_THRESH  = 0.70
DYNAMIC_CONF_THRESH = 0.75
MOVEMENT_THRESH     = 0.02   # below this = static sign

# ============================================================
# THEME
# ============================================================
THEMES = {
    "dark": {
        "bg":         "#0D1117",
        "panel":      "#161B22",
        "card":       "#21262D",
        "border":     "#30363D",
        "accent":     "#58A6FF",
        "accent2":    "#3FB950",
        "accent3":    "#F78166",
        "text":       "#E6EDF3",
        "text_dim":   "#8B949E",
        "btn_bg":     "#21262D",
        "btn_active": "#388BFD",
        "danger":     "#DA3633",
        "warn":       "#D29922",
        "hist_bg":    "#161B22",
        "hist_fg":    "#C9D1D9",
        "scroll":     "#30363D",
    },
    "light": {
        "bg":         "#F6F8FA",
        "panel":      "#FFFFFF",
        "card":       "#F0F3F6",
        "border":     "#D0D7DE",
        "accent":     "#0969DA",
        "accent2":    "#1A7F37",
        "accent3":    "#CF222E",
        "text":       "#1F2328",
        "text_dim":   "#656D76",
        "btn_bg":     "#FFFFFF",
        "btn_active": "#0969DA",
        "danger":     "#CF222E",
        "warn":       "#9A6700",
        "hist_bg":    "#F6F8FA",
        "hist_fg":    "#1F2328",
        "scroll":     "#D0D7DE",
    }
}

SIDEBAR_W  = 320
FT_TITLE   = ("Segoe UI", 11, "bold")
FT_BODY    = ("Segoe UI", 10)
FT_SMALL   = ("Segoe UI", 9)
FT_MONO    = ("Consolas", 10)
FT_PRED    = ("Segoe UI", 36, "bold")


# ============================================================
# SENTENCE BUILDER
# ============================================================
class SentenceBuilder:
    def __init__(self, confirm_frames=20, conf_thresh=0.70):
        self.confirm_frames = confirm_frames
        self.conf_thresh    = conf_thresh
        self.sentence       = []
        self._streak_label  = None
        self._streak_count  = 0

    def feed(self, label, confidence):
        if confidence < self.conf_thresh:
            self._streak_label = None
            self._streak_count = 0
            return
        if label == self._streak_label:
            self._streak_count += 1
        else:
            self._streak_label = label
            self._streak_count = 1
        if self._streak_count == self.confirm_frames:
            if not self.sentence or self.sentence[-1] != label:
                self.sentence.append(label)
            self._streak_count = 0

    def get(self):
        return " ".join(self.sentence)

    def delete_last(self):
        if self.sentence:
            self.sentence.pop()

    def clear(self):
        self.sentence = []
        self._streak_label = None
        self._streak_count = 0


# ============================================================
# SMOOTHER
# ============================================================
class Smoother:
    def __init__(self, window=10):
        self._buf = []
        self._win = window

    def update(self, label):
        self._buf.append(label)
        if len(self._buf) > self._win:
            self._buf.pop(0)

    def get(self):
        if not self._buf:
            return None
        from collections import Counter
        return Counter(self._buf).most_common(1)[0][0]

    def reset(self):
        self._buf = []


# ============================================================
# APP
# ============================================================
class ISLApp:

    def __init__(self, root):
        self.root = root
        self.root.title("ISL Detector - Indian Sign Language")
        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)

        self._theme_name = "dark"
        self.T = THEMES["dark"]

        # State
        self.running        = False
        self.cap            = None
        self.zoom_level     = 1.0
        self.recording      = False
        self.video_writer   = None
        self.rec_start      = None
        self._photo         = None
        self.history_log    = []

        # Models
        self.static_model   = None
        self.dynamic_model  = None
        self._models_loaded = False

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face  = mp.solutions.face_mesh
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands_model = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.face_model = self.mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        # Inference state
        self.dyn_buffer      = deque(maxlen=SEQUENCE_LEN)
        self.prev_features   = None
        self.static_smoother = Smoother(window=12)
        self.dynamic_smoother= Smoother(window=6)
        self.sentence_builder= SentenceBuilder(
            confirm_frames=20, conf_thresh=STATIC_CONF_THRESH
        )
        self.current_label = "-"
        self.current_conf  = 0.0
        self.current_mode  = "Static"

        self._build_ui()
        self._apply_theme()

        threading.Thread(target=self._load_models, daemon=True).start()

        self.root.bind("<space>",  lambda e: self._toggle_camera())
        self.root.bind("<r>",      lambda e: self._toggle_record())
        self.root.bind("<c>",      lambda e: self._capture())
        self.root.bind("<Delete>", lambda e: self._delete_last())
        self.root.bind("<Escape>", lambda e: self._clear_sentence())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # --------------------------------------------------------
    # UI BUILD
    # --------------------------------------------------------
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0, minsize=SIDEBAR_W)
        self.root.rowconfigure(1, weight=1)

        # Header
        self.header = tk.Frame(self.root, height=48)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.header.columnconfigure(1, weight=1)
        self.header.grid_propagate(False)

        self.lbl_logo = tk.Label(
            self.header, text="[ISL] ISL DETECTOR",
            font=("Segoe UI", 14, "bold"), padx=16
        )
        self.lbl_logo.grid(row=0, column=0, sticky="w", pady=8)

        self.lbl_status = tk.Label(
            self.header, text="* Models loading...", font=FT_SMALL
        )
        self.lbl_status.grid(row=0, column=1, sticky="w", padx=8)

        self.lbl_clock = tk.Label(self.header, font=FT_SMALL, padx=16)
        self.lbl_clock.grid(row=0, column=2, sticky="e")
        self._tick_clock()

        # Camera
        self.main = tk.Frame(self.root)
        self.main.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=8)
        self.main.rowconfigure(0, weight=1)
        self.main.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.main, bg="#000000", highlightthickness=2)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Sidebar
        self._build_sidebar()

    def _build_sidebar(self):
        self.sb_outer = tk.Frame(self.root, width=SIDEBAR_W)
        self.sb_outer.grid(row=1, column=1, sticky="nsew",
                           padx=(0, 10), pady=8)
        self.sb_outer.grid_propagate(False)
        self.sb_outer.rowconfigure(0, weight=1)
        self.sb_outer.columnconfigure(0, weight=1)

        self.sb_canvas = tk.Canvas(
            self.sb_outer, width=SIDEBAR_W - 16, highlightthickness=0
        )
        self.sb_canvas.grid(row=0, column=0, sticky="nsew")

        self.sb_scroll = tk.Scrollbar(
            self.sb_outer, orient="vertical", command=self.sb_canvas.yview
        )
        self.sb_scroll.grid(row=0, column=1, sticky="ns")
        self.sb_canvas.configure(yscrollcommand=self.sb_scroll.set)

        self.sidebar = tk.Frame(self.sb_canvas, width=SIDEBAR_W - 20)
        self.sidebar.columnconfigure(0, weight=1)
        self.sb_canvas.create_window(
            (0, 0), window=self.sidebar, anchor="nw", width=SIDEBAR_W - 20
        )
        self.sidebar.bind("<Configure>",
            lambda e: self.sb_canvas.configure(
                scrollregion=self.sb_canvas.bbox("all")))
        self.sb_canvas.bind_all("<MouseWheel>",
            lambda e: self.sb_canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))

        self._build_cards()

    def _build_cards(self):
        p = {"padx": 6, "pady": (0, 8), "sticky": "ew"}
        r = 0

        # 1. Prediction
        self.card_pred = tk.LabelFrame(
            self.sidebar, text=" Prediction ", font=FT_TITLE, padx=8, pady=8
        )
        self.card_pred.grid(row=r, column=0, **p); r += 1
        self.card_pred.columnconfigure(0, weight=1)

        self.lbl_mode = tk.Label(
            self.card_pred, text="MODE: STATIC", font=FT_SMALL, anchor="w"
        )
        self.lbl_mode.grid(row=0, column=0, sticky="ew")

        self.lbl_pred = tk.Label(
            self.card_pred, text="-", font=FT_PRED
        )
        self.lbl_pred.grid(row=1, column=0, pady=(4, 2))

        self.lbl_conf = tk.Label(
            self.card_pred, text="Confidence: -", font=FT_BODY, anchor="w"
        )
        self.lbl_conf.grid(row=2, column=0, sticky="ew")

        self.conf_canvas = tk.Canvas(
            self.card_pred, height=10, bd=0, highlightthickness=0
        )
        self.conf_canvas.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.conf_rect = self.conf_canvas.create_rectangle(
            0, 0, 0, 10, fill="#3FB950", outline=""
        )

        # 2. Sentence
        self.card_sent = tk.LabelFrame(
            self.sidebar, text=" Translated Sentence ",
            font=FT_TITLE, padx=8, pady=8
        )
        self.card_sent.grid(row=r, column=0, **p); r += 1
        self.card_sent.columnconfigure(0, weight=1)
        self.card_sent.columnconfigure(1, weight=1)

        self.txt_sentence = tk.Text(
            self.card_sent, height=4, font=("Segoe UI", 13),
            wrap="word", bd=0, state="disabled", relief="flat",
            padx=4, pady=4
        )
        self.txt_sentence.grid(row=0, column=0, columnspan=2,
                               sticky="ew", pady=(0, 6))

        self.btn_del = self._btn(
            self.card_sent, "Delete Last", self._delete_last, row=1, col=0
        )
        self.btn_clr = self._btn(
            self.card_sent, "Clear All", self._clear_sentence, row=1, col=1
        )

        # 3. History
        self.card_hist = tk.LabelFrame(
            self.sidebar, text=" Detection History ",
            font=FT_TITLE, padx=8, pady=8
        )
        self.card_hist.grid(row=r, column=0, **p); r += 1
        self.card_hist.columnconfigure(0, weight=1)

        hi = tk.Frame(self.card_hist)
        hi.grid(row=0, column=0, sticky="ew")
        hi.columnconfigure(0, weight=1)

        self.txt_history = tk.Text(
            hi, height=7, font=FT_MONO,
            wrap="none", bd=0, state="disabled",
            relief="flat", padx=4, pady=4
        )
        self.txt_history.grid(row=0, column=0, sticky="ew")

        hvsb = tk.Scrollbar(hi, orient="vertical",
                            command=self.txt_history.yview)
        hvsb.grid(row=0, column=1, sticky="ns")
        self.txt_history.configure(yscrollcommand=hvsb.set)

        self.btn_hclr = self._btn(
            self.card_hist, "Clear History",
            self._clear_history, row=1, col=0
        )

        # 4. Controls
        self.card_ctrl = tk.LabelFrame(
            self.sidebar, text=" Controls ",
            font=FT_TITLE, padx=8, pady=8
        )
        self.card_ctrl.grid(row=r, column=0, **p); r += 1
        self.card_ctrl.columnconfigure(0, weight=1)
        self.card_ctrl.columnconfigure(1, weight=1)

        self.btn_cam = self._btn(
            self.card_ctrl, "> Start Camera",
            self._toggle_camera, row=0, col=0, cs=2
        )
        self.btn_cap = self._btn(
            self.card_ctrl, "Capture",
            self._capture, row=1, col=0
        )
        self.btn_rec = self._btn(
            self.card_ctrl, "Record",
            self._toggle_record, row=1, col=1
        )
        self.btn_zin = self._btn(
            self.card_ctrl, "Zoom +",
            self._zoom_in, row=2, col=0
        )
        self.btn_zout = self._btn(
            self.card_ctrl, "Zoom -",
            self._zoom_out, row=2, col=1
        )
        self.lbl_zoom = tk.Label(
            self.card_ctrl, text="Zoom: 1.00x", font=FT_SMALL
        )
        self.lbl_zoom.grid(row=3, column=0, columnspan=2, pady=(4, 0))

        # 5. Settings
        self.card_set = tk.LabelFrame(
            self.sidebar, text=" Settings ",
            font=FT_TITLE, padx=8, pady=8
        )
        self.card_set.grid(row=r, column=0, **p); r += 1
        self.card_set.columnconfigure(0, weight=1)

        self.btn_theme = self._btn(
            self.card_set, "Light Theme",
            self._toggle_theme, row=0, col=0
        )

        tk.Label(
            self.card_set, text="Min Confidence Threshold",
            font=FT_SMALL, anchor="w"
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        tr = tk.Frame(self.card_set)
        tr.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        tr.columnconfigure(0, weight=1)

        self.var_thresh = tk.DoubleVar(value=0.70)
        self.slider = ttk.Scale(
            tr, from_=0.3, to=0.99,
            variable=self.var_thresh, orient="horizontal",
            style="ISL.Horizontal.TScale"
        )
        self.slider.grid(row=0, column=0, sticky="ew")
        self.lbl_thresh = tk.Label(
            tr, text="0.70", font=FT_SMALL, width=4, anchor="w"
        )
        self.lbl_thresh.grid(row=0, column=1, padx=(6, 0))
        self.var_thresh.trace_add("write", self._on_thresh)

        self.var_lm = tk.BooleanVar(value=True)
        self.chk_lm = tk.Checkbutton(
            self.card_set, text="Show Hand Landmarks",
            variable=self.var_lm, font=FT_SMALL
        )
        self.chk_lm.grid(row=3, column=0, sticky="w", pady=(6, 0))

        self.var_hist = tk.BooleanVar(value=True)
        self.chk_hist = tk.Checkbutton(
            self.card_set, text="Auto-log to History",
            variable=self.var_hist, font=FT_SMALL
        )
        self.chk_hist.grid(row=4, column=0, sticky="w", pady=(2, 0))

        # 6. Shortcuts
        self.card_keys = tk.LabelFrame(
            self.sidebar, text=" Keyboard Shortcuts ",
            font=FT_TITLE, padx=8, pady=6
        )
        self.card_keys.grid(row=r, column=0, **p); r += 1

        shortcuts = [
            ("Space",  "Start / Stop Camera"),
            ("R",      "Start / Stop Recording"),
            ("C",      "Capture Screenshot"),
            ("Delete", "Remove Last Word"),
            ("Esc",    "Clear Sentence"),
        ]
        for i, (k, v) in enumerate(shortcuts):
            tk.Label(self.card_keys, text=f"{k:9}",
                     font=FT_MONO, anchor="w").grid(
                row=i, column=0, sticky="w")
            tk.Label(self.card_keys, text=v,
                     font=FT_SMALL, anchor="w").grid(
                row=i, column=1, sticky="w", padx=(4, 0))

    def _btn(self, parent, text, cmd, row, col, cs=1):
        b = tk.Button(
            parent, text=text, command=cmd,
            font=FT_BODY, relief="flat", bd=0,
            padx=6, pady=5, cursor="hand2"
        )
        b.grid(row=row, column=col, columnspan=cs,
               sticky="ew", padx=2, pady=2)
        return b

    # --------------------------------------------------------
    # THEME
    # --------------------------------------------------------
    def _apply_theme(self):
        T = self.T
        self.root.configure(bg=T["bg"])
        self.header.configure(bg=T["panel"])
        self.lbl_logo.configure(bg=T["panel"], fg=T["accent"])
        self.lbl_status.configure(bg=T["panel"], fg=T["text_dim"])
        self.lbl_clock.configure(bg=T["panel"], fg=T["text_dim"])
        self.main.configure(bg=T["bg"])
        self.canvas.configure(bg="#000000",
                              highlightbackground=T["border"])
        self.sb_outer.configure(bg=T["bg"])
        self.sb_canvas.configure(bg=T["bg"])
        self.sb_scroll.configure(bg=T["scroll"], troughcolor=T["bg"])
        self.sidebar.configure(bg=T["bg"])

        for card in [self.card_pred, self.card_sent, self.card_hist,
                     self.card_ctrl, self.card_set, self.card_keys]:
            card.configure(bg=T["card"], fg=T["accent"],
                           relief="groove", bd=1)
            self._theme_children(card, T)

        self.lbl_pred.configure(bg=T["card"], fg=T["accent"])
        self.lbl_conf.configure(bg=T["card"], fg=T["text_dim"])
        self.lbl_mode.configure(bg=T["card"], fg=T["text_dim"])
        self.conf_canvas.configure(bg=T["border"])
        self.txt_sentence.configure(
            bg=T["panel"], fg=T["text"],
            insertbackground=T["text"], selectbackground=T["accent"]
        )
        self.txt_history.configure(
            bg=T["hist_bg"], fg=T["hist_fg"],
            insertbackground=T["text"], selectbackground=T["accent"]
        )
        self.lbl_zoom.configure(bg=T["card"], fg=T["text_dim"])
        self.lbl_thresh.configure(bg=T["card"], fg=T["text_dim"])

        for chk in [self.chk_lm, self.chk_hist]:
            chk.configure(
                bg=T["card"], fg=T["text"],
                selectcolor=T["card"],
                activebackground=T["card"],
                activeforeground=T["text"]
            )

        all_btns = [self.btn_del, self.btn_clr, self.btn_cam,
                    self.btn_cap, self.btn_rec, self.btn_zin,
                    self.btn_zout, self.btn_theme, self.btn_hclr]
        for b in all_btns:
            b.configure(
                bg=T["btn_bg"], fg=T["text"],
                activebackground=T["btn_active"],
                activeforeground="#FFFFFF"
            )

        self.btn_theme.configure(
            text="Light Theme" if self._theme_name == "dark"
            else "Dark Theme"
        )
        style = ttk.Style()
        style.configure("ISL.Horizontal.TScale",
                        troughcolor=T["border"],
                        background=T["accent"],
                        sliderlength=18)

    def _theme_children(self, widget, T):
        for child in widget.winfo_children():
            cls = child.__class__.__name__
            try:
                if cls == "Label":
                    child.configure(bg=T["card"], fg=T["text"])
                elif cls == "Frame":
                    child.configure(bg=T["card"])
                    self._theme_children(child, T)
                elif cls == "Button":
                    child.configure(
                        bg=T["btn_bg"], fg=T["text"],
                        activebackground=T["btn_active"],
                        activeforeground="#FFFFFF"
                    )
            except Exception:
                pass

    def _toggle_theme(self):
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        self.T = THEMES[self._theme_name]
        self._apply_theme()

    # --------------------------------------------------------
    # CLOCK
    # --------------------------------------------------------
    def _tick_clock(self):
        now = datetime.datetime.now().strftime("%H:%M:%S  %d/%m/%Y")
        self.lbl_clock.configure(text=now)
        self.root.after(1000, self._tick_clock)

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------
    def _load_models(self):
        errors = []

        # Static
        try:
            self.static_model = joblib.load(resource_path("static_sign_model.pkl"))
            print("Static model loaded OK")
        except Exception as e:
            errors.append(f"Static: {e}")

        # Dynamic
        try:
            from keras.models import load_model
            self.dynamic_model = load_model(resource_path("dynamic_sign_model.h5"))
            print("Dynamic model loaded OK")
        except Exception as e:
            errors.append(f"Dynamic: {e}")

        self._models_loaded = True

        if errors:
            msg = " | ".join(errors)
            self.root.after(0, lambda m=msg: self.lbl_status.configure(
                text=f"* Warning: {m}", fg=self.T["warn"]
            ))
        else:
            self.root.after(0, lambda: self.lbl_status.configure(
                text="* Models ready - Static + Dynamic",
                fg=self.T["accent2"]
            ))

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------
    def _toggle_camera(self):
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if not self._models_loaded:
            messagebox.showinfo("Wait", "Models still loading.")
            return
        if self.static_model is None:
            messagebox.showerror("Error",
                "static_sign_model.pkl not found in SLD folder.")
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.btn_cam.configure(text="[STOP] Stop Camera")
        self._process_frame()

    def _stop_camera(self):
        self.running = False
        if self.recording:
            self._stop_recording()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.configure(text="> Start Camera")
        self.canvas.delete("all")

    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------
    def _process_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._process_frame)
            return

        frame = cv2.flip(frame, 1)
        if self.zoom_level != 1.0:
            frame = self._zoom(frame)

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_res = self.hands_model.process(rgb)
        f_res = self.face_model.process(rgb)

        hand_present = bool(h_res.multi_hand_landmarks)

        if self.var_lm.get() and hand_present:
            for hlm in h_res.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hlm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(
                        color=(0, 200, 255), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(
                        color=(255, 255, 255), thickness=1)
                )

        label, conf = "-", 0.0
        thresh = self.var_thresh.get()

        if hand_present:
            feat = extract_static_features(h_res)

            # Movement detection
            if self.prev_features is not None:
                movement = np.linalg.norm(feat - self.prev_features)
            else:
                movement = 1.0
            self.prev_features = feat

            # Static prediction
            s_pred = self.static_model.predict([feat])[0]
            s_label = STATIC_LABELS[s_pred]
            s_proba = self.static_model.predict_proba([feat])[0]
            s_conf  = float(np.max(s_proba))

            self.static_smoother.update(s_label)
            label = self.static_smoother.get() or s_label
            conf  = s_conf
            self.current_mode = "Static"

            # Dynamic prediction
            if self.dynamic_model is not None:
                dyn_feat = extract_dynamic_features(h_res, f_res)
                if dyn_feat.shape[0] == 129:
                    self.dyn_buffer.append(dyn_feat)

                if len(self.dyn_buffer) == SEQUENCE_LEN:
                    seq  = np.expand_dims(np.array(self.dyn_buffer), axis=0)
                    pred = self.dynamic_model.predict(seq, verbose=0)[0]
                    d_conf  = float(np.max(pred))
                    d_label = DYNAMIC_LABELS[int(np.argmax(pred))]

                    self.dynamic_smoother.update(d_label)
                    dm_label = self.dynamic_smoother.get()

                    # Dynamic overrides static if significantly more confident
                    # OR if there is clear movement
                    if (d_conf >= DYNAMIC_CONF_THRESH and
                            (d_conf > s_conf + 0.10 or movement > MOVEMENT_THRESH)):
                        label = dm_label
                        conf  = d_conf
                        self.current_mode = "Dynamic"
        else:
            self.dyn_buffer.clear()
            self.prev_features = None
            self.static_smoother.reset()
            self.dynamic_smoother.reset()

        # Sentence builder
        if label != "-" and conf >= thresh:
            prev = self.sentence_builder.get()
            self.sentence_builder.feed(label, conf)
            new = self.sentence_builder.get()
            if new != prev:
                self._refresh_sentence()
                if self.var_hist.get():
                    self._add_history(label, conf, self.current_mode)

        self.current_label = label
        self.current_conf  = conf
        self._update_pred(label, conf)
        frame = self._draw_hud(frame, label, conf)

        if self.recording and self.video_writer:
            self.video_writer.write(frame)
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (28, 28), 10, (0, 0, 220), -1)
            cv2.putText(frame,
                        f"REC {time.time()-self.rec_start:.0f}s",
                        (46, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 220), 2)

        self._show_frame(frame)
        self.root.after(15, self._process_frame)

    def _zoom(self, frame):
        h, w = frame.shape[:2]
        z = self.zoom_level
        cx, cy = w // 2, h // 2
        nw, nh = int(w / z), int(h / z)
        x1 = max(cx - nw // 2, 0)
        y1 = max(cy - nh // 2, 0)
        return cv2.resize(frame[y1:y1+nh, x1:x1+nw], (w, h),
                          interpolation=cv2.INTER_LINEAR)

    def _draw_hud(self, frame, label, conf):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (340, 76), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
        mc = (0, 255, 150) if self.current_mode == "Static" else (255, 200, 0)
        cv2.putText(frame, f"{self.current_mode.upper()} MODE",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mc, 1)
        cv2.putText(frame, str(label),
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    (255, 255, 255), 3)
        bw = 220; bx = w - bw - 20; by = 18
        fil = int(bw * conf)
        bc = ((0,220,80) if conf>=0.75 else
              (40,180,255) if conf>=0.5 else (80,80,255))
        cv2.rectangle(frame, (bx,by), (bx+bw, by+14), (50,50,50), -1)
        cv2.rectangle(frame, (bx,by), (bx+fil, by+14), bc, -1)
        cv2.putText(frame, f"{conf*100:.1f}%",
                    (bx, by+30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (210,210,210), 1)
        return frame

    def _show_frame(self, frame):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # --------------------------------------------------------
    # PREDICTION UI
    # --------------------------------------------------------
    def _update_pred(self, label, conf):
        T = self.T
        self.lbl_pred.configure(text=str(label))
        self.lbl_conf.configure(text=f"Confidence: {conf*100:.1f}%")
        self.lbl_mode.configure(text=f"MODE: {self.current_mode.upper()}")
        bw = self.conf_canvas.winfo_width()
        filled = int(bw * conf)
        color = (T["accent2"] if conf >= 0.75 else
                 T["accent"]  if conf >= 0.50 else T["accent3"])
        self.conf_canvas.coords(self.conf_rect, 0, 0, filled, 10)
        self.conf_canvas.itemconfigure(self.conf_rect, fill=color)

    # --------------------------------------------------------
    # SENTENCE
    # --------------------------------------------------------
    def _refresh_sentence(self):
        text = self.sentence_builder.get()
        self.txt_sentence.configure(state="normal")
        self.txt_sentence.delete("1.0", "end")
        self.txt_sentence.insert("end", text)
        self.txt_sentence.configure(state="disabled")

    def _delete_last(self):
        self.sentence_builder.delete_last()
        self._refresh_sentence()

    def _clear_sentence(self):
        self.sentence_builder.clear()
        self._refresh_sentence()

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------
    def _add_history(self, label, conf, mode):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}]  {label:<12}  {conf*100:5.1f}%  ({mode})\n"
        self.history_log.append(entry)
        self.txt_history.configure(state="normal")
        self.txt_history.insert("end", entry)
        self.txt_history.see("end")
        self.txt_history.configure(state="disabled")

    def _clear_history(self):
        self.history_log = []
        self.txt_history.configure(state="normal")
        self.txt_history.delete("1.0", "end")
        self.txt_history.configure(state="disabled")

    # --------------------------------------------------------
    # CAPTURE / RECORD
    # --------------------------------------------------------
    def _capture(self):
        if not self.running or not self.cap:
            messagebox.showinfo("Camera", "Start the camera first.")
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        os.makedirs("captures", exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("captures", f"capture_{ts}.png")
        cv2.imwrite(path, frame)
        messagebox.showinfo("Saved", f"Screenshot saved:\n{path}")

    def _toggle_record(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self.running:
            messagebox.showinfo("Camera", "Start the camera first.")
            return
        os.makedirs("recordings", exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("recordings", f"recording_{ts}.mp4")
        self.video_writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720)
        )
        self.recording = True
        self.rec_start = time.time()
        self.btn_rec.configure(text="Stop Rec")
        self.lbl_status.configure(text="* Recording...",
                                  fg=self.T["danger"])

    def _stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.btn_rec.configure(text="Record")
        self.lbl_status.configure(
            text="* Models ready", fg=self.T["accent2"])

    # --------------------------------------------------------
    # ZOOM
    # --------------------------------------------------------
    def _zoom_in(self):
        self.zoom_level = min(self.zoom_level + 0.25, 3.0)
        self.lbl_zoom.configure(text=f"Zoom: {self.zoom_level:.2f}x")

    def _zoom_out(self):
        self.zoom_level = max(self.zoom_level - 0.25, 1.0)
        self.lbl_zoom.configure(text=f"Zoom: {self.zoom_level:.2f}x")

    # --------------------------------------------------------
    # MISC
    # --------------------------------------------------------
    def _on_thresh(self, *args):
        v = self.var_thresh.get()
        self.lbl_thresh.configure(text=f"{v:.2f}")
        self.sentence_builder.conf_thresh = v

    def _on_close(self):
        self.running = False
        if self.recording:
            self._stop_recording()
        if self.cap:
            self.cap.release()
        self.hands_model.close()
        self.face_model.close()
        self.root.destroy()


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("ISL.Horizontal.TScale",
                    troughcolor="#30363D",
                    background="#58A6FF",
                    sliderlength=18)
    app = ISLApp(root)
    root.mainloop()
