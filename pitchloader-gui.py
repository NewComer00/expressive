import os
import sys
import time
import ctypes
import gettext
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, field
from traceback import TracebackException

import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox, ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import pitchloader
from utils.persist import persist_to_file


# Initialize gettext for localization
parser = argparse.ArgumentParser(description='Choose application language.')
parser.add_argument('--lang', default='zh_CN', help='Set language for localization (e.g. zh_CN, en)')
args = parser.parse_args()

lang = args.lang
locale_dir = os.path.join(os.path.dirname(__file__), 'locales')

try:
    lang_translations = gettext.translation('app', localedir=locale_dir, languages=[lang], fallback=True)
    _ = lang_translations.gettext
except FileNotFoundError:
    print(f"Translation file for language '{lang}' not found. Falling back to default (en).")


# Description text for the software
DESCRIPTION = f"{pitchloader.APP_NAME} v{pitchloader.APP_VERSION}\n" + \
_("""
[Software Features]
This software automatically matches and generates Pitch Deviation expression parameters in the corresponding track of an OpenUtau project file, based on the reference vocal audio provided by the user. The software does not modify your original OpenUtau project file.

[Usage Steps]
1. Select a song to be covered by a virtual singer and use tools like UVR to get the vocal track.
2. Create a singing track in OpenUtau with notes but without parameters. The tempo and note lengths do not need to be an exact match to the original song—just similar.
3. Input the following files into this software:
   - "Reference Vocal Audio File" (WAV format)
   - "Exported Virtual Singer Vocal File from OpenUtau" (WAV format)
   - "OpenUtau Project File" (USTX format)
4. Enter the track number for which you want to generate Pitch Deviation expression parameters.
5. Click the "Run" button and wait for the progress bar to complete. The current step will be displayed in the status bar below the progress bar.
6. The pitch extraction process takes a few minutes. Using an Nvidia GPU (driver version >= 452.39) can significantly speed up the process. The software has a caching mechanism, so if the same audio file is processed again, it will use the cached data.
7. Once completed, maximize the software window to view the pitch curve on the right side. The OpenUtau project file with Pitch Deviation expression parameters will be saved next to your original project file.
""")


# Define the Settings dataclass with default values
@persist_to_file("settings.json")
@dataclass
class Settings:
    ref_wav_path: str = field(default="")  # Path to reference WAV file
    utau_wav_path: str = field(default="")  # Path to UTAU WAV file
    ustx_path: str = field(default="")  # Path to USTX project file
    confidence_threshold_r: float = field(default=0.6)  # Confidence threshold for ref pitch detection
    confidence_threshold_u: float = field(default=0.8)  # Confidence threshold for ustx pitch detection
    utau_track: int = field(default=1)  # Vocal track number
    time_align_radius: int = field(default=1)  # Time alignment radius
    semitone_shift: int = field(default=99)  # Semitone Shift
    pitch_align_smooth: int = field(default=2)  # Pitch alignment smoothing value
    scaler: float = field(default=2.0)  # Scaler for pitch delta


# ToolTip class for displaying tooltips on hover
class ToolTip:
    def __init__(self, widget, text="Info"):
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, bg="lightyellow", relief="solid", borderwidth=1)
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


# Utility functions
def append_to_filename(filename: str, append_str: str) -> str:
    path = Path(filename)
    return str(path.parent / f"{path.stem}{append_str}{path.suffix}")


def add_cuda_path():
    """Add CUDA path to library searching path."""
    import nvidia.cuda_nvcc
    import nvidia.cuda_runtime
    import nvidia.cudnn
    import nvidia.cublas
    import nvidia.cusolver
    import nvidia.cusparse
    import nvidia.cufft
    import nvidia.curand
    for package in [nvidia.cuda_nvcc, nvidia.cuda_runtime, nvidia.cudnn, nvidia.cublas,
                    nvidia.cusolver, nvidia.cusparse, nvidia.cufft, nvidia.curand]:
        lib_path = Path(package.__path__[0]) / "bin"
        if os.name == "nt":
            os.environ["PATH"] = \
                str(lib_path) + ';' + os.environ.get('PATH', '')
        else:
            os.environ["LD_LIBRARY_PATH"] = \
                str(lib_path) + ':' + os.environ.get('LD_LIBRARY_PATH', '')


# Main GUI class
class PitchLoaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(_("Pitch Loader GUI"))

        # Load settings
        self.settings = Settings()
        self.settings.load()

        # Create UI
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.08, hspace=0.2)
        self.entries = dict()
        self.canvas = None
        self.create_widgets()

        self.has_gpu = None

    def create_widgets(self):
        """Create UI elements dynamically."""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plot_frame = tk.Frame(main_frame, borderwidth=10,
                              highlightbackground="grey", highlightthickness=1)
        plot_frame.pack(fill=tk.BOTH, side=tk.RIGHT, anchor=tk.E, expand=True)
        plot_frame.pack_propagate(0)

        panel_frame = tk.Frame(main_frame, borderwidth=10,
                               highlightbackground="grey", highlightthickness=1)
        panel_frame.pack(fill=tk.BOTH, side=tk.LEFT, anchor=tk.W, expand=False)

        # Canvas
        self.canvas = canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X, expand=True)
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Buttons
        button_frame = tk.Frame(panel_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text=_("Restore Defaults"),
                   command=self.restore_defaults).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text=_("Run Pitch Loader"),
                   command=self.run_pitchloader).pack(side=tk.LEFT, padx=0, fill=tk.BOTH, expand=True)

        # File Path Inputs
        file_inputs = [
            (_("Ref. WAV Path:"), "ref_wav_path", [(_("WAV Files"), "*.wav")], _("Reference WAV file, typically a vocal-only track generated by UVR.")),
            (_("USTX WAV Path:"), "utau_wav_path", [(_("WAV Files"), "*.wav")], _("WAV file exported from OpenUtau, corresponding to the reference WAV.")),
            (_("USTX File Path:"), "ustx_path", [(_("USTX Files"), "*.ustx")], _("OpenUtau project file. The processed USTX will be created alongside it.")),
        ]

        for i, (label_text, key, file_type, info) in enumerate(file_inputs):
            frame = tk.Frame(panel_frame)
            frame.pack(fill=tk.X, side=tk.TOP, anchor=tk.NW)
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            self.entries[key] = entry = ttk.Entry(frame, width=30)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
            entry.insert(0, getattr(self.settings, key))
            entry.bind("<KeyRelease>", lambda e,
                       k=key: self.update_setting(k, e.widget.get(), str))
            ttk.Button(frame, text=_("Browse"), command=lambda k=key,
                       t=file_type: self.browse_file(k, t)).pack(side=tk.LEFT)
            ToolTip(entry, info)

        # Numeric Fields with Help Info
        num_inputs = {
            _("USTX Info."): [
                (_("Vocal Track:"), "utau_track", int, _("Index of the vocal track to process (starting from 1).")),
            ],
            _("Pitch Extraction"): [
                (_("Confidence Threshold (Ref.):"), "confidence_threshold_r", float, _("Threshold for filtering uncertain pitch values in Ref. WAV.")),
                (_("Confidence Threshold (USTX):"), "confidence_threshold_u", float, _("Threshold for filtering uncertain pitch values in USTX WAV.")),
            ],
            _("Time Alignment"): [
                    (_("Time Align Radius:"), "time_align_radius", int, _("Radius parameter for the FastDTW algorithm.\nA larger radius allows for more flexible alignment but increases computation time.\nRecommended value: 1.")),
            ],
            _("Pitch Alignment"): [
                (_("Semitone Shift:"), "semitone_shift", int, _("Semitone shift between the USTX and Ref. WAV.\nExample: If the USTX WAV is an octave higher than the Ref. WAV, set to 12; otherwise, -12.\nSet to 99 to enable automatic shift estimation.")),
                (_("Pitch Align Smooth:"), "pitch_align_smooth", int, _("Controls the smoothing strength for pitch alignment.")),
            ],
            _("Pitch Deviation Generation"): [
                (_("Scaler:"), "scaler", float, _("Scaling factor for pitch deviation.")),
            ],
        }

        for group_name, group in num_inputs.items():
            group_frame = tk.Frame(panel_frame, borderwidth=3,
                                   highlightbackground="grey", highlightthickness=1)
            group_title = ttk.Label(group_frame, text=group_name)
            group_title.pack(side=tk.TOP, anchor=tk.NW)
            group_frame.pack(fill=tk.X)
            for label_text, key, _type, info in group:
                frame = tk.Frame(group_frame)
                frame.pack(padx=5, side=tk.TOP, anchor=tk.NW)
                ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
                self.entries[key] = entry = ttk.Entry(frame, width=5)
                entry.pack(side=tk.LEFT, expand=True, padx=5)
                entry.insert(0, getattr(self.settings, key))
                # Force early binding https://stackoverflow.com/a/76423445
                entry.bind("<KeyRelease>", lambda e,
                           k=key, t=_type: self.update_setting(k, e.widget.get(), t))
                ToolTip(entry, info)

        # Progress Bar & Status Line
        status_frame = tk.Frame(panel_frame)

        self.progress_bar = ttk.Progressbar(
            status_frame, orient="horizontal", length=400, mode="determinate", maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(
            status_frame, text=_("Status: Waiting for input..."),
            foreground="blue", anchor="w")
        self.status_label.pack(fill=tk.X)

        # https://stackoverflow.com/a/71599924/15283141
        status_frame.bind('<Configure>',
                          lambda e: self.status_label.config(
                              wraplength=self.status_label.winfo_width()))
        status_frame.pack(fill=tk.X, pady=10)

        # ScrolledText
        text_area = scrolledtext.ScrolledText(panel_frame, wrap=tk.CHAR, width=30, height=10)
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_area.insert(tk.INSERT, DESCRIPTION)
        text_area.config(state=tk.DISABLED)

    def update_setting(self, key, value, _type):
        """Update a setting and save it."""
        if len(value) == 0:
            self.update_status(_('❌The value of "{}" should not be empty.').format(key))
            return
        try:
            setattr(self.settings, key, _type(value))
            self.settings.save()
        except ValueError:
            self.update_status(_('❌The type of "{}" should be "{}".').format(key, _type.__name__))
        else:
            self.update_status(_('✔️The value of "{}" is set to "{}".').format(key, _type(value)))

    def browse_file(self, key, file_types):
        """Open file dialog and update entry field."""
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            setattr(self.settings, key, file_path)
            self.settings.save()
            self.update_ui()

    def restore_defaults(self):
        """Reset settings to default values (excluding file paths)."""
        self.settings.restore()
        self.update_ui()

    def update_ui(self):
        """Update the UI with current settings."""
        for key in self.settings.__dict__:
            if key in self.entries:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, getattr(self.settings, key))

    def run_pitchloader(self):
        """Run the pitchloader process in a separate thread."""
        def task():
            try:
                self.update_status(_("Starting process..."))
                self.ax1.clear()
                self.ax2.clear()
                self.ax1.set_title(_("Input Sequences"))
                self.ax2.set_title(_("Processed Sequences"))
                self.ax2.set_xlabel(_("Time (s)"))
                self.ax2.set_ylabel(_("Frequency (Hz)"))
                self.canvas.draw()

                # Step 0: Test GPU
                if self.has_gpu is None:
                    self.update_status(_("Searching for available Nvidia GPU..."))
                    self.has_gpu = self.test_gpu()
                    time.sleep(2)

                # Step 1: Load USTX file
                self.update_status(_("Loading USTX file..."))
                ustx_dict = pitchloader.load_ustx(self.settings.ustx_path)
                ustx_tempo = ustx_dict['tempos'][0]['bpm']
                self.progress_bar.step(10)

                # Step 2: Extract USTX WAV features
                self.update_status(_("Extracting pitch from USTX WAV..."))
                utau_tick, utau_pitch, utau_features = pitchloader.get_wav_features(
                    self.settings.utau_wav_path,
                    tempo=ustx_tempo,
                    confidence_threshold=self.settings.confidence_threshold_u
                )
                self.progress_bar.step(10)

                # Step 3: Extract Reference WAV features
                self.update_status(_("Extracting pitch from Reference WAV..."))
                ref_tick, ref_pitch, ref_features = pitchloader.get_wav_features(
                    self.settings.ref_wav_path,
                    tempo=ustx_tempo,
                    confidence_threshold=self.settings.confidence_threshold_r
                )
                self.progress_bar.step(20)

                seqs_to_plot = [(_("Ref. WAV Pitch"), ref_tick, ref_pitch, 'b'),
                                (_("USTX WAV Pitch"), utau_tick, utau_pitch, 'r')]
                for (l, t, s, c) in seqs_to_plot:
                    self.ax1.scatter(pitchloader.ticks_to_time(t, ustx_tempo),
                                     s, label=l, color=c, lw=0, alpha=0.5, s=3)
                self.ax1.legend(fontsize='xx-small')
                self.canvas.draw()

                # Step 4: Align sequence time
                # NOTICE: UTAU WAV is the reference, and Ref. WAV is the query
                self.update_status(_("Aligning sequence time..."))
                unified_tick, (time_aligned_ref_pitch, *_unused), (unified_utau_pitch, *_unused) = \
                    pitchloader.align_sequence_time(
                        query_time=ref_tick,
                        queries=(ref_pitch, *ref_features),
                        reference_time=utau_tick,
                        references=(utau_pitch, *utau_features),
                        align_radius=self.settings.time_align_radius
                    )
                self.progress_bar.step(20)

                # Step 5: Align sequence pitch
                self.update_status(_("Aligning sequence pitch..."))
                semitone_shift = None
                if self.settings.semitone_shift != 99:
                    semitone_shift = self.settings.semitone_shift
                time_pitch_aligned_ref_pitch, estimated_shift = pitchloader.align_sequence_pitch(
                    time_aligned_ref_pitch,
                    unified_utau_pitch,
                    semitone_shift,
                    self.settings.pitch_align_smooth
                )
                self.progress_bar.step(10)

                # Step 6: Get pitch delta
                self.update_status(_("Calculating pitch delta..."))
                delta_pitch = pitchloader.get_pitch_delta(
                    time_pitch_aligned_ref_pitch,
                    unified_utau_pitch,
                    self.settings.scaler)
                self.progress_bar.step(10)

                seqs_to_plot = [(_("Aligned Ref. WAV Pitch"), unified_tick, time_pitch_aligned_ref_pitch, 'b'),
                                (_("Aligned USTX WAV Pitch"), unified_tick, unified_utau_pitch, 'r'),
                                (_("Pitch Deviation"), unified_tick, delta_pitch, 'y')]
                for (l, t, s, c) in seqs_to_plot:
                    self.ax2.scatter(pitchloader.ticks_to_time(t, ustx_tempo),
                                     s, label=l, color=c, lw=0, alpha=0.5, s=3)
                self.ax2.legend(fontsize='xx-small')
                self.canvas.draw()

                # Step 7: Edit USTX pitch
                self.update_status(_("Editing USTX pitch..."))
                pitchloader.edit_ustx_pitch(
                    ustx_dict,
                    self.settings.utau_track,
                    unified_tick,
                    delta_pitch)
                self.progress_bar.step(10)

                # Step 8: Save USTX
                self.update_status(_("Saving output..."))
                ustx_output_path = append_to_filename(
                    self.settings.ustx_path,
                    _(" - PitchLoader Output"))
                pitchloader.save_ustx(ustx_dict, ustx_output_path)
                self.progress_bar.step(10)

                succeed_msg = _("Pitch loaded successfully!")
                if semitone_shift is None:
                    succeed_msg += _('\nThe estimated semitone shift is "{}".').format(int(estimated_shift))
                self.update_status(succeed_msg)

            except Exception as e:
                traceback = ''.join(TracebackException.from_exception(e).format())
                messagebox.showerror(_("Error"), _("An error occurred!\n{}").format(traceback))
            finally:
                self.progress_bar.stop()

        # Run process in background
        threading.Thread(target=task, daemon=True).start()

    def update_status(self, message):
        """Update the status label."""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def test_gpu(self):
        """Test if the GPU is available."""
        # Import upon use to avoid import errors
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.update_status(message=_("Nvidia GPU detected!"))
            return True
        else:
            self.update_status(message=_("No usable GPU detected. Pitch extraction may be slow."))
            return False


# Run the application
if __name__ == "__main__":
    add_cuda_path()

    # DPI Setting
    # https://stackoverflow.com/a/70720420/15283141
    ctypes.windll.shcore.SetProcessDpiAwareness(1)

    root = tk.Tk()

    # Handle window closing properly
    # https://stackoverflow.com/a/55206851/15283141
    def on_closing():
        root.quit()
        root.destroy()
        sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    app = PitchLoaderGUI(root)
    root.mainloop()
