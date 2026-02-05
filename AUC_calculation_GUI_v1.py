# TKinter-based GUI for the Raman Spectroscopy AUC Analysis Tool.
# Imports analysis functions from  AUC_calculation_v1.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, sys, io
from pathlib import Path
import pandas as pd

import AUC_calculation_v1 as raman          

# ─────────────────────────────────────────────────────────
#  PALETTE
C_BG       = "#2b2b2b"   # root background
C_FRAME    = "#333333"   # label-frame fill
C_ENTRY    = "#3c3c3c"   # entry / combobox background
C_FG       = "#e0e0e0"   # primary text
C_DIM      = "#666666"   # disabled text
C_ACCENT   = "#4fc3f7"   # headings / summary
C_ACC_DK   = "#0288d1"   # accent-button normal
C_LOG_BG   = "#1e1e1e"   # log panel background
C_ROW_ALT  = "#383838"   # treeview alternating row


# ─────────────────────────────────────────────────────────
class RamanGUI:
    # ─── INIT ────────────────────────────────────────────
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectroscopy — AUC Analysis")
        self.root.geometry("940x820")
        self.root.minsize(800, 660)
        self.root.configure(bg=C_BG)

        self.results_df    = None
        self.selected_path = tk.StringVar()

        self._styles()
        self._build()

    # ─── STYLES ──────────────────────────────────────────
    def _styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("TFrame",                  background=C_BG)
        s.configure("TLabelframe",             background=C_FRAME)
        s.configure("TLabelframe.Label",       background=C_FRAME, foreground=C_FG,
                    font=("Helvetica", 10, "bold"))
        s.configure("TLabel",                  background=C_FRAME, foreground=C_FG,
                    font=("Helvetica", 9))
        s.configure("TCheckbutton",            background=C_FRAME, foreground=C_FG)
        s.configure("TRadiobutton",            background=C_FRAME, foreground=C_FG)
        s.configure("TCombobox",               fieldbackground=C_ENTRY, foreground=C_FG)
        s.map(      "TCombobox",               fieldbackground=[("readonly", C_ENTRY)])

        # ── accent button (Run)
        s.configure("Accent.TButton",          background=C_ACC_DK, foreground="white",
                    font=("Helvetica", 10, "bold"), padding=(24, 8))
        s.map(      "Accent.TButton",          background=[("active", C_ACCENT),
                                                            ("disabled", "#555")])
        # ── secondary buttons (Browse / Export)
        s.configure("Sub.TButton",             background="#444", foreground=C_FG,
                    font=("Helvetica", 9), padding=(14, 5))
        s.map(      "Sub.TButton",             background=[("active", "#555"),
                                                            ("disabled", "#3a3a3a")],
                    foreground=[("disabled", C_DIM)])

        # ── results treeview
        s.configure("Treeview",                background=C_FRAME, foreground=C_FG,
                    fieldbackground=C_FRAME, rowheight=24, font=("Helvetica", 9))
        s.configure("Treeview.Heading",        background="#3a3a3a", foreground=C_ACCENT,
                    font=("Helvetica", 9, "bold"))
        s.map(      "Treeview",                background=[("selected", C_ACC_DK)])

        # ── summary / status
        s.configure("Summary.TLabel",          background=C_FRAME, foreground=C_ACCENT,
                    font=("Helvetica", 10, "bold"))
        s.configure("Status.TLabel",           background=C_LOG_BG, foreground="#888",
                    font=("Helvetica", 8), padding=(6, 3))

    # ─── BUILD UI ────────────────────────────────────────
    def _build(self):
        # ── helper: styled Entry widget ──────────────────
        def ent(parent, row, col, default, w=8):
            e = tk.Entry(parent, width=w, bg=C_ENTRY, fg=C_FG,
                         disabledbackground="#2a2a2a", disabledforeground=C_DIM,
                         insertbackground=C_FG, relief=tk.FLAT, bd=4,
                         font=("Helvetica", 9))
            e.grid(row=row, column=col, pady=4, padx=2)
            e.insert(0, default)
            return e

        # ──────────── INPUT ──────────────────────────────
        inp = ttk.LabelFrame(self.root, text=" Input ", padding=(12, 10))
        inp.pack(fill=tk.X, padx=15, pady=(14, 4))

        # single / batch radio buttons
        mf = ttk.Frame(inp)
        mf.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 6))
        self.mode = tk.StringVar(value="single")
        ttk.Radiobutton(mf, text="Single File",  variable=self.mode,
                        value="single", command=self._on_mode).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Radiobutton(mf, text="Batch Folder", variable=self.mode,
                        value="batch",  command=self._on_mode).pack(side=tk.LEFT)

        # path entry + browse
        ttk.Label(inp, text="Path:").grid(row=1, column=0, sticky=tk.W)
        self.path_entry = tk.Entry(inp, textvariable=self.selected_path, width=74,
                                   bg=C_ENTRY, fg=C_FG, insertbackground=C_FG,
                                   relief=tk.FLAT, bd=4, font=("Helvetica", 9))
        self.path_entry.grid(row=1, column=1, padx=(8, 6))
        ttk.Button(inp, text="Browse…", style="Sub.TButton",
                   command=self._browse).grid(row=1, column=2)

        # ──────────── PARAMETERS ─────────────────────────
        par = ttk.LabelFrame(self.root, text=" Parameters ", padding=(12, 10))
        par.pack(fill=tk.X, padx=15, pady=4)

        # D-band range
        ttk.Label(par, text="D-band (cm⁻¹):").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.d_min = ent(par, 0, 1, "1100")
        ttk.Label(par, text="–").grid(row=0, column=2)
        self.d_max = ent(par, 0, 3, "1500")

        # G-band range
        ttk.Label(par, text="G-band (cm⁻¹):").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.g_min = ent(par, 1, 1, "1500")
        ttk.Label(par, text="–").grid(row=1, column=2)
        self.g_max = ent(par, 1, 3, "1750")

        # baseline dropdown
        ttk.Label(par, text="Baseline:").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.baseline = ttk.Combobox(par, values=["none", "linear"], state="readonly", width=10)
        self.baseline.set("linear")
        self.baseline.grid(row=2, column=1, pady=4)

        # separator between baseline and smoothing
        ttk.Separator(par, orient=tk.VERTICAL).grid(row=2, column=4, padx=18, pady=4, sticky="ns")

        # smoothing checkbox + window / poly entries
        self.smooth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(par, text="Smoothing (S-G)", variable=self.smooth_var,
                        command=self._toggle_smooth).grid(row=2, column=5, pady=4, padx=(0, 8))
        ttk.Label(par, text="Window:").grid(row=2, column=6, pady=4)
        self.sw = ent(par, 2, 7, "11", w=5)
        self.sw.config(state="disabled")
        ttk.Label(par, text="Poly:").grid(row=2, column=8, pady=4)
        self.sp = ent(par, 2, 9, "3", w=5)
        self.sp.config(state="disabled")

        # skip rows
        ttk.Label(par, text="Skip rows:").grid(row=3, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.skiprows = ent(par, 3, 1, "0", w=5)

        # ──────────── RUN BUTTON ─────────────────────────
        self.run_btn = ttk.Button(self.root, text="▶  Run Analysis",
                                  style="Accent.TButton", command=self._run)
        self.run_btn.pack(pady=(10, 4))

        # ──────────── RESULTS TABLE ──────────────────────
        res = ttk.LabelFrame(self.root, text=" Results ", padding=(10, 6))
        res.pack(fill=tk.BOTH, expand=True, padx=15, pady=4)

        # treeview lives in its own frame so the scrollbar packs cleanly
        tree_wrap = ttk.Frame(res)
        tree_wrap.pack(fill=tk.BOTH, expand=True)

        cols   = ("name", "d_auc", "d_rng", "g_auc", "g_rng", "ratio")
        hdrs   = ("Sample", "D AUC", "D Range (cm⁻¹)", "G AUC", "G Range (cm⁻¹)", "D/G Ratio")
        widths = (240, 80, 130, 80, 130, 80)

        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", selectmode="browse")
        for c, h, w in zip(cols, hdrs, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor=(tk.W if c == "name" else tk.CENTER), minwidth=60)
        self.tree.tag_configure("alt", background=C_ROW_ALT)   # alternating row colour

        vsb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT,  fill=tk.BOTH, expand=True)
        vsb.pack(       side=tk.RIGHT, fill=tk.Y)

        # summary label (left) + Export button (right)
        sf = ttk.Frame(res)
        sf.pack(fill=tk.X, pady=(6, 0))
        self.summary_lbl = ttk.Label(sf, text="", style="Summary.TLabel")
        self.summary_lbl.pack(side=tk.LEFT)
        self.export_btn  = ttk.Button(sf, text="Export CSV", style="Sub.TButton",
                                      command=self._export, state="disabled")
        self.export_btn.pack(side=tk.RIGHT)

        # ──────────── LOG ────────────────────────────────
        log_f = ttk.LabelFrame(self.root, text=" Log ", padding=(6, 4))
        log_f.pack(fill=tk.X, padx=15, pady=(4, 4))

        log_wrap = ttk.Frame(log_f)
        log_wrap.pack(fill=tk.X)
        self.log_box = tk.Text(log_wrap, height=7, bg=C_LOG_BG, fg="#999",
                               font=("Consolas", 8), state="disabled",
                               relief=tk.FLAT, wrap=tk.WORD)
        log_vsb = ttk.Scrollbar(log_wrap, orient=tk.VERTICAL, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=log_vsb.set)
        self.log_box.pack(side=tk.LEFT,  fill=tk.X, expand=True)
        log_vsb.pack(     side=tk.RIGHT, fill=tk.Y)

        # ──────────── STATUS BAR ─────────────────────────
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status, style="Status.TLabel",
                  anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    # ─────────────────────────────────────────────────────
    #  WIDGET CALLBACKS
    # ─────────────────────────────────────────────────────
    def _on_mode(self):

        self.selected_path.set("")

    def _browse(self):

        if self.mode.get() == "single":
            p = filedialog.askopenfilename(
                title="Select Raman Spectrum",
                filetypes=[("Text", "*.txt"), ("CSV", "*.csv"), ("All", "*.*")])
        else:
            p = filedialog.askdirectory(title="Select Folder")
        if p:
            self.selected_path.set(p)

    def _toggle_smooth(self):

        st = "normal" if self.smooth_var.get() else "disabled"
        self.sw.config(state=st)
        self.sp.config(state=st)

    # ── log helpers ──────────────────────────────────────
    def _log(self, txt):
        self.log_box.config(state="normal")
        self.log_box.insert(tk.END, txt)
        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state="disabled")

    # ─────────────────────────────────────────────────────
    #  INPUT VALIDATION
    # ─────────────────────────────────────────────────────
    def _get_params(self):

        path = self.selected_path.get().strip()
        if not path:
            messagebox.showwarning("Warning", "Please select a file or folder first.")
            return None
        try:
            return dict(
                path             = path,
                mode             = self.mode.get(),
                d_range          = (float(self.d_min.get()), float(self.d_max.get())),
                g_range          = (float(self.g_min.get()), float(self.g_max.get())),
                baseline         = self.baseline.get(),
                smooth           = self.smooth_var.get(),
                smooth_window    = int(self.sw.get()),
                smooth_polyorder = int(self.sp.get()),
                skiprows         = int(self.skiprows.get()),
            )
        except ValueError:
            messagebox.showerror("Error", "One or more fields contain invalid values.\n"
                                          "Ranges must be numbers; window / poly / skiprows must be integers.")
            return None

    # ─────────────────────────────────────────────────────
    #  RUN ANALYSIS
    # ─────────────────────────────────────────────────────
    def _run(self):

        p = self._get_params()
        if p is None:
            return

        # reset everything
        self._clear_log()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.summary_lbl.config(text="")
        self.export_btn.config(state="disabled")
        self.run_btn.config(state="disabled")
        self.status.set("Running…")

        # heavy work goes in a daemon thread so the GUI stays responsive
        threading.Thread(target=self._execute, args=(p,), daemon=True).start()

    def _execute(self, p):

        orig = sys.stdout
        buf  = io.StringIO()
        sys.stdout = buf                  # capture prints from raman module

        try:
            if p["mode"] == "single":
                x, y = raman.load_raman_data(p["path"], skiprows=p["skiprows"])

                d_auc, d_bnd = raman.auc_inward(
                    x, y, *p["d_range"],
                    baseline=p["baseline"], smooth=p["smooth"],
                    smooth_window=p["smooth_window"], smooth_polyorder=p["smooth_polyorder"])
                g_auc, g_bnd = raman.auc_inward(
                    x, y, *p["g_range"],
                    baseline=p["baseline"], smooth=p["smooth"],
                    smooth_window=p["smooth_window"], smooth_polyorder=p["smooth_polyorder"])

                self.results_df = pd.DataFrame([{
                    "Sample":        Path(p["path"]).stem,
                    "D_AUC":         d_auc,
                    "D_range_used":  f"{d_bnd[0]:.2f}–{d_bnd[1]:.2f}",
                    "G_AUC":         g_auc,
                    "G_range_used":  f"{g_bnd[0]:.2f}–{g_bnd[1]:.2f}",
                    "AUC(D)/AUC(G)": d_auc / g_auc,
                }])

            else:   # batch
                self.results_df = raman.analyze_raman_batch(
                    folder           = p["path"],
                    d_range          = p["d_range"],
                    g_range          = p["g_range"],
                    baseline         = p["baseline"],
                    smooth           = p["smooth"],
                    smooth_window    = p["smooth_window"],
                    smooth_polyorder = p["smooth_polyorder"],
                    skiprows         = p["skiprows"],
                )
                # batch returns a DF with sample names as index → make it a column
                self.results_df = self.results_df.reset_index().rename(columns={"index": "Sample"})

            # schedule GUI update on the main thread
            self.root.after(0, self._finish, buf.getvalue(), None)

        except Exception as e:
            self.root.after(0, self._finish, buf.getvalue(), str(e))

        finally:
            sys.stdout = orig             # always restore stdout

    # ─────────────────────────────────────────────────────
    #  FINISH — runs back on the main thread
    # ─────────────────────────────────────────────────────
    def _finish(self, log_txt, error):
        if log_txt:
            self._log(log_txt)

        if error:
            self._log(f"\n❌  {error}\n")
            self.status.set(f"Error — {error}")
            messagebox.showerror("Error", error)
        else:
            self._populate()

        self.run_btn.config(state="normal")   # re-enable Run

    def _populate(self):

        if self.results_df is None or self.results_df.empty:
            self.status.set("No results.")
            return

        for i, (_, r) in enumerate(self.results_df.iterrows()):
            tag = ("alt",) if i % 2 else ()   # alternating row shading
            self.tree.insert("", tk.END, tags=tag, values=(
                r["Sample"],
                f"{r['D_AUC']:.2f}",
                r["D_range_used"],
                f"{r['G_AUC']:.2f}",
                r["G_range_used"],
                f"{r['AUC(D)/AUC(G)']:.4f}",
            ))

        # summary line below the table
        ratios = self.results_df["AUC(D)/AUC(G)"]
        if len(ratios) > 1:
            self.summary_lbl.config(
                text=f"Mean D/G = {ratios.mean():.4f} ± {ratios.std():.4f}   (n = {len(ratios)})")
        else:
            self.summary_lbl.config(text=f"D/G = {ratios.iloc[0]:.4f}")

        self.export_btn.config(state="normal")
        n = len(self.results_df)
        self.status.set(f"Done — {n} spectrum{'' if n == 1 else 'tra'} processed.")

    # ─────────────────────────────────────────────────────
    #  EXPORT
    # ─────────────────────────────────────────────────────
    def _export(self):

        if self.results_df is None or self.results_df.empty:
            return
        fp = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if fp:
            self.results_df.to_csv(fp, index=False)
            self.status.set(f"Exported → {Path(fp).name}")
            messagebox.showinfo("Saved", f"Results saved to:\n{fp}")


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    RamanGUI(root)
    root.mainloop()
