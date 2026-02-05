# TKinter-based GUI for the Raman Spectroscopy AUC Analysis Tool.
# Imports analysis functions from  AUC_calculation_v2.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, sys, io
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

import AUC_calculation_v2 as raman          

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
        self.last_params   = None

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
        self.baseline = ttk.Combobox(par, values=["none", "linear", "polynomial"], state="readonly", width=10)
        self.baseline.bind("<<ComboboxSelected>>", self._toggle_baseline_poly)
        self.baseline.set("linear")
        self.baseline.grid(row=2, column=1, pady=4)

        # baseline polynomial order
        ttk.Label(par, text="Poly Order:").grid(row=2, column=2, pady=4, padx=(12, 4))
        self.baseline_poly = ent(par, 2, 3, "2", w=5)
        self.baseline_poly.config(state="disabled")

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
        
        # show plot checkbox
        ttk.Label(par, text="").grid(row=3, column=2)  # Spacer
        self.show_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(par, text="Show plot preview", variable=self.show_plot_var
                       ).grid(row=3, column=3, columnspan=2, pady=4, sticky=tk.W)

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
        self.tree.tag_configure("alt", background=C_ROW_ALT)

        vsb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT,  fill=tk.BOTH, expand=True)
        vsb.pack(       side=tk.RIGHT, fill=tk.Y)

        # summary label (left) + buttons (right)
        sf = ttk.Frame(res)
        sf.pack(fill=tk.X, pady=(6, 0))
        self.summary_lbl = ttk.Label(sf, text="", style="Summary.TLabel")
        self.summary_lbl.pack(side=tk.LEFT)
        
        self.export_plots_btn = ttk.Button(sf, text="Export All Plots", style="Sub.TButton",
                                          command=self._export_plots, state="disabled")
        self.export_plots_btn.pack(side=tk.RIGHT, padx=(0, 6))
        
        self.preview_btn = ttk.Button(sf, text="Preview Selected", style="Sub.TButton",
                                      command=self._preview_selected, state="disabled")
        self.preview_btn.pack(side=tk.RIGHT, padx=(0, 6))
        
        self.export_btn  = ttk.Button(sf, text="Export CSV", style="Sub.TButton",
                                      command=self._export, state="disabled")
        self.export_btn.pack(side=tk.RIGHT, padx=(0, 6))

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

    def _toggle_baseline_poly(self, event=None):
        st = "normal" if self.baseline.get() == "polynomial" else "disabled"
        self.baseline_poly.config(state=st)

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
                path                = path,
                mode                = self.mode.get(),
                d_range             = (float(self.d_min.get()), float(self.d_max.get())),
                g_range             = (float(self.g_min.get()), float(self.g_max.get())),
                baseline            = self.baseline.get(),
                baseline_poly_order = int(self.baseline_poly.get()),  
                smooth              = self.smooth_var.get(),
                smooth_window       = int(self.sw.get()),
                smooth_polyorder    = int(self.sp.get()),
                skiprows            = int(self.skiprows.get()),
                show_plot           = self.show_plot_var.get(),
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
        self.preview_btn.config(state="disabled")
        self.export_plots_btn.config(state="disabled")
        self.run_btn.config(state="disabled")
        self.status.set("Running…")

        # heavy work goes in a daemon thread so the GUI stays responsive
        threading.Thread(target=self._execute, args=(p,), daemon=True).start()

    def _execute(self, p):
        orig = sys.stdout
        buf  = io.StringIO()
        sys.stdout = buf

        try:
            if p["mode"] == "single":
                x, y = raman.load_raman_data(p["path"], skiprows=p["skiprows"])

                d_auc, d_bnd = raman.auc_inward(
                    x, y, *p["d_range"],
                    baseline=p["baseline"], 
                    baseline_poly_order=p["baseline_poly_order"], 
                    smooth=p["smooth"],
                    smooth_window=p["smooth_window"], 
                    smooth_polyorder=p["smooth_polyorder"],
                    return_plot_data=False)
                    
                g_auc, g_bnd = raman.auc_inward(
                    x, y, *p["g_range"],
                    baseline=p["baseline"], 
                    baseline_poly_order=p["baseline_poly_order"], 
                    smooth=p["smooth"],
                    smooth_window=p["smooth_window"], 
                    smooth_polyorder=p["smooth_polyorder"],
                    return_plot_data=False)

                self.results_df = pd.DataFrame([{
                    "Sample":        Path(p["path"]).stem,
                    "D_AUC":         d_auc,
                    "D_range_used":  f"{d_bnd[0]:.2f}–{d_bnd[1]:.2f}",
                    "G_AUC":         g_auc,
                    "G_range_used":  f"{g_bnd[0]:.2f}–{g_bnd[1]:.2f}",
                    "AUC(D)/AUC(G)": d_auc / g_auc,
                }])
                
                # Generate plot if requested
                if p["show_plot"]:
                    fig = raman.plot_auc_analysis(
                        x, y, p["d_range"], p["g_range"],
                        baseline=p["baseline"], 
                        baseline_poly_order=p["baseline_poly_order"],
                        smooth=p["smooth"], 
                        smooth_window=p["smooth_window"],
                        smooth_polyorder=p["smooth_polyorder"],
                        title=Path(p["path"]).stem
                    )
                    self.root.after(0, self._show_plot, fig)

            else:   # batch
                self.results_df = raman.analyze_raman_batch(
                    folder              = p["path"],
                    d_range             = p["d_range"],
                    g_range             = p["g_range"],
                    baseline            = p["baseline"],
                    baseline_poly_order = p["baseline_poly_order"],
                    smooth              = p["smooth"],
                    smooth_window       = p["smooth_window"],
                    smooth_polyorder    = p["smooth_polyorder"],
                    skiprows            = p["skiprows"],
                )
                self.results_df = self.results_df.reset_index().rename(columns={"index": "Sample"})

            self.root.after(0, self._finish, buf.getvalue(), None, p)

        except Exception as e:
            self.root.after(0, self._finish, buf.getvalue(), str(e), p)

        finally:
            sys.stdout = orig

    # ─────────────────────────────────────────────────────
    #  FINISH — runs back on the main thread
    # ─────────────────────────────────────────────────────
    def _finish(self, log_txt, error, params):
        if log_txt:
            self._log(log_txt)

        if error:
            self._log(f"\n❌  {error}\n")
            self.status.set(f"Error — {error}")
            messagebox.showerror("Error", error)
        else:
            self._populate()
            self.last_params = params

        self.run_btn.config(state="normal")

    def _populate(self):
        if self.results_df is None or self.results_df.empty:
            self.status.set("No results.")
            return

        for i, (_, r) in enumerate(self.results_df.iterrows()):
            tag = ("alt",) if i % 2 else ()
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
        
        # Enable preview and export plots buttons for batch mode
        if hasattr(self, 'last_params') and self.last_params.get('mode') == 'batch':
            self.preview_btn.config(state="normal")
            self.export_plots_btn.config(state="normal")
        
        n = len(self.results_df)
        self.status.set(f"Done — {n} spectrum{'' if n == 1 else 'tra'} processed.")

    # ─────────────────────────────────────────────────────
    #  PLOTTING
    # ─────────────────────────────────────────────────────
    def _show_plot(self, fig):
        """Display matplotlib figure in a new window."""
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Spectrum Analysis")
        plot_window.geometry("1000x850")
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for zoom/pan/save
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()

    def _preview_selected(self):
        """Preview plot for selected row in batch results."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a spectrum from the results table.")
            return
        
        # Get sample name from selected row
        item = self.tree.item(selection[0])
        sample_name = item['values'][0]
        
        # Load the file
        folder = Path(self.last_params['path'])
        filepath = folder / f"{sample_name}.txt"
        
        if not filepath.exists():
            messagebox.showerror("Error", f"File not found: {filepath}")
            return
        
        try:
            x, y = raman.load_raman_data(filepath, skiprows=self.last_params['skiprows'])
            
            fig = raman.plot_auc_analysis(
                x, y,
                self.last_params['d_range'],
                self.last_params['g_range'],
                baseline=self.last_params['baseline'],
                baseline_poly_order=self.last_params['baseline_poly_order'],
                smooth=self.last_params['smooth'],
                smooth_window=self.last_params['smooth_window'],
                smooth_polyorder=self.last_params['smooth_polyorder'],
                title=sample_name
            )
            self._show_plot(fig)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot:\n{str(e)}")

    def _export_plots(self):
        """Export all plots as PNG files."""
        if self.results_df is None or self.results_df.empty:
            return
        
        # Ask user to select output folder
        folder = filedialog.askdirectory(title="Select Folder to Save Plots")
        if not folder:
            return
        
        output_folder = Path(folder)
        input_folder = Path(self.last_params['path'])
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Exporting Plots")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Exporting plots...", 
                 font=("Helvetica", 10)).pack(pady=(20, 10))
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                       maximum=100, length=350)
        progress_bar.pack(pady=10)
        
        status_label = ttk.Label(progress_window, text="")
        status_label.pack(pady=5)
        
        def export_thread():
            total = len(self.results_df)
            success_count = 0
            failed = []
            
            for i, (_, row) in enumerate(self.results_df.iterrows()):
                sample_name = row['Sample']
                
                try:
                    # Update progress
                    progress = (i / total) * 100
                    progress_var.set(progress)
                    status_label.config(text=f"Processing {i+1}/{total}: {sample_name}")
                    progress_window.update()
                    
                    # Load data
                    filepath = input_folder / f"{sample_name}.txt"
                    x, y = raman.load_raman_data(filepath, skiprows=self.last_params['skiprows'])
                    
                    # Generate plot
                    fig = raman.plot_auc_analysis(
                        x, y,
                        self.last_params['d_range'],
                        self.last_params['g_range'],
                        baseline=self.last_params['baseline'],
                        baseline_poly_order=self.last_params['baseline_poly_order'],
                        smooth=self.last_params['smooth'],
                        smooth_window=self.last_params['smooth_window'],
                        smooth_polyorder=self.last_params['smooth_polyorder'],
                        title=sample_name
                    )
                    
                    # Save as PNG with same stem
                    output_path = output_folder / f"{sample_name}.png"
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)  # Close to free memory
                    
                    success_count += 1
                    
                except Exception as e:
                    failed.append((sample_name, str(e)))
            
            # Final update
            progress_var.set(100)
            progress_window.destroy()
            
            # Show results
            if failed:
                failed_msg = "\n".join([f"  - {name}: {err}" for name, err in failed[:5]])
                if len(failed) > 5:
                    failed_msg += f"\n  ... and {len(failed)-5} more"
                messagebox.showwarning("Export Complete with Errors",
                    f"Exported {success_count}/{total} plots to:\n{output_folder}\n\n"
                    f"Failed:\n{failed_msg}")
            else:
                messagebox.showinfo("Export Complete",
                    f"Successfully exported all {success_count} plots to:\n{output_folder}")
            
            self.status.set(f"Exported {success_count} plots → {output_folder.name}")
        
        # Run export in thread
        threading.Thread(target=export_thread, daemon=True).start()

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
