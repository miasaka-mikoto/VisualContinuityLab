"""Small Tk fallback used when PySide6 is unavailable.

It intentionally exposes the same core actions as the Qt application (import,
analyze, export) without trying to replicate every visualization widget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .adapter import AnalysisAdapter, FrameRecord, result_to_json


def run_tk(adapter: AnalysisAdapter | None = None) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    engine = adapter or AnalysisAdapter()
    records: list[FrameRecord] = []
    result: dict = {}

    root = tk.Tk()
    root.title("Visual Continuity Lab — AI 图像角色一致性与连续性实验室")
    root.geometry("1050x700")
    root.configure(bg="#0d151d")

    listbox = tk.Listbox(root, bg="#111c25", fg="#d8e4ec", selectbackground="#2c5960")
    listbox.pack(side="left", fill="y", padx=10, pady=10)
    right = tk.Frame(root, bg="#0d151d")
    right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
    status = tk.StringVar(value="Ready — add a sequence to begin")
    tk.Label(right, textvariable=status, bg="#0d151d", fg="#9fb2c0", anchor="w").pack(fill="x")
    health = tk.Text(right, height=8, bg="#111c25", fg="#d8e4ec", relief="flat")
    health.pack(fill="x", pady=8)
    issues = ttk.Treeview(right, columns=("type", "start", "end", "severity", "evidence"), show="headings")
    for key, title in (("type", "Type"), ("start", "Start"), ("end", "End"), ("severity", "Severity"), ("evidence", "Evidence")):
        issues.heading(key, text=title)
        issues.column(key, width=120 if key != "evidence" else 420)
    issues.pack(fill="both", expand=True)

    def refresh() -> None:
        listbox.delete(0, tk.END)
        for record in records:
            listbox.insert(tk.END, f"{record.frame_id}  {Path(record.path).name}")

    def add(paths: Sequence[str]) -> None:
        nonlocal records
        records.extend(engine.load_paths(paths))
        for i, record in enumerate(records, 1):
            record.frame_number, record.frame_id = i, f"F{i:04d}"
        refresh()
        status.set(f"Loaded {len(records)} frame(s)")

    def add_images() -> None:
        add(filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")]))

    def add_folder() -> None:
        folder = filedialog.askdirectory()
        if folder:
            add([folder])

    def analyze() -> None:
        nonlocal result
        if not records:
            messagebox.showinfo("No sequence", "Add images or a folder first.")
            return
        result = engine.analyze(records)
        health.delete("1.0", tk.END)
        for key, value in result.get("health", {}).items():
            health.insert(tk.END, f"{key}: {value:.1f}\n" if isinstance(value, (float, int)) else f"{key}: {value}\n")
        for item in issues.get_children():
            issues.delete(item)
        for issue in result.get("issues", []):
            issues.insert("", tk.END, values=(issue.get("type"), issue.get("start_frame"), issue.get("end_frame"), issue.get("severity"), issue.get("evidence")))
        status.set(f"Analysis complete — {len(result.get('issues', []))} issue(s)")

    def export() -> None:
        if not result:
            return
        target = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if target:
            Path(target).write_text(result_to_json({**result, "sequence": [x.to_dict() for x in records]}), encoding="utf-8")
            status.set(f"Exported {target}")

    toolbar = tk.Frame(right, bg="#0d151d")
    toolbar.pack(fill="x", pady=4)
    for title, callback in (("Add Images", add_images), ("Add Folder", add_folder), ("Analyze", analyze), ("Export JSON", export)):
        tk.Button(toolbar, text=title, command=callback, bg="#203342", fg="#e5f1f6", relief="flat", padx=10).pack(side="left", padx=(0, 6))
    root.mainloop()

