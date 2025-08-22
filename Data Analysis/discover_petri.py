#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys, os, gzip
from pathlib import Path
from typing import Optional, Tuple, List

# ---- تنظیم خروجی‌ها
OUT_DIR = Path.cwd() / "outputs_petri"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNML_PATH = OUT_DIR / "model.pnml"
PNG_PATH  = OUT_DIR / "model.png"

# ---- ایمپورت‌های pm4py (سازگار با pm4py 2.7.x)
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as im_alg
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

# --------- کمک‌تابع: پیدا کردن اولین فایل XES/XES.GZ ----------
def find_first_xes(candidates_dirs: List[Path]) -> Optional[Path]:
    exts = (".xes", ".xes.gz")
    for d in candidates_dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts or "".join(p.suffixes).lower().endswith(".xes.gz"):
                return p
    return None

def default_search_dirs() -> List[Path]:
    here = Path.cwd()
    home = Path.home()
    guess_dirs = [
        here,
        here.parent,
        here / "Data Analysis",
        here / "data",
        here / "datasets",
        here / "raw_datasets",
        home / "Downloads",
        home / "documents",
        home / "Documents",
    ]
    # یکتا سازی
    uniq: List[Path] = []
    seen = set()
    for d in guess_dirs:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq

# --------- خواندن لاگ ----------
def load_log(xes_path: Path):
    print(f">> Using log: {xes_path}")
    # pm4py خودش gzip رو هندل می‌کند؛ هم برای .xes هم .xes.gz
    log = xes_importer.apply(str(xes_path))
    print(f">> Log loaded. traces: {len(log)}")
    return log

# --------- کشف پتری‌نت با IMf (مسیر درست: درخت -> تبدیل به پتری) ----------
def discover_petri_from_log(log) -> Tuple[object, object, object]:
    print(">> Discover Process Tree (IMf) ...")
    ptree = im_alg.apply(log, variant=im_alg.Variants.IMf, parameters={"noise_threshold": 0.2})
    # تبدیل درخت به پتری‌نت
    print(">> Convert Process Tree -> Petri Net ...")
    net, imark, fmark = pt_converter.apply(ptree, variant=pt_converter.Variants.TO_PETRI_NET)
    return net, imark, fmark

# --------- ذخیره خروجی‌ها ----------
def save_outputs(net, imark, fmark):
    # PNML
    try:
        pnml_exporter.apply(net, imark, fmark, str(PNML_PATH))
        print(f"   pnml: {PNML_PATH}")
    except Exception as e:
        print("   PNML export failed:", e)

    # PNG
    try:
        gviz = pn_vis.apply(net, imark, fmark)
        pn_vis.save(gviz, str(PNG_PATH))
        print(f"   png : {PNG_PATH}")
    except Exception as e:
        print("   PNG render failed (graphviz?):", e)

def main():
    # اگر آرگومان مسیر لاگ دادی، همون رو می‌خوانه؛ وگرنه خودش می‌گرده
    xes_path: Optional[Path] = None
    if len(sys.argv) > 1:
        xes_path = Path(sys.argv[1]).expanduser()
        if not xes_path.exists():
            print("!! مسیر ورودی پیدا نشد:", xes_path)
            xes_path = None

    if xes_path is None:
        xes_path = find_first_xes(default_search_dirs())

    if xes_path is None:
        raise FileNotFoundError("هیچ فایل .xes یا .xes.gz پیدا نشد. یک لاگ داخل پوشه پروژه یا Downloads قرار بده.")

    log = load_log(xes_path)
    net, imark, fmark = discover_petri_from_log(log)
    print(">> Export outputs ...")
    save_outputs(net, imark, fmark)
    print("\n✅ Done. Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
