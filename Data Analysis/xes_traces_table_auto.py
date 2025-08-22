#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, gzip, csv
import xml.etree.ElementTree as ET
from typing import Optional

# ---- تنظیمات ----
PRINT_EVERY = 5000  # هر چند تریس یک‌بار پیام پیشرفت
MAX_TRACES: Optional[int] = None  # None یعنی همه تریس‌ها

# کلیدهای متداول برای شناسه‌ی کیس و نام اکتیویتی داخل XES
TRACE_ID_KEYS = ("case:concept:name", "concept:name", "case:id", "case-id")
EVENT_ACT_KEYS = ("concept:name", "activity", "Activity")

def open_maybe_gz(path):
    return gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")

def local(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag

def extract_case_id(trace_elem):
    for s in trace_elem.findall("./{*}string"):
        k = s.attrib.get("key")
        if k in TRACE_ID_KEYS:
            v = s.attrib.get("value")
            if v:
                return v
    return None

def extract_activities_in_trace(trace_elem):
    seq = []
    for ev in trace_elem.findall("./{*}event"):
        act = None
        for s in ev.findall("./{*}string"):
            if s.attrib.get("key") in EVENT_ACT_KEYS:
                act = s.attrib.get("value")
                break
        if act is not None:
            seq.append(act)
    return seq

def resolve_default_input(script_dir):
    # 1) ../raw_datasets/Hospital Billing - Event Log.xes[.gz]
    candidates = [
        os.path.normpath(os.path.join(script_dir, "..", "raw_datasets", "Hospital Billing - Event Log.xes.gz")),
        os.path.normpath(os.path.join(script_dir, "..", "raw_datasets", "Hospital Billing - Event Log.xes")),
        os.path.expanduser(os.path.join("~", "Downloads", "Hospital Billing - Event Log.xes.gz")),
        os.path.expanduser(os.path.join("~", "Downloads", "Hospital Billing - Event Log.xes")),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "XES file not found. Tried:\n  - " + "\n  - ".join(candidates) +
        "\nTip: pass the XES path explicitly as the first argument."
    )

def resolve_paths(cli_in: Optional[str], cli_out: Optional[str]):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # ورودی
    if cli_in and os.path.isfile(cli_in):
        xes_path = os.path.normpath(cli_in)
    else:
        xes_path = resolve_default_input(script_dir)

    # خروجی
    default_out_dir = os.path.normpath(os.path.join(script_dir, "..", "outputs"))
    os.makedirs(default_out_dir, exist_ok=True)
    default_out_csv = os.path.join(default_out_dir, "hospital_billing_all_traces.csv")

    out_csv = os.path.normpath(cli_out) if cli_out else default_out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    return xes_path, out_csv

def main():
    # استفاده:
    #   python3 xes_traces_dump_all_auto.py [optional:/path/to/log.xes[.gz]] [optional:/path/to/output.csv]
    cli_in = sys.argv[1] if len(sys.argv) >= 2 else None
    cli_out = sys.argv[2] if len(sys.argv) >= 3 else None

    xes_path, out_csv = resolve_paths(cli_in, cli_out)

    total_traces = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fp_out:
        writer = csv.writer(fp_out)
        writer.writerow(["#", "Case ID", "Trace Length", "Activities"])

        with open_maybe_gz(xes_path) as f:
            for ev, elem in ET.iterparse(f, events=("end",)):
                if local(elem.tag) == "trace":
                    total_traces += 1
                    cid = extract_case_id(elem) or f"Trace#{total_traces}"
                    acts = extract_activities_in_trace(elem)
                    writer.writerow([total_traces, cid, len(acts), " | ".join(acts)])
                    elem.clear()

                    if PRINT_EVERY and (total_traces % PRINT_EVERY == 0):
                        print(f"[progress] written {total_traces} traces ...")

                    if MAX_TRACES is not None and total_traces >= MAX_TRACES:
                        break

    print("=== XES Dump (ALL) ===")
    print(f"File: {xes_path}")
    print(f"Written traces: {total_traces}")
    print(f"CSV: {out_csv}")

if __name__ == "__main__":
    main()
