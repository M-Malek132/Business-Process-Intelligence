#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, gzip, csv
import xml.etree.ElementTree as ET

TRACE_ID_KEYS = ("case:concept:name", "concept:name", "case:id", "case-id")
EVENT_ACT_KEYS = ("concept:name", "activity", "Activity")
PRINT_EVERY = 5000  # هر چند تا ترِیس یکبار پیام پیشرفت بده

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

def main():
    if len(sys.argv) != 3:
        print("Usage:\n  python3 xes_traces_dump_all.py /path/to/log.xes[.gz] /path/to/output.csv")
        sys.exit(1)

    xes_path = sys.argv[1]
    out_csv  = sys.argv[2]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    total_traces = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fp_out:
        writer = csv.writer(fp_out)
        writer.writerow(["#", "Case ID", "Trace Length", "Activities"])

        with open_maybe_gz(xes_path) as f:
            # فقط روی رویداد پایان trace کار می‌کنیم و همان‌جا یک ردیف می‌نویسیم
            for ev, elem in ET.iterparse(f, events=("end",)):
                if local(elem.tag) == "trace":
                    total_traces += 1
                    cid = extract_case_id(elem) or f"Trace#{total_traces}"
                    acts = extract_activities_in_trace(elem)
                    writer.writerow([total_traces, cid, len(acts), " | ".join(acts)])
                    elem.clear()

                    if total_traces % PRINT_EVERY == 0:
                        print(f"[progress] written {total_traces} traces ...")

    print(f"✅ Done! Written {total_traces} traces to {out_csv}")

if __name__ == "__main__":
    main()
