#!/usr/bin/env python3
"""Rewrite annotation frame numbers onto real video frames.

Masks written by the web annotator are named for a counter that advanced on a fixed 30 fps clock
rather than on the video's own frames. Navigating backwards seeks to `counter / 30` seconds, and
that is the image a tracing was drawn on, so the real frame is the one whose display window
contains that time.

Because 30 counter slots cover less than 30 real frames on a slower clip, two counters can fall
inside one frame's window. Those frames are merged: structures annotated on only one counter are
carried over untouched, and a class annotated on several is taken from the counter holding more
structures, or from the higher counter when they hold the same number.

Reads the source tree only; everything is written to a new destination.
"""

import argparse
import bisect
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

COUNTER_CLOCK = 30.0
MASK_PATTERN = re.compile(r"^frame_(\d{6})_(.+?)_(\d+)\.png$", re.IGNORECASE)


def frame_times(video):
    """Presentation time of every frame, in order."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    times = sorted(float(t) for t in out.replace(",", "").split() if t)
    if not times:
        raise ValueError(f"no frames found in {video}")
    return times


def is_variable_rate(times):
    if len(times) < 3:
        return False
    deltas = {round(times[i + 1] - times[i], 4) for i in range(len(times) - 1)}
    return len(deltas) > 1


def real_frame(times, counter):
    """The frame displayed by a seek to counter/30 seconds."""
    index = bisect.bisect_right(times, counter / COUNTER_CLOCK) - 1
    return max(0, min(len(times) - 1, index))


def read_masks(folder):
    """{counter: {class: [(instance_id, path), ...]}}"""
    counters = defaultdict(lambda: defaultdict(list))
    for path in sorted(folder.iterdir()):
        match = MASK_PATTERN.match(path.name)
        if match:
            counters[int(match.group(1))][match.group(2).lower()].append((int(match.group(3)), path))
    return counters


def resolve(counters, group):
    """Pick the surviving masks for one real frame. Returns (kept, dropped, decisions)."""
    if len(group) == 1:
        only = group[0]
        return [(only, cls, inst, path)
                for cls, items in counters[only].items() for inst, path in items], [], []

    sizes = {c: sum(len(v) for v in counters[c].values()) for c in group}
    owners = defaultdict(list)
    for counter in group:
        for cls in counters[counter]:
            owners[cls].append(counter)

    kept, dropped, decisions = [], [], []
    for cls, holders in sorted(owners.items()):
        if len(holders) == 1:
            winner = holders[0]
        else:
            # More structures wins; a tie goes to the higher counter, the first one reached
            # when scrolling backwards.
            winner = max(holders, key=lambda c: (sizes[c], c))
            reason = "more structures" if sizes[winner] > min(sizes[h] for h in holders) else "higher counter"
            decisions.append((cls, winner, [h for h in holders if h != winner], reason))
        for counter in holders:
            target = kept if counter == winner else dropped
            for inst, path in counters[counter][cls]:
                target.append((counter, cls, inst, path))
    return kept, dropped, decisions


def migrate_video(folder, video, dest, report):
    times = frame_times(video)
    counters = read_masks(folder)
    if not counters:
        return 0, 0
    if is_variable_rate(times):
        report.append([folder.name, "warning", "", "", "", "", "variable frame rate"])

    by_frame = defaultdict(list)
    for counter in sorted(counters):
        by_frame[real_frame(times, counter)].append(counter)

    dest.mkdir(parents=True, exist_ok=True)
    written = removed = 0
    for frame, group in sorted(by_frame.items()):
        kept, dropped, decisions = resolve(counters, group)
        for cls, winner, losers, reason in decisions:
            report.append([folder.name, "merge", frame, ",".join(str(c) for c in group), cls,
                           f"kept counter {winner}", f"{reason}; dropped {','.join(str(c) for c in losers)}"])
        used = set()
        for counter, cls, inst, path in sorted(kept, key=lambda k: (k[1], k[2])):
            instance = inst
            while (cls, instance) in used:
                instance += 1
            used.add((cls, instance))
            name = f"frame_{frame:06d}_{cls}_{instance:03d}.png"
            shutil.copy2(path, dest / name)
            report.append([folder.name, "rename", frame, counter, cls, f"{path.name} -> {name}", ""])
            written += 1
        for counter, cls, inst, path in dropped:
            report.append([folder.name, "drop", frame, counter, cls, path.name, "duplicate label on this frame"])
            removed += 1
    return written, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", type=Path, help="source annotation root (one folder per video)")
    parser.add_argument("videos", type=Path, help="root to search for the source videos")
    parser.add_argument("destination", type=Path, help="new tree to write; must not exist")
    args = parser.parse_args()

    if args.destination.exists():
        sys.exit(f"destination already exists: {args.destination}")
    if not args.annotations.is_dir():
        sys.exit(f"not a directory: {args.annotations}")

    videos = {p.stem: p for p in args.videos.rglob("*") if p.suffix.lower() in {".mp4", ".m4v", ".mov", ".webm"}}
    report = [["video", "action", "real_frame", "counter", "class", "detail", "note"]]
    args.destination.mkdir(parents=True)

    total_written = total_dropped = skipped = 0
    for folder in sorted(p for p in args.annotations.iterdir() if p.is_dir()):
        video = videos.get(folder.name)
        if video is None:
            report.append([folder.name, "skipped", "", "", "", "", "no matching video found"])
            skipped += 1
            continue
        written, dropped = migrate_video(folder, video, args.destination / folder.name, report)
        total_written += written
        total_dropped += dropped

    manifest = args.annotations / "nerve_manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, args.destination / manifest.name)

    with (args.destination / "migration_report.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(report)

    merges = sum(1 for row in report[1:] if row[1] == "merge")
    print(f"wrote {total_written} masks to {args.destination}")
    print(f"{merges} merged frame(s), {total_dropped} duplicate mask(s) dropped, {skipped} folder(s) skipped")
    print(f"report: {args.destination / 'migration_report.csv'}")


if __name__ == "__main__":
    main()
