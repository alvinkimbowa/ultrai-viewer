"use strict";

// Reads exact per-frame presentation times out of an ISOBMFF container (.mp4/.m4v/.mov).
// HTMLVideoElement exposes only `duration`, so frame-accurate navigation needs the sample
// tables the container already carries: `stts` gives decode deltas, `ctts` the composition
// offsets that turn them into presentation times, `mdhd` the timescale they are counted in.
window.MP4Frames = (() => {
  const CONTAINERS = new Set(["moov", "trak", "mdia", "minf", "stbl"]);

  function boxes(view, start, end) {
    const found = [];
    let pos = start;
    while (pos + 8 <= end) {
      let size = view.getUint32(pos);
      const type = String.fromCharCode(view.getUint8(pos + 4), view.getUint8(pos + 5), view.getUint8(pos + 6), view.getUint8(pos + 7));
      let header = 8;
      if (size === 1) { size = Number(view.getBigUint64(pos + 8)); header = 16; }
      else if (size === 0) size = end - pos;
      if (size < header || pos + size > end) break;
      found.push({ type, body: pos + header, end: pos + size });
      pos += size;
    }
    return found;
  }

  function walk(view, start, end, visit) {
    for (const box of boxes(view, start, end)) {
      visit(box);
      if (CONTAINERS.has(box.type)) walk(view, box.body, box.end, visit);
    }
  }

  function readTrack(view, trakStart, trakEnd) {
    let timescale = 0, isVideo = false, stts = null, ctts = null;
    walk(view, trakStart, trakEnd, (box) => {
      if (box.type === "hdlr") {
        const handler = String.fromCharCode(view.getUint8(box.body + 8), view.getUint8(box.body + 9), view.getUint8(box.body + 10), view.getUint8(box.body + 11));
        if (handler === "vide") isVideo = true;
      } else if (box.type === "mdhd") {
        timescale = view.getUint8(box.body) === 1 ? view.getUint32(box.body + 20) : view.getUint32(box.body + 12);
      } else if (box.type === "stts") stts = box;
      else if (box.type === "ctts") ctts = box;
    });
    if (!isVideo || !timescale || !stts) return null;

    const decode = [];
    const sttsCount = view.getUint32(stts.body + 4);
    let time = 0;
    for (let i = 0, p = stts.body + 8; i < sttsCount && p + 8 <= stts.end; i++, p += 8) {
      const samples = view.getUint32(p), delta = view.getUint32(p + 4);
      for (let s = 0; s < samples; s++) { decode.push(time); time += delta; }
    }
    if (!decode.length) return null;

    const offsets = new Array(decode.length).fill(0);
    if (ctts) {
      const version = view.getUint8(ctts.body);
      const cttsCount = view.getUint32(ctts.body + 4);
      let index = 0;
      for (let i = 0, p = ctts.body + 8; i < cttsCount && p + 8 <= ctts.end; i++, p += 8) {
        const samples = view.getUint32(p);
        const offset = version === 1 ? view.getInt32(p + 4) : view.getUint32(p + 4);
        for (let s = 0; s < samples && index < offsets.length; s++) offsets[index++] = offset;
      }
    }

    const times = decode.map((dts, i) => (dts + offsets[i]) / timescale);
    times.sort((a, b) => a - b);
    // Composition offsets push the first frame off zero; the media timeline a player exposes
    // starts at the earliest presentation time, so rebase onto it.
    const first = times[0];
    return times.map((t) => t - first);
  }

  // Scans only the top-level box list, then reads `moov` alone, so a large file is never
  // pulled into memory whole.
  async function frameTimes(file) {
    let pos = 0;
    while (pos + 8 <= file.size) {
      const head = new DataView(await file.slice(pos, Math.min(pos + 16, file.size)).arrayBuffer());
      if (head.byteLength < 8) break;
      let size = head.getUint32(0);
      const type = String.fromCharCode(head.getUint8(4), head.getUint8(5), head.getUint8(6), head.getUint8(7));
      if (size === 1) {
        if (head.byteLength < 16) break;
        size = Number(head.getBigUint64(8));
      } else if (size === 0) size = file.size - pos;
      if (size < 8) break;
      if (type === "moov") {
        const view = new DataView(await file.slice(pos, pos + size).arrayBuffer());
        let best = null;
        for (const box of boxes(view, 0, view.byteLength)) {
          if (box.type !== "moov") continue;
          for (const child of boxes(view, box.body, box.end)) {
            if (child.type !== "trak") continue;
            const times = readTrack(view, child.body, child.end);
            if (times && (!best || times.length > best.length)) best = times;
          }
        }
        return best;
      }
      pos += size;
    }
    return null;
  }

  return { frameTimes };
})();
