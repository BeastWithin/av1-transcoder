#!/usr/bin/env python3
"""
av1tx - analyze video files and transcode them to AV1.

* Default encoder: VAAPI hardware AV1 encoder (av1_vaapi), e.g. AMD RDNA4 (RX 9060 XT).
  Optional software encoder: SVT-AV1 (--encoder svt), slower but better compression.
* Each file is analyzed with ffprobe (codec, resolution, fps, bit depth, bitrate,
  interlacing, colour metadata, audio/subtitle tracks) and given a rough quality class.
* Quality guard: after encoding, the AV1 output is compared to the original with SSIM
  on several sample windows. If it is below --ssim-target, the encoder is re-run at
  higher quality. If the target cannot be reached, the original is kept untouched.
* Safety: the output is written to a *.partial.mkv file and only renamed once it has
  been verified. A file is only accepted if the AV1 output is smaller than the source.
  Originals are never deleted unless --delete-original is given.
* Video is re-encoded; audio, subtitles, chapters and metadata are copied bit-exact.

Requires ffmpeg + ffprobe. Python standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {
    ".mkv", ".mp4", ".m4v", ".mov", ".avi", ".wmv", ".flv", ".webm",
    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".vob", ".3gp",
}
OUTPUT_SUFFIX = ".av1.mkv"  # outputs are named <name>.av1.mkv (also used for samples)
PARTIAL_TAG = ".partial"

# Pixel formats the AV1 encoders accept: pix_fmt -> (bit depth, VAAPI upload format)
SUPPORTED_PIX = {
    "yuv420p": (8, "nv12"),
    "yuvj420p": (8, "nv12"),
    "yuv420p10le": (10, "p010le"),
}

VAAPI_MIN_Q = 20    # AV1 base_q_idx scale (0-255): lower = better quality
SVT_MIN_CRF = 10    # SVT-AV1 CRF scale: lower = better quality


# --------------------------------------------------------------------------- helpers

def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def hms(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def parse_rate(s: str) -> float:
    try:
        if "/" in s:
            n, d = s.split("/", 1)
            return float(n) / float(d) if float(d) else 0.0
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def last_lines(text: str, n: int = 3) -> str:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return " | ".join(lines[-n:]) if lines else "(no output)"


# --------------------------------------------------------------------------- analysis

@dataclass
class Media:
    path: Path
    size: int
    duration: float
    container: str
    video: dict | None
    audio: list
    subs: list
    codec: str = ""
    profile: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    pix_fmt: str = ""
    bit_depth: int = 8
    field_order: str = "unknown"
    interlaced: bool = False
    bitrate: float = 0.0   # bits/s of the video stream (estimated if not tagged)
    bpp: float = 0.0       # bits per pixel per frame - rough "how much data per picture"


def ffprobe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {last_lines(r.stderr)}")
    return json.loads(r.stdout)


def analyze(path: Path) -> Media:
    data = ffprobe(path)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    size = int(fmt.get("size") or path.stat().st_size)
    duration = float(fmt.get("duration") or 0.0)
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]

    m = Media(path=path, size=size, duration=duration,
              container=fmt.get("format_long_name") or fmt.get("format_name", "?"),
              video=vids[0] if vids else None, audio=audio, subs=subs)
    v = m.video
    if v is None:
        return m

    m.codec = v.get("codec_name", "?")
    m.profile = v.get("profile", "")
    m.width = int(v.get("width") or 0)
    m.height = int(v.get("height") or 0)
    m.fps = parse_rate(v.get("avg_frame_rate") or "0") or parse_rate(v.get("r_frame_rate") or "0")
    m.pix_fmt = v.get("pix_fmt", "")
    m.bit_depth = int(v.get("bits_per_raw_sample") or 0) or (10 if "10" in m.pix_fmt else 8)
    m.field_order = v.get("field_order", "unknown") or "unknown"
    m.interlaced = m.field_order in ("tt", "bb", "tb", "bt")

    audio_bps = sum(int(a.get("bit_rate") or 0) for a in audio)
    vbps = int(v.get("bit_rate") or 0)
    if not vbps and duration > 0:  # MKV often does not tag per-stream bitrate
        vbps = max(0.0, size * 8 / duration - audio_bps)
    m.bitrate = float(vbps)
    pixel_rate = m.width * m.height * m.fps
    m.bpp = m.bitrate / pixel_rate if pixel_rate else 0.0
    return m


def quality_class(bpp: float) -> str:
    if bpp >= 0.20:
        return "high"
    if bpp >= 0.10:
        return "medium"
    if bpp >= 0.05:
        return "low"
    return "very low"


def start_quality(m: Media, encoder: str) -> int:
    """Heuristic starting point. Higher bits/pixel -> start at higher quality.
    The SSIM check afterwards corrects this if it is not enough."""
    tier = quality_class(m.bpp)
    if encoder == "vaapi":
        return {"high": 70, "medium": 90, "low": 110, "very low": 110}[tier]
    return {"high": 26, "medium": 30, "low": 34, "very low": 34}[tier]


def skip_reason(m: Media) -> str | None:
    if m.video is None:
        return "no video stream"
    if m.codec == "av1":
        return "already AV1"
    if m.pix_fmt not in SUPPORTED_PIX:
        return f"pixel format '{m.pix_fmt or '?'}' not supported (needs 8/10-bit 4:2:0)"
    if m.width <= 0 or m.height <= 0:
        return "unknown video dimensions"
    return None


def print_media(m: Media):
    print(f"  file      : {m.path}")
    print(f"  container : {m.container}, {human(m.size)}, {hms(m.duration)}")
    if m.video is None:
        print("  video     : none")
        return
    print(f"  video     : {m.codec} {m.profile} {m.width}x{m.height} @ {m.fps:.3f} fps, "
          f"{m.pix_fmt} ({m.bit_depth}-bit), field order: {m.field_order}")
    v = m.video
    color = "/".join(v.get(k) or "?" for k in ("color_primaries", "color_transfer", "color_space"))
    print(f"  bitrate   : {m.bitrate / 1e6:.2f} Mb/s, colour {color}")
    print(f"  quality   : {quality_class(m.bpp)} (bits per pixel {m.bpp:.3f})")
    codecs = ", ".join(a.get("codec_name", "?") for a in m.audio) or "none"
    print(f"  audio     : {len(m.audio)} track(s) [{codecs}], subtitles: {len(m.subs)}")


# --------------------------------------------------------------------------- encoding

def build_cmd(m: Media, out: Path, args, q: int) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error",
           "-nostats", "-progress", "pipe:1"]
    filters: list[str] = []
    if m.interlaced:
        filters.append("yadif=mode=send_frame:parity=auto:deint=all")
    if m.width % 2 or m.height % 2:  # AV1 hardware encoders need even dimensions
        filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    if args.encoder == "vaapi":
        cmd += ["-init_hw_device", f"vaapi=va:{args.device}", "-filter_hw_device", "va"]
    if args.sample:
        cmd += ["-t", str(args.sample)]
    cmd += ["-i", str(m.path)]

    if args.encoder == "vaapi":
        upload_fmt = SUPPORTED_PIX[m.pix_fmt][1]
        filters += [f"format={upload_fmt}", "hwupload"]
        cmd += ["-vf", ",".join(filters),
                "-c:v", "av1_vaapi", "-rc_mode", "CQP", "-global_quality", str(q)]
    else:
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-c:v", "libsvtav1", "-crf", str(q), "-preset", str(args.preset)]

    # keyframe every ~10 s (smaller files than the default 12-frame GOP)
    gop = max(1, round(m.fps * 10)) if m.fps else 240
    cmd += ["-g", str(gop)]

    # keep colour metadata (HDR / BT.2020 / BT.709 tags) so playback stays identical
    v = m.video or {}
    for key, flag in (("color_primaries", "-color_primaries"),
                      ("color_transfer", "-color_trc"),
                      ("color_space", "-colorspace"),
                      ("color_range", "-color_range")):
        val = v.get(key)
        if val and val not in ("unknown", "unspecified", "reserved"):
            cmd += [flag, val]

    cmd += ["-map", f"0:{v.get('index', 0)}",
            "-map", "0:a?", "-map", "0:s?",
            "-c:a", "copy", "-c:s", "copy",
            "-map_metadata", "0", "-map_chapters", "0",
            "-f", "matroska", str(out)]
    return cmd


def run_encode(cmd: list[str], total: float) -> tuple[int, str]:
    """Run ffmpeg, show live progress, return (returncode, stderr text)."""
    with tempfile.TemporaryFile("w+") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True)
        pct, speed = 0.0, "?"
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    t = int(line.split("=", 1)[1]) / 1e6
                    if total:
                        pct = min(100.0, t / total * 100)
                except ValueError:
                    pass
            elif line.startswith("speed="):
                speed = line.split("=", 1)[1]
                print(f"\r    encoding {pct:5.1f}%  speed {speed}   ", end="", flush=True)
        rc = proc.wait()
        errf.seek(0)
        err = errf.read()
    print("\r" + " " * 60 + "\r", end="")
    return rc, err


def verify_output(m: Media, tmp: Path, sample: float) -> str | None:
    """Return a problem description, or None if the output looks right."""
    try:
        data = ffprobe(tmp)
    except RuntimeError as e:
        return str(e)
    streams = data.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"]
    if not vids or vids[0].get("codec_name") != "av1":
        return "output has no AV1 video stream"
    n_audio = sum(1 for s in streams if s.get("codec_type") == "audio")
    if n_audio != len(m.audio):
        return f"audio track count changed ({len(m.audio)} -> {n_audio})"
    if not sample:
        dur = float(data.get("format", {}).get("duration") or 0.0)
        if m.duration and abs(dur - m.duration) > 1.0:
            return f"duration mismatch ({m.duration:.2f}s -> {dur:.2f}s)"
    return None


SSIM_RE = re.compile(r"All:([0-9.]+)")


def ssim_check(src: Path, dst: Path, total: float, win: float = 8.0) -> float | None:
    """SSIM (0..1, 1 = identical) on several windows; returns the WORST window."""
    if total <= 0:
        return None
    if total <= 3 * win:
        windows = [(0.0, total)]
    else:
        windows = [(max(0.0, f * total - win / 2), win) for f in (0.2, 0.5, 0.8)]
    flt = ("[0:v:0]format=yuv420p,setpts=PTS-STARTPTS[a];"
           "[1:v:0]format=yuv420p,setpts=PTS-STARTPTS[b];[a][b]ssim")
    scores = []
    for start, length in windows:
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info",
               "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
               "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(dst),
               "-lavfi", flt, "-f", "null", "-"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        found = SSIM_RE.findall(r.stderr)
        if not found:
            return None
        scores.append(float(found[-1]))
    return min(scores)


def improve(encoder: str, q: int) -> int | None:
    """Next quality setting (higher quality). None when already at the limit."""
    if encoder == "vaapi":
        nq = max(VAAPI_MIN_Q, q - 15)
    else:
        nq = max(SVT_MIN_CRF, q - 4)
    return nq if nq != q else None


# --------------------------------------------------------------------------- per file

def output_path(m: Media, args) -> Path:
    name = m.path.stem + (".sample" if args.sample else "") + OUTPUT_SUFFIX
    folder = Path(args.out_dir) if args.out_dir else m.path.parent
    return folder / name


def process(path: Path, args) -> tuple[str, str]:
    m = analyze(path)
    print_media(m)

    reason = skip_reason(m)
    if reason:
        print(f"  -> skipped: {reason}")
        return "skipped", reason

    out = output_path(m, args)
    if out.exists() and not args.overwrite:
        print(f"  -> skipped: {out.name} already exists (use --overwrite)")
        return "skipped", "output already exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + PARTIAL_TAG + out.suffix)

    total = min(args.sample, m.duration) if args.sample else m.duration
    q = args.quality if args.quality is not None else start_quality(m, args.encoder)
    attempt, score = 0, None

    while True:
        attempt += 1
        tmp.unlink(missing_ok=True)
        print(f"  encoder {args.encoder}, quality {q}, attempt {attempt}")
        rc, err = run_encode(build_cmd(m, tmp, args, q), total)
        if rc != 0:
            tmp.unlink(missing_ok=True)
            return "failed", f"ffmpeg exit {rc}: {last_lines(err)}"

        problem = verify_output(m, tmp, args.sample)
        if problem:
            tmp.unlink(missing_ok=True)
            return "failed", problem

        if args.no_verify:
            break

        score = ssim_check(m.path, tmp, total)
        if score is None:
            tmp.unlink(missing_ok=True)
            return "failed", "could not measure SSIM quality"
        print(f"  quality   : SSIM {score:.4f} (target >= {args.ssim_target})")
        if score >= args.ssim_target:
            break

        nq = improve(args.encoder, q)
        if nq is None or attempt >= args.max_tries:
            tmp.unlink(missing_ok=True)
            return "kept original", (f"quality target not reached (best SSIM {score:.4f} "
                                     f"at quality {q}); original left untouched")
        print("  quality too low, re-encoding at higher quality")
        q = nq

    new_size = tmp.stat().st_size
    quality_txt = f"quality {q}" + (f", SSIM {score:.4f}" if score is not None else ", not verified")

    if args.sample:
        os.replace(tmp, out)
        est = new_size / total * m.duration if total else 0
        msg = (f"sample {human(new_size)} for {total:.0f}s, estimated full file "
               f"{human(est)} vs {human(m.size)} original ({quality_txt})")
        return "sample", msg

    if new_size >= m.size:
        tmp.unlink(missing_ok=True)
        return "kept original", (f"AV1 output ({human(new_size)}) is not smaller than the "
                                 f"source ({human(m.size)}) at {quality_txt}")

    os.replace(tmp, out)
    saved = (1 - new_size / m.size) * 100
    msg = f"{human(m.size)} -> {human(new_size)} (-{saved:.1f}%), {quality_txt}"
    if args.delete_original:
        m.path.unlink()
        msg += ", original deleted"
    return "ok", msg


# --------------------------------------------------------------------------- CLI

def collect(inputs: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS and not is_generated(f):
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: not found: {item}", file=sys.stderr)
    return files


def is_generated(f: Path) -> bool:
    name = f.name.lower()
    return name.endswith(OUTPUT_SUFFIX) or PARTIAL_TAG in name


def check_vaapi(device: str):
    if not os.path.exists(device):
        die(f"VAAPI device {device} not found (use --device, or --encoder svt)")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    if "av1_vaapi" not in r.stdout:
        die("this ffmpeg build has no av1_vaapi encoder (use --encoder svt)")


def selftest(args) -> int:
    """Encode a short synthetic clip at 8 and 10 bit to check the hardware path."""
    print("self-test: VAAPI AV1 encode, 8-bit and 10-bit")
    ok = True
    for fmt, label in (("nv12", "8-bit"), ("p010le", "10-bit")):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.mkv"
            cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
                   "-init_hw_device", f"vaapi=va:{args.device}", "-filter_hw_device", "va",
                   "-f", "lavfi", "-i", "testsrc2=duration=2:size=1280x720:rate=30",
                   "-vf", f"format={fmt},hwupload", "-c:v", "av1_vaapi",
                   "-rc_mode", "CQP", "-global_quality", "100", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            good = r.returncode == 0 and out.exists()
            ok &= good
            print(f"  {label}: {'OK' if good else 'FAILED - ' + last_lines(r.stderr)}")
    print("self-test " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Analyze videos and transcode them to AV1 (VAAPI hardware by default).")
    p.add_argument("inputs", nargs="*", help="video files and/or folders")
    p.add_argument("-r", "--recursive", action="store_true", help="scan folders recursively")
    p.add_argument("-o", "--out-dir", help="write outputs here (default: next to each source)")
    p.add_argument("--encoder", choices=["vaapi", "svt"], default="vaapi",
                   help="vaapi = AMD/Intel hardware AV1 (fast); svt = SVT-AV1 software (smaller files, slow)")
    p.add_argument("--device", default="/dev/dri/renderD128", help="VAAPI render device")
    p.add_argument("--quality", type=int,
                   help="starting quality: vaapi global_quality (lower=better, default by bitrate) "
                        "or svt CRF (lower=better)")
    p.add_argument("--ssim-target", type=float, default=0.98,
                   help="minimum SSIM vs original; below this quality is raised (default 0.98)")
    p.add_argument("--max-tries", type=int, default=4, help="encode attempts per file (default 4)")
    p.add_argument("--no-verify", action="store_true", help="skip the SSIM quality check (faster)")
    p.add_argument("--preset", type=int, default=6, help="SVT-AV1 preset 0-13 (default 6)")
    p.add_argument("--sample", type=float, metavar="SECONDS",
                   help="encode only the first N seconds, to calibrate settings")
    p.add_argument("--analyze", action="store_true", help="only analyze, do not encode")
    p.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    p.add_argument("--delete-original", action="store_true",
                   help="delete the source after a verified, smaller AV1 file was written")
    p.add_argument("--selftest", action="store_true", help="check the VAAPI encoder and exit")
    args = p.parse_args(argv)
    if not args.selftest and not args.inputs:
        p.error("no input files or folders given")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        die("ffmpeg and ffprobe must be installed")
    if args.encoder == "vaapi" and not args.analyze:
        check_vaapi(args.device)
    if args.selftest:
        return selftest(args)

    files = collect(args.inputs, args.recursive)
    if not files:
        die("no video files found")

    results = []
    for f in files:
        print(f"\n== {f.name}")
        try:
            if args.analyze:
                m = analyze(f)
                print_media(m)
                reason = skip_reason(m)
                q = start_quality(m, args.encoder)
                print(f"  -> would {'skip: ' + reason if reason else f'encode (start quality {q})'}")
                status, info = ("analyzed", reason or f"start quality {q}")
            else:
                status, info = process(f, args)
        except Exception as e:  # keep the batch going
            status, info = "error", str(e)
        print(f"  [{status}] {info}")
        results.append((f, status, info))

    print("\n== summary")
    saved = 0
    for f, status, info in results:
        print(f"  {status:>14}  {f.name}: {info}")
    for f, status, info in results:
        if status == "ok":
            saved += 1
    print(f"  {saved} of {len(results)} file(s) transcoded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
