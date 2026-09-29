"""ffmpeg/ffprobe wrappers. Every decode in breakdig goes through here so the
index and the export see exactly the same timeline for lossy files."""

import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

FFMPEG_HINT = (
    "ffmpeg was not found on PATH. Install it and open a new terminal:\n"
    "  winget install Gyan.FFmpeg        (Windows)\n"
    "  brew install ffmpeg               (macOS)\n"
    "  sudo apt install ffmpeg           (Debian/Ubuntu)"
)

# Hide the console window ffmpeg would otherwise flash when run from the web UI on Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class DecodeError(Exception):
    pass


@dataclass
class Probe:
    duration: float
    sample_rate: int
    channels: int
    bits: int  # 0 when the codec has no fixed bit depth (mp3, aac, vorbis)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    except FileNotFoundError:
        raise DecodeError(FFMPEG_HINT) from None


def _last_line(stderr: bytes) -> str:
    lines = [ln.strip() for ln in stderr.decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-1] if lines else "unknown ffmpeg error"


def probe(path: str) -> Probe:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
              "stream=sample_rate,channels,bits_per_raw_sample,bits_per_sample:format=duration",
              "-of", "json", path])
    if r.returncode != 0:
        raise DecodeError(_last_line(r.stderr))
    info = json.loads(r.stdout or b"{}")
    streams = info.get("streams") or []
    if not streams:
        raise DecodeError("no audio stream")
    s = streams[0]
    bits = 0
    for k in ("bits_per_raw_sample", "bits_per_sample"):
        try:
            bits = bits or int(s.get(k) or 0)
        except ValueError:
            pass
    try:
        duration = float(info.get("format", {}).get("duration") or 0.0)
    except ValueError:
        duration = 0.0
    return Probe(duration, int(s.get("sample_rate") or 0), int(s.get("channels") or 0), bits)


def decode(path: str, start: float | None = None, end: float | None = None,
           info: Probe | None = None) -> np.ndarray:
    """Decode at the source rate and layout to float32 of shape (frames, channels).

    start and end trim the decoded stream by sample index instead of seeking the
    input. With -ss, the first ~90 ms of an MP3 come out as decoder warm-up and AAC
    no longer lines up with the full decode that bar times were measured on;
    trimming by timestamp is still a few ms off on Vorbis."""
    p = info or probe(path)
    sr, ch = p.sample_rate, p.channels
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn"]
    trim = []
    if start is not None:
        trim.append(f"start_sample={round(start * sr)}")
    if end is not None:
        trim.append(f"end_sample={round(end * sr)}")
    if trim:
        cmd += ["-af", f"atrim={':'.join(trim)},asetpts=PTS-STARTPTS"]
    cmd += ["-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-c:a", "pcm_f32le", "-"]
    r = _run(cmd)
    if r.returncode != 0:
        raise DecodeError(_last_line(r.stderr))
    x = np.frombuffer(r.stdout, dtype="<f4")
    if x.size == 0:
        raise DecodeError("decoded to zero samples")
    return x[: x.size // ch * ch].reshape(-1, ch)


def decode_to_wav(path: str, out_wav: str, sample_rate: int, gain: float) -> None:
    """Decode to a stereo float32 WAV at a fixed rate, scaled by gain."""
    r = _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", path, "-vn", "-ac", "2",
              "-ar", str(sample_rate), "-af", f"volume={gain}", "-c:a", "pcm_f32le", "-rf64", "auto", out_wav])
    if r.returncode != 0:
        raise DecodeError(_last_line(r.stderr))
