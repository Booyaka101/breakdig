"""Encode float PCM to MP3 through ffmpeg's libmp3lame, then tag with mutagen."""

import subprocess
from pathlib import Path

import numpy as np
from mutagen.id3 import COMM, ID3, TALB, TLEN, TIT2, TPE1

from . import audio

# Hide the console window ffmpeg would otherwise flash when run from the web UI on Windows.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BITRATE = "192k"


class Encoder:
    """Streams (frames, channels) float32 blocks into libmp3lame at a fixed bitrate.

    The blocks go down a pipe as raw little-endian float PCM, so a setlist-sized
    drill never sits in memory twice."""

    def __init__(self, path: str | Path, channels: int, sample_rate: int, bitrate: str = BITRATE):
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le",
               "-ar", str(sample_rate), "-ac", str(channels), "-i", "pipe:0",
               "-c:a", "libmp3lame", "-b:a", bitrate, "-map_metadata", "-1",
               # The muxer is forced because the file is written to a .part path first.
               "-f", "mp3", str(path)]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                         creationflags=NO_WINDOW)
        except FileNotFoundError:
            raise audio.DecodeError(audio.FFMPEG_HINT) from None
        self.path, self.sample_rate = path, sample_rate

    def write(self, block: np.ndarray) -> None:
        flat = np.ascontiguousarray(block, dtype=np.float32)
        # One second per write, so a long pass never turns into one huge bytes object on
        # top of the array it came from.
        step = max(1, self.sample_rate)
        try:
            for i in range(0, len(flat), step):
                self.proc.stdin.write(flat[i: i + step].tobytes())
        except BrokenPipeError:
            stderr = self.proc.stderr.read()
            self.proc.wait()
            raise audio.DecodeError(audio._last_line(stderr)) from None

    def close(self) -> None:
        self.proc.stdin.close()
        stderr = self.proc.stderr.read()
        if self.proc.wait() != 0:
            raise audio.DecodeError(audio._last_line(stderr))
        self.proc.stderr.close()


def add_id3(path: str | Path, artist: str, title: str, album: str, seconds: float,
            comment: str) -> None:
    """ID3v2.3 tags: who and what, the length in ms, and where the drill came from."""
    tags = ID3()
    tags.add(TPE1(encoding=3, text=artist))
    tags.add(TIT2(encoding=3, text=title))
    if album:
        tags.add(TALB(encoding=3, text=album))
    tags.add(TLEN(encoding=3, text=str(round(seconds * 1000))))
    tags.add(COMM(encoding=3, lang="eng", desc="breakdig", text=comment))
    tags.save(path, v2_version=3)
