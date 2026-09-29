"""Synthesise the worked-example track: 16 bars of 4/4 at 120 BPM, 44.1 kHz.

Drums play throughout: kick on beats 1 and 3, snare on 2 and 4, hats on every
eighth, and a crash on beat 1 of every bar so the downbeat is unambiguous.
A sine bass line and a triad pad play in bars 1-4 and 9-16 and are silent in
bars 5-8, so the only drums-alone section is bars 5-8 (8.000 s to 16.000 s).

Run directly to write fixture.wav plus its true stems into a folder:
    python tests/fixtures/make_fixture.py out_dir
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120
BARS = 16
BEAT = 60 / BPM
BAR = 4 * BEAT
LENGTH = BARS * BAR
HARMONY_BARS = [*range(0, 4), *range(8, 16)]  # 0-based bars where bass and pad play


def _env(n, decay):
    return np.exp(-np.arange(n) / (decay * SR))


def _place(buf, sound, t):
    i = int(round(t * SR))
    n = min(len(sound), len(buf) - i)
    buf[i: i + n] += sound[:n]


def _highpass(x, alpha=0.95):
    y = np.empty_like(x)
    prev_x = prev_y = 0.0
    for i, v in enumerate(x):
        prev_y = alpha * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    return y


def drums(rng):
    out = np.zeros(int(LENGTH * SR))
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    freq = 50 + 100 * np.exp(-t * 30)
    kick = np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(n, 0.12) * 0.9
    n = int(0.25 * SR)
    snare = (rng.standard_normal(n) * 0.5 + np.sin(2 * np.pi * 190 * np.arange(n) / SR) * 0.5) * _env(n, 0.06) * 0.6
    n = int(0.08 * SR)
    hat = _highpass(rng.standard_normal(n)) * _env(n, 0.012) * 0.25
    n = int(0.5 * SR)
    crash = _highpass(rng.standard_normal(n), 0.9) * _env(n, 0.12) * 0.15
    for bar in range(BARS):
        t0 = bar * BAR
        _place(out, crash, t0)
        for beat in range(4):
            _place(out, kick if beat in (0, 2) else snare, t0 + beat * BEAT)
        for eighth in range(8):
            _place(out, hat, t0 + eighth * BEAT / 2)
    return out


def _fade(n, ms=10):
    f = np.ones(n)
    k = min(int(ms / 1000 * SR), n // 2)
    f[:k] = np.linspace(0, 1, k)
    f[n - k:] = np.linspace(1, 0, k)
    return f


def bass():
    out = np.zeros(int(LENGTH * SR))
    line = [55.0, 55.0, 65.41, 73.42]  # A1 A1 C2 D2, one note per beat
    n = int(BEAT * SR)
    t = np.arange(n) / SR
    for bar in HARMONY_BARS:
        for beat, f in enumerate(line):
            note = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
            _place(out, note * _fade(n) * 0.45, bar * BAR + beat * BEAT)
    return out


def pad():
    out = np.zeros(int(LENGTH * SR))
    n = int(BAR * SR)
    t = np.arange(n) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63)) / 3  # A major triad
    for bar in HARMONY_BARS:
        _place(out, chord * _fade(n, 20) * 0.35, bar * BAR)
    return out


def make(out_dir) -> dict[str, Path]:
    """Write fixture.wav and the true stems; return their paths keyed mix/drums/bass/vocals/other."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    stems = {"drums": drums(rng), "bass": bass(), "vocals": np.zeros(int(LENGTH * SR)), "other": pad()}
    mix = sum(stems.values())
    peak = np.abs(mix).max()
    scale = 0.89 / peak
    paths = {}
    for name, x in {"mix": mix, **stems}.items():
        p = out_dir / ("fixture.wav" if name == "mix" else f"{name}.wav")
        sf.write(p, (x * scale).astype(np.float32), SR, subtype="PCM_16" if name == "mix" else "FLOAT")
        paths[name] = p
    return paths


if __name__ == "__main__":
    for k, v in make(sys.argv[1] if len(sys.argv) > 1 else ".").items():
        print(f"{k:<7} {v}")
