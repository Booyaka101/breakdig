"""Build dist/breakdig-<version>-windows.zip from the wheel in dist/.

Run `python -m build` first. The zip holds the wheel, install.bat, breakdig.bat,
README.md and LICENSE; install.bat makes a venv next to itself.
"""

import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from breakdig import __version__  # noqa: E402

dist = root / "dist"
wheel = dist / f"breakdig-{__version__}-py3-none-any.whl"
if not wheel.exists():
    sys.exit(f"{wheel.name} not found; run: python -m build")

name = f"breakdig-{__version__}-windows"
out = dist / f"{name}.zip"
files = [wheel, root / "packaging/windows/install.bat", root / "packaging/windows/breakdig.bat",
         root / "README.md", root / "LICENSE"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        data = f.read_bytes()
        if f.suffix == ".bat":
            data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        z.writestr(f"{name}/{f.name}", data)
print(out)
