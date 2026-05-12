from __future__ import annotations

import subprocess
from pathlib import Path


def transcode_to_browser_mp4(source: str | Path, target: str | Path | None = None) -> str:
    """Convert a video to browser-friendly H.264 MP4 when imageio-ffmpeg is available."""
    src = Path(source)
    dst = Path(target) if target else src.with_name(f"{src.stem}_h264.mp4")
    try:
        import imageio_ffmpeg
    except Exception:
        return str(src)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(dst)
    except Exception:
        return str(src)
