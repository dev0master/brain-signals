"""تنزيل ملف مع إعادة المحاولة — الاتصال بخوادم البيانات ينقطع أحياناً."""
import sys
import time
import urllib.request
from pathlib import Path


def download_file(url: str, dest: Path, attempts: int = 4, quiet: bool = False) -> None:
    for attempt in range(1, attempts + 1):
        try:
            _download_once(url, dest, quiet)
            return
        except OSError as exc:  # انقطاع أو مهلة
            if attempt == attempts:
                raise
            wait = 5 * attempt
            print(f"\n  انقطع الاتصال ({exc}). محاولة {attempt + 1}/{attempts} بعد {wait} ثوانٍ…")
            time.sleep(wait)


def _download_once(url: str, dest: Path, quiet: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 18):
            f.write(chunk)
            done += len(chunk)
            if total and not quiet:
                sys.stdout.write(f"\r  {dest.name}: {done / 1e6:5.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    tmp.replace(dest)  # الاسم النهائي فقط بعد اكتمال التنزيل
    if not quiet:
        sys.stdout.write("\n")
