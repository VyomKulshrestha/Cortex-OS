"""Screen Understanding — OCR, element detection, screen analysis.

Combines screenshot capture with OCR engines per platform:
  - Windows: WinRT OCR (built-in Win10+), tesseract CLI, EasyOCR
  - macOS:   Vision framework (built-in), tesseract CLI, EasyOCR
  - Linux:   tesseract CLI, EasyOCR
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

logger = logging.getLogger("pilot.system.vision")

# ──────────────────────────────────────────────────────────────────────
#  Screenshot capture
# ──────────────────────────────────────────────────────────────────────

async def _capture_screenshot_bytes(region: tuple[int, int, int, int] | None = None) -> bytes:
    """Capture screenshot and return PNG bytes."""
    try:
        from io import BytesIO

        import pyautogui

        img = pyautogui.screenshot(region=region)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.debug(f"pyautogui screenshot failed ({e}), falling back to system command")
        # Fallback: capture via system command and read file
        tmp = os.path.join(tempfile.gettempdir(), f"pilot_screen_{os.getpid()}.png")
        from pilot.system.screen import screenshot

        await screenshot(tmp)
        data = Path(tmp).read_bytes()
        os.unlink(tmp)
        return data


# ──────────────────────────────────────────────────────────────────────
#  Main OCR entry point
# ──────────────────────────────────────────────────────────────────────

async def screen_ocr(
    region: tuple[int, int, int, int] | None = None,
    language: str = "eng",
) -> str:
    """Extract ALL text from the screen (or a region) using OCR.

    region: (left, top, width, height) or None for full screen.
    Returns all detected text.

    Tries platform-native OCR first, then EasyOCR, then Tesseract.
    """
    img_bytes = await _capture_screenshot_bytes(region)
    errors: list[str] = []

    # ── Platform-native OCR ──────────────────────────────────────────
    if CURRENT_PLATFORM == Platform.WINDOWS:
        try:
            result = await _ocr_windows_native(img_bytes)
            if result and result.strip():
                logger.info("OCR succeeded via Windows native (WinRT)")
                return result
        except Exception as e:
            errors.append(f"Windows native OCR: {e}")
            logger.debug("Windows native OCR failed: %s", e)

    elif CURRENT_PLATFORM == Platform.MACOS:
        try:
            result = await _ocr_macos_native(img_bytes)
            if result and result.strip():
                logger.info("OCR succeeded via macOS Vision framework")
                return result
        except Exception as e:
            errors.append(f"macOS Vision OCR: {e}")
            logger.debug("macOS Vision OCR failed: %s", e)

    # ── Tesseract CLI (cross-platform, no Python wrapper needed) ─────
    try:
        result = await _ocr_tesseract_cli(img_bytes, language)
        if result and result.strip():
            logger.info("OCR succeeded via tesseract CLI")
            return result
    except Exception as e:
        errors.append(f"Tesseract CLI: {e}")
        logger.debug("Tesseract CLI failed: %s", e)

    # ── EasyOCR (GPU accelerated) ────────────────────────────────────
    try:
        result = await _ocr_easyocr(img_bytes, language)
        if result and result.strip():
            logger.info("OCR succeeded via EasyOCR")
            return result
    except ImportError:
        errors.append("EasyOCR: not installed")
    except Exception as e:
        errors.append(f"EasyOCR: {e}")
        logger.debug("EasyOCR failed: %s", e)

    # ── pytesseract Python wrapper ───────────────────────────────────
    try:
        result = await _ocr_pytesseract(img_bytes, language)
        if result and result.strip():
            logger.info("OCR succeeded via pytesseract")
            return result
    except ImportError:
        errors.append("pytesseract: not installed")
    except Exception as e:
        errors.append(f"pytesseract: {e}")
        logger.debug("pytesseract failed: %s", e)

    # ── All engines failed ───────────────────────────────────────────
    install_hint = _get_install_hint()
    raise RuntimeError(
        f"No OCR engine could extract text.\n"
        f"Tried: {'; '.join(errors)}\n"
        f"To fix: {install_hint}"
    )


def _get_install_hint() -> str:
    """Return platform-specific install instructions."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return (
            "Windows 10/11 native OCR should work automatically. "
            "If it fails, install Tesseract: "
            "winget install UB-Mannheim.TesseractOCR  OR  "
            "pip install easyocr"
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        return (
            "macOS Vision framework should work automatically (10.15+). "
            "If it fails: brew install tesseract  OR  pip install easyocr"
        )
    else:
        return (
            "Install tesseract: sudo apt install tesseract-ocr  OR  "
            "sudo dnf install tesseract  OR  pip install easyocr"
        )


# ──────────────────────────────────────────────────────────────────────
#  Windows native OCR (WinRT)
# ──────────────────────────────────────────────────────────────────────

async def _ocr_windows_native(img_bytes: bytes) -> str:
    """Use Windows 10/11 built-in OCR via PowerShell WinRT APIs.
    
    Writes the PS script to a temp file to avoid escaping issues.
    """
    tmp_img = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.png")
    tmp_ps1 = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.ps1")
    Path(tmp_img).write_bytes(img_bytes)

    # PowerShell script that uses WinRT OCR
    ps_script = f"""
try {{
    Add-Type -AssemblyName System.Runtime.WindowsRuntime

    # Load WinRT types
    [void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
    [void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime]
    [void][Windows.Storage.StorageFile, Windows.Foundation, ContentType=WindowsRuntime]

    # Helper to await WinRT async operations
    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{
        $_.Name -eq 'AsTask' -and
        $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    }})[0]

    Function Await($WinRtTask, $ResultType) {{
        $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
        $netTask = $asTask.Invoke($null, @($WinRtTask))
        $netTask.Wait(-1) | Out-Null
        $netTask.Result
    }}

    # Open image file
    $imgPath = '{tmp_img.replace(chr(92), chr(92) + chr(92))}'
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imgPath)) ([Windows.Storage.StorageFile])

    # Open stream and create decoder
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

    # Create OCR engine and recognize
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if ($engine -eq $null) {{
        Write-Error "OCR engine could not be created"
        exit 1
    }}
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

    # Output text
    Write-Output $result.Text

    # Cleanup
    $stream.Dispose()
}} catch {{
    Write-Error $_.Exception.Message
    exit 1
}}
"""
    Path(tmp_ps1).write_text(ps_script, encoding="utf-8")

    try:
        code, out, err = await run_command(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", tmp_ps1],
            timeout=15,
        )
        if code == 0 and out.strip():
            return out.strip()
        # If exit code is 0 but no output, might be empty screen
        if code == 0:
            return ""
        raise RuntimeError(f"Windows OCR failed (exit {code}): {err.strip()}")
    finally:
        for f in (tmp_img, tmp_ps1):
            try:
                os.unlink(f)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
#  macOS native OCR (Vision framework)
# ──────────────────────────────────────────────────────────────────────

async def _ocr_macos_native(img_bytes: bytes) -> str:
    """Use macOS Vision framework for OCR (macOS 10.15+)."""
    tmp_img = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.png")
    Path(tmp_img).write_bytes(img_bytes)

    # Swift script that uses Vision framework
    swift_script = f"""
import Vision
import AppKit

let imgPath = "{tmp_img}"
guard let image = NSImage(contentsOfFile: imgPath),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {{
    fputs("Failed to load image", stderr)
    exit(1)
}}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try handler.perform([request])

guard let observations = request.results else {{
    exit(0)
}}

for observation in observations {{
    if let candidate = observation.topCandidates(1).first {{
        print(candidate.string)
    }}
}}
"""

    tmp_swift = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.swift")
    Path(tmp_swift).write_text(swift_script)

    try:
        code, out, err = await run_command(
            ["swift", tmp_swift],
            timeout=20,
        )
        if code == 0:
            return out.strip()
        raise RuntimeError(f"macOS Vision OCR failed: {err.strip()}")
    finally:
        for f in (tmp_img, tmp_swift):
            try:
                os.unlink(f)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
#  Tesseract CLI (cross-platform, no Python wrapper)
# ──────────────────────────────────────────────────────────────────────

async def _ocr_tesseract_cli(img_bytes: bytes, language: str) -> str:
    """Use tesseract binary directly without pytesseract wrapper."""
    # Check if tesseract is installed
    tesseract_bin = shutil.which("tesseract")
    if not tesseract_bin:
        raise RuntimeError("tesseract binary not found in PATH")

    tmp_img = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.png")
    tmp_out = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}_out")
    Path(tmp_img).write_bytes(img_bytes)

    try:
        code, out, err = await run_command(
            [tesseract_bin, tmp_img, tmp_out, "-l", language],
            timeout=30,
        )
        result_file = Path(f"{tmp_out}.txt")
        if result_file.exists():
            text = result_file.read_text(encoding="utf-8", errors="replace")
            return text.strip()
        if code != 0:
            raise RuntimeError(f"tesseract failed (exit {code}): {err.strip()}")
        return ""
    finally:
        for f in (tmp_img, f"{tmp_out}.txt"):
            try:
                os.unlink(f)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
#  EasyOCR
# ──────────────────────────────────────────────────────────────────────

async def _ocr_easyocr(img_bytes: bytes, language: str) -> str:
    from io import BytesIO

    import easyocr
    import numpy as np
    from PIL import Image

    # Map common language codes to easyocr format
    lang_map = {
        "eng": "en", "fra": "fr", "deu": "de", "spa": "es", "ita": "it",
        "por": "pt", "rus": "ru", "jpn": "ja", "kor": "ko", "chi_sim": "ch_sim",
    }
    lang_code = lang_map.get(language, language[:2] if len(language) >= 2 else "en")

    def _do():
        cache_key = f"_easyocr_reader_{lang_code}"
        reader = getattr(_ocr_easyocr, cache_key, None)
        if reader is None:
            try:
                reader = easyocr.Reader([lang_code], gpu=True, verbose=False)
            except Exception:
                reader = easyocr.Reader([lang_code], gpu=False, verbose=False)
            setattr(_ocr_easyocr, cache_key, reader)
        img = Image.open(BytesIO(img_bytes))
        results = reader.readtext(np.array(img))
        lines = [text for (_, text, conf) in results if conf > 0.3]
        return "\n".join(lines)

    return await asyncio.to_thread(_do)


# ──────────────────────────────────────────────────────────────────────
#  pytesseract (Python wrapper)
# ──────────────────────────────────────────────────────────────────────

async def _ocr_pytesseract(img_bytes: bytes, language: str) -> str:
    from io import BytesIO

    import pytesseract
    from PIL import Image

    def _do():
        img = Image.open(BytesIO(img_bytes))
        return pytesseract.image_to_string(img, lang=language)

    return await asyncio.to_thread(_do)


# ──────────────────────────────────────────────────────────────────────
#  screen_find_text
# ──────────────────────────────────────────────────────────────────────

async def screen_find_text(
    target_text: str,
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Find specific text on screen and return its approximate location."""
    img_bytes = await _capture_screenshot_bytes(region)

    try:
        from io import BytesIO

        import easyocr
        import numpy as np
        from PIL import Image

        def _do():
            cache_key = "_easyocr_reader_en"
            reader = getattr(_ocr_easyocr, cache_key, None)
            if reader is None:
                try:
                    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                except Exception:
                    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                setattr(_ocr_easyocr, cache_key, reader)
            img = Image.open(BytesIO(img_bytes))
            results = reader.readtext(np.array(img))
            matches = []
            for bbox, text, conf in results:
                if target_text.lower() in text.lower():
                    cx = int(sum(p[0] for p in bbox) / 4)
                    cy = int(sum(p[1] for p in bbox) / 4)
                    matches.append({
                        "text": text,
                        "center": (cx, cy),
                        "confidence": round(conf, 3),
                        "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                    })
            return matches

        matches = await asyncio.to_thread(_do)
        if not matches:
            return f"Text '{target_text}' not found on screen"
        return json.dumps({"matches": matches, "count": len(matches)}, indent=2)

    except ImportError:
        pass

    # Fallback: use OCR and search in text
    try:
        all_text = await screen_ocr(region)
        if target_text.lower() in all_text.lower():
            return json.dumps({
                "found": True,
                "text": target_text,
                "note": "Text found in OCR output but exact coordinates unavailable without easyocr",
                "context": all_text[:500],
            }, indent=2)
        return f"Text '{target_text}' not found on screen"
    except Exception as e:
        return f"Text search failed: {e}"


# ──────────────────────────────────────────────────────────────────────
#  screen_analyze (vision LLM)
# ──────────────────────────────────────────────────────────────────────

async def screen_analyze(
    prompt: str = "Describe what you see on the screen",
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Analyze the screen using a vision-capable LLM."""
    img_bytes = await _capture_screenshot_bytes(region)
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    # Try Ollama with vision model
    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            for model in ["llava:7b", "llava", "bakllava", "moondream"]:
                try:
                    resp = await client.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "images": [b64_image],
                            "stream": False,
                        },
                    )
                    if resp.status_code == 200:
                        return resp.json().get("response", "No response")
                except Exception:
                    continue

        # Fallback: just do OCR and describe
        ocr_text = await screen_ocr(region)
        return f"[Vision model not available — falling back to OCR]\nScreen text content:\n{ocr_text[:2000]}"
    except Exception as e:
        return f"Screen analysis failed: {e}"


# ──────────────────────────────────────────────────────────────────────
#  screen_element_map
# ──────────────────────────────────────────────────────────────────────

async def screen_element_map(
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Create a map of interactive elements on screen."""
    img_bytes = await _capture_screenshot_bytes(region)

    try:
        from io import BytesIO

        import easyocr
        import numpy as np
        from PIL import Image

        def _do():
            cache_key = "_easyocr_reader_en"
            reader = getattr(_ocr_easyocr, cache_key, None)
            if reader is None:
                try:
                    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                except Exception:
                    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                setattr(_ocr_easyocr, cache_key, reader)
            img = Image.open(BytesIO(img_bytes))
            results = reader.readtext(np.array(img))
            elements = []
            for i, (bbox, text, conf) in enumerate(results):
                if conf < 0.3 or not text.strip():
                    continue
                cx = int(sum(p[0] for p in bbox) / 4)
                cy = int(sum(p[1] for p in bbox) / 4)
                w = int(max(p[0] for p in bbox) - min(p[0] for p in bbox))
                h = int(max(p[1] for p in bbox) - min(p[1] for p in bbox))

                elem_type = "text"
                text_lower = text.lower().strip()
                if text_lower in (
                    "ok", "cancel", "save", "close", "yes", "no", "apply",
                    "submit", "next", "back", "done", "open", "delete",
                    "remove", "install", "run",
                ):
                    elem_type = "button"
                elif w > 100 and h < 30:
                    elem_type = "label"
                elif text_lower.startswith("http") or text_lower.startswith("www"):
                    elem_type = "link"

                elements.append({
                    "id": i,
                    "type": elem_type,
                    "text": text,
                    "center": {"x": cx, "y": cy},
                    "size": {"w": w, "h": h},
                    "confidence": round(conf, 3),
                })
            return elements

        elements = await asyncio.to_thread(_do)
        return json.dumps(
            {
                "elements": elements,
                "count": len(elements),
                "note": "Use mouse_click with center coordinates to interact",
            },
            indent=2,
        )

    except ImportError:
        return "Install easyocr for element detection: pip install easyocr"
