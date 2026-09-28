"""
Karbotrix Intelligence - AI Video Analyzer & Summarizer
-------------------------------------------------------
Upload an .mp4 (e.g. an Instagram reel), let Gemini analyze it, and download
a UTF-8 / Hindi-safe PDF report.

Run:
    pip install -r requirements.txt
    export GOOGLE_API_KEY="your-key"        # or use .streamlit/secrets.toml / sidebar
    streamlit run app.py
"""

import io
import os
import re
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from fpdf import FPDF

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BRAND = "Karbotrix Intelligence"
DEFAULT_MODEL = "gemini-3.8-flash"  # older 1.5 / 2.0 / 2.5 models are retired or being retired
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
MAX_RETRIES = 4  # retries per model for temporary errors (503/429/500)
POLL_INTERVAL_S = 3
PROCESSING_TIMEOUT_S = 300

PROMPT = (
    "Analyze this video thoroughly. Provide: 1. Core Subject, "
    "2. Detailed Summary of what is said/shown, "
    "3. The main message/intention of the creator, "
    "4. Any readable text on screen. Format nicely with headings."
)

FONT_DIR = Path(__file__).parent / "fonts"
_NOTO = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts"
FONT_FILES = {
    "NotoSans-Regular.ttf": f"{_NOTO}/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
    "NotoSans-Bold.ttf": f"{_NOTO}/NotoSans/hinted/ttf/NotoSans-Bold.ttf",
    "NotoSansDevanagari-Regular.ttf": f"{_NOTO}/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
    "NotoSansDevanagari-Bold.ttf": f"{_NOTO}/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf",
}


# --------------------------------------------------------------------------- #
# Fonts (Unicode / Hindi support)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def ensure_fonts() -> bool:
    """Download Noto fonts once (Latin + Devanagari). Returns True if all present.
    You can also drop the .ttf files into ./fonts yourself for offline use."""
    FONT_DIR.mkdir(exist_ok=True)
    for name, url in FONT_FILES.items():
        target = FONT_DIR / name
        if target.exists() and target.stat().st_size > 10_000:
            continue
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                target.write_bytes(resp.read())
        except Exception:
            target.unlink(missing_ok=True)
            return False
    return True


# --------------------------------------------------------------------------- #
# PDF generation
# --------------------------------------------------------------------------- #
class ReportPDF(FPDF):
    def __init__(self, unicode_ok: bool):
        super().__init__(format="A4")
        self.unicode_ok = unicode_ok
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)
        self.font_main = "Helvetica"

        if unicode_ok:
            self.add_font("NotoSans", "", str(FONT_DIR / "NotoSans-Regular.ttf"))
            self.add_font("NotoSans", "B", str(FONT_DIR / "NotoSans-Bold.ttf"))
            self.add_font("NotoDeva", "", str(FONT_DIR / "NotoSansDevanagari-Regular.ttf"))
            self.add_font("NotoDeva", "B", str(FONT_DIR / "NotoSansDevanagari-Bold.ttf"))
            self.font_main = "NotoSans"
            # Glyphs missing in NotoSans (e.g. Hindi) fall back to Devanagari font
            self.set_fallback_fonts(["NotoDeva"])
            try:  # proper Devanagari conjunct shaping (needs `uharfbuzz`)
                self.set_text_shaping(True)
            except Exception:
                pass

    def header(self):
        self.set_fill_color(20, 30, 60)
        self.rect(0, 0, self.w, 22, style="F")
        self.set_xy(18, 6)
        self.set_text_color(255, 255, 255)
        self.set_font(self.font_main, "B", 15)
        self.cell(0, 10, BRAND, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_y(30)

    def footer(self):
        self.set_y(-12)
        self.set_font(self.font_main, "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"{BRAND}  |  Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _clean_inline(text: str) -> str:
    """Strip markdown emphasis/code markers that the PDF can't style inline."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _latin1_safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(analysis: str, filename: str) -> bytes:
    """Convert Gemini's markdown-ish text into a formatted PDF (bytes)."""
    unicode_ok = ensure_fonts()
    pdf = ReportPDF(unicode_ok)
    pdf.set_title(f"{BRAND} - Video Analysis")
    pdf.set_author(BRAND)
    pdf.add_page()
    fm = pdf.font_main

    def out(text: str) -> str:
        return text if unicode_ok else _latin1_safe(text)

    # Title block
    pdf.set_font(fm, "B", 20)
    pdf.multi_cell(0, 10, out("Video Analysis Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(fm, "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0, 5,
        out(f"Source file: {filename}\nGenerated: {datetime.now():%d %b %Y, %H:%M}"),
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    for raw in analysis.splitlines():
        line = raw.rstrip()
        if not line.strip():
            pdf.ln(2)
            continue
        if re.fullmatch(r"[-*_]{3,}", line.strip()):
            pdf.ln(2)
            continue

        heading = re.match(r"^\s*(#{1,6})\s+(.*)$", line)
        numbered_bold = re.match(r"^\s*\*\*(\d+\.\s.+?)\*\*:?\s*$", line)
        bullet = re.match(r"^(\s*)[-*•]\s+(.*)$", line)

        if heading or numbered_bold:
            text = heading.group(2) if heading else numbered_bold.group(1)
            level = len(heading.group(1)) if heading else 2
            size = {1: 16, 2: 14, 3: 12}.get(level, 11)
            pdf.ln(3)
            pdf.set_font(fm, "B", size)
            pdf.set_text_color(20, 30, 100)
            pdf.multi_cell(0, size * 0.55 + 2, out(_clean_inline(text)),
                           new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        elif bullet:
            indent = min(len(bullet.group(1)) // 2, 3) * 6
            pdf.set_font(fm, "", 10.5)
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(0, 5.8, out("•  " + _clean_inline(bullet.group(2))),
                           new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font(fm, "", 10.5)
            pdf.multi_cell(0, 5.8, out(_clean_inline(line)),
                           new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# --------------------------------------------------------------------------- #
# Gemini (google-genai SDK - supports both AIza... and new AQ.... API keys)
# --------------------------------------------------------------------------- #
def clean_key(api_key: str) -> str:
    """Remove accidental whitespace/quotes from a pasted key."""
    return api_key.strip().strip("\"'").strip()


def make_client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=clean_key(api_key),
        http_options=types.HttpOptions(timeout=600_000),  # milliseconds
    )


def wait_until_active(client, video_file, status):
    """Poll until Gemini finishes processing the uploaded video."""
    started = time.time()
    while video_file.state is not None and video_file.state.name == "PROCESSING":
        if time.time() - started > PROCESSING_TIMEOUT_S:
            raise TimeoutError("Video processing timed out. Try a shorter clip.")
        status.update(label="Gemini is processing the video...")
        time.sleep(POLL_INTERVAL_S)
        video_file = client.files.get(name=video_file.name)
    if video_file.state is not None and video_file.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process this video.")
    return video_file


def _is_transient(msg: str) -> bool:
    return any(k in msg for k in ("503", "unavailable", "429", "resource_exhausted",
                                  "500", "internal", "overloaded", "high demand",
                                  "deadline", "timed out", "timeout"))


def _is_not_found(msg: str) -> bool:
    return "404" in msg or "not_found" in msg or "not found" in msg or "no longer available" in msg


def discover_flash_models(client) -> list:
    """Ask the API which Flash models this key can actually use (newest first)."""
    skip = ("image", "live", "audio", "tts", "embedding", "robotics", "native", "veo")
    found = []
    try:
        for m in client.models.list():
            n = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in n and "generateContent" in actions and not any(x in n for x in skip):
                found.append(n)
    except Exception:
        pass
    return sorted(found, reverse=True)


def _try_model(client, name: str, contents):
    """Return text, or raise. Retries temporary errors with exponential backoff.
    Raises LookupError if the model doesn't exist."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=name, contents=contents)
            if not response.text:
                raise RuntimeError("Gemini returned an empty response (possibly blocked).")
            return response.text
        except Exception as exc:
            msg = str(exc).lower()
            if _is_not_found(msg):
                raise LookupError(str(exc)) from exc
            if not _is_transient(msg):
                raise  # auth errors, bad request, etc. -> don't retry
            if attempt == MAX_RETRIES:
                raise
            wait = min(2 ** attempt * 2, 30)  # 4s, 8s, 16s...
            st.info(f"`{name}` is busy. Retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)


def generate_analysis(client, video_file, model_name: str) -> str:
    """Run the prompt. Retries temporary errors (503 etc.), falls back to other
    models, and finally auto-discovers whichever Flash models your key can use."""
    contents = types.Content(
        role="user",
        parts=[
            types.Part.from_uri(
                file_uri=video_file.uri,
                mime_type=video_file.mime_type or "video/mp4",
            ),
            types.Part.from_text(text=PROMPT),
        ],
    )
    tried, last_exc = [], None

    def run(names):
        nonlocal last_exc
        for name in names:
            if name in tried:
                continue
            tried.append(name)
            try:
                return _try_model(client, name, contents)
            except LookupError as exc:
                last_exc = exc
                st.warning(f"`{name}` is not available anymore; trying another model...")
            except Exception as exc:
                if not _is_transient(str(exc).lower()):
                    raise
                last_exc = exc
                st.warning(f"`{name}` is overloaded; trying another model...")
        return None

    text = run([model_name, *FALLBACK_MODELS])
    if text is None:
        text = run(discover_flash_models(client))
    if text is None:
        raise RuntimeError(
            f"No Gemini model worked (tried: {', '.join(tried)}). "
            f"Type a current model name in the sidebar (see ai.google.dev/gemini-api/docs/models). "
            f"Last error: {last_exc}"
        )
    return text


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if any(k in low for k in ("api key not valid", "access_token_type_unsupported",
                              "401", "unauthenticated", "api_key_invalid")):
        return (
            "Authentication failed. Check that:\n"
            "1. The key is pasted fully (no spaces/quotes).\n"
            "2. You are on the latest SDK: `pip install -U google-genai` "
            "(the old `google-generativeai` package does NOT work with `AQ.` keys).\n"
            "3. The Generative Language API is enabled for the key's project.\n\n"
            f"Details: {msg}"
        )
    return f"Something went wrong: {msg}"


def analyze_video(uploaded, api_key: str, model_name: str) -> str:
    client = make_client(api_key)
    tmp_path = None
    remote_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.status("Analyzing video...", expanded=True) as status:
            status.update(label="Uploading to Gemini...")
            remote_file = client.files.upload(
                file=tmp_path, config=types.UploadFileConfig(mime_type="video/mp4")
            )
            remote_file = wait_until_active(client, remote_file, status)
            status.update(label="Generating analysis...")
            text = generate_analysis(client, remote_file, model_name)
            status.update(label="Analysis complete", state="complete", expanded=False)
        return text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if remote_file is not None:  # also free the file stored on Google's side
            try:
                client.files.delete(name=remote_file.name)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        """
        <style>
        .kb-header {
            background: linear-gradient(120deg, #0f1b3d 0%, #24408f 100%);
            padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        }
        .kb-header h1 { color: #fff; margin: 0; font-size: 2rem; }
        .kb-header p  { color: #c7d2f5; margin: .3rem 0 0; }
        .kb-badge {
            display:inline-block; background:#ffffff22; color:#fff;
            padding:.15rem .7rem; border-radius:999px; font-size:.75rem;
            letter-spacing:.08em; text-transform:uppercase; margin-bottom:.6rem;
        }
        </style>
        <div class="kb-header">
            <span class="kb-badge">Automation</span>
            <h1>Karbotrix Intelligence</h1>
            <p>AI Video Analyzer &amp; Summarizer &mdash; turn any reel into a detailed PDF report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_api_key(sidebar_value: str) -> str:
    if sidebar_value:
        return sidebar_value
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")


def make_zip(items) -> bytes:
    buf = io.BytesIO()
    used = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in items:
            name = f"{r['name']}_karbotrix_report.pdf"
            n = 1
            while name in used:  # avoid duplicate names inside the ZIP
                n += 1
                name = f"{r['name']}_{n}_karbotrix_report.pdf"
            used.add(name)
            zf.writestr(name, r["pdf"])
    return buf.getvalue()


def main():
    st.set_page_config(page_title=f"{BRAND} | Video Analyzer", page_icon="🎬", layout="centered")
    inject_css()

    with st.sidebar:
        st.header("Settings")
        key_input = st.text_input("Gemini API key", type="password",
                                  help="Works with AIza... and AQ.... keys. Or set GOOGLE_API_KEY / st.secrets.")
        model_name = st.text_input("Model", value=DEFAULT_MODEL)
        st.caption("Supports Hindi and other UTF-8 content in the PDF report.")

    with st.spinner("Preparing PDF fonts..."):
        fonts_ok = ensure_fonts()
    if not fonts_ok:
        st.sidebar.warning(
            "Could not download Hindi fonts (no internet?). PDFs will be Latin-only. "
            "Place the .ttf files from FONT_FILES into the ./fonts folder."
        )

    uploaded_files = st.file_uploader(
        "Upload one or more videos (.mp4)", type=["mp4"], accept_multiple_files=True
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} video(s) selected")
        with st.expander("Preview videos"):
            for f in uploaded_files:
                st.caption(f.name)
                st.video(f)

    c1, c2 = st.columns([3, 1])
    analyze = c1.button("Analyze Videos", type="primary", disabled=not uploaded_files,
                        use_container_width=True)
    if c2.button("Clear results", use_container_width=True):
        st.session_state.pop("results", None)
        st.rerun()

    if analyze:
        api_key = get_api_key(key_input)
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar or via GOOGLE_API_KEY.")
            st.stop()

        model = model_name.strip() or DEFAULT_MODEL
        total = len(uploaded_files)
        results = []
        progress = st.progress(0.0, text="Starting...")
        for i, f in enumerate(uploaded_files, start=1):
            progress.progress((i - 1) / total, text=f"Processing {i}/{total}: {f.name}")
            try:
                analysis = analyze_video(f, api_key, model)
                results.append({
                    "ok": True,
                    "name": Path(f.name).stem,
                    "analysis": analysis,
                    "pdf": build_pdf(analysis, f.name),
                })
            except Exception as exc:  # one failed video must not stop the rest
                results.append({"ok": False, "name": f.name, "error": friendly_error(exc)})
        progress.progress(1.0, text="All videos processed")
        st.session_state["results"] = results

    results = st.session_state.get("results")
    if results:
        good = [r for r in results if r["ok"]]
        if good:
            st.success(f"{len(good)} of {len(results)} report(s) ready.")
        else:
            st.warning("No reports were generated. See the errors below and try again.")

        if len(good) > 1:
            st.download_button(
                f"Download all {len(good)} reports (ZIP)",
                data=make_zip(good),
                file_name="karbotrix_reports.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                key="dl_zip",
            )

        for i, r in enumerate(results):
            if r["ok"]:
                st.download_button(
                    f"Download PDF: {r['name']}",
                    data=r["pdf"],
                    file_name=f"{r['name']}_karbotrix_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_{i}",
                )
                with st.expander(f"Preview: {r['name']}"):
                    st.markdown(r["analysis"])
            else:
                st.error(f"{r['name']}: {r['error']}")

    st.divider()
    st.caption(f"© {datetime.now().year} {BRAND}")


if __name__ == "__main__":
    main()