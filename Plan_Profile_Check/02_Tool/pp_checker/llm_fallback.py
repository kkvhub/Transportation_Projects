"""
llm_fallback.py — optional Claude API reader for regions OCR can't resolve.

Activates only when ANTHROPIC_API_KEY is set (env var or Streamlit secret).
Sends a high-zoom PNG crop and asks for the table values as JSON.
"""
from __future__ import annotations
import base64
import json
import os
import re

PROMPT = (
    "This image is a curve data table from an Indian road plan drawing. "
    "Return ONLY a JSON object with any of these keys you can read: "
    "hip_ch (metres, e.g. 121252.611), E, N, V, delta_deg (decimal degrees), "
    "R, Ts, Lc, Ls, e (percent). Use null when unreadable."
)


class LLMReader:
    def __init__(self, api_key=None, model="claude-haiku-4-5-20251001"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                self._client = None

    @property
    def available(self):
        return self._client is not None

    def reread_hip(self, pg, grid, prev):
        if not self.available:
            return prev
        import fitz
        pad = 2
        rect = fitz.Rect(grid["x0"] - pad, grid["rows"][0] - pad,
                         grid["x1"] + pad, grid["rows"][-1] + pad)
        png = pg.pixmap(clip=rect, zoom=10).tobytes("png")
        try:
            msg = self._client.messages.create(
                model=self.model, max_tokens=500,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(png).decode()}},
                    {"type": "text", "text": PROMPT}]}])
            text = msg.content[0].text
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                data = json.loads(m.group())
                out = dict(prev)
                for k, v in data.items():
                    if v is not None:
                        out[k] = v
                out["llm_reread"] = True
                return out
        except Exception:
            pass
        return prev
