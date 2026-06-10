"""
VLMClient — LocateAnything-3B qua vLLM (OpenAI-compatible API). CHẠY CLOUD.

Transport là CODE THẬT (HTTP /v1/chat/completions, ảnh base64). Phần cần tinh
chỉnh theo model thật = TEMPLATE PROMPT + parse output (LocateAnything có thể
trả format riêng). Đã ràng buộc model trả JSON để parse ổn định.

2 vai (theo kiến trúc 2-tier):
  locate()  : open-vocab localize (M1 xe zero-shot, M2 cold-start fire/smoke).
  verify()  : xác minh ngữ nghĩa (cắt báo giả mà debounce không cắt được).

Chỉ dùng stdlib (urllib) + cv2 để encode ảnh → không thêm dependency.
"""

from __future__ import annotations

import base64
import json
from urllib import request as _req


class VLMClient:
    def __init__(
        self,
        endpoint: str | None,
        model: str = "locate-anything-3b",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = (endpoint or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.endpoint)

    # --- transport (THẬT) -------------------------------------------------- #
    @staticmethod
    def _to_data_url(image_bgr) -> str:
        import cv2  # noqa: PLC0415

        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise ValueError("Không encode được JPEG")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _chat(self, image_bgr, prompt: str) -> str:
        url = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._to_data_url(image_bgr)}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = _req.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with _req.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return out["choices"][0]["message"]["content"]

    # --- 2 vai ------------------------------------------------------------- #
    def locate(self, image_bgr, classes: list[str]) -> list[dict]:
        """Open-vocab localize -> [{label, box:[x1,y1,x2,y2], score}] (pixel)."""
        prompt = (
            "Locate these objects in the image: " + ", ".join(classes) + ".\n"
            'Return ONLY a JSON list like '
            '[{"label": "fire", "box": [x1,y1,x2,y2], "score": 0.0}]. '
            "Box in pixels (top-left origin). Empty list [] if none."
        )
        data = _extract_json(self._chat(image_bgr, prompt), "[", "]")
        if not isinstance(data, list):
            return []
        out = []
        for d in data:
            try:
                out.append(
                    {
                        "label": str(d["label"]),
                        "box": [float(v) for v in d["box"]],
                        "score": float(d.get("score", 1.0)),
                    }
                )
            except Exception:
                continue
        return out

    def verify(self, image_bgr, question: str) -> dict:
        """Xác minh ngữ nghĩa -> {is_true: bool, reason: str}."""
        prompt = question.strip() + '\nAnswer ONLY JSON: {"is_true": true/false, "reason": "..."}.'
        obj = _extract_json(self._chat(image_bgr, prompt), "{", "}")
        if not isinstance(obj, dict):
            return {"is_true": False, "reason": "parse_error"}
        return {"is_true": bool(obj.get("is_true", False)), "reason": str(obj.get("reason", ""))}


def _extract_json(text: str, opener: str, closer: str):
    """Bóc khối JSON đầu tiên trong text (model hay kèm lời dẫn)."""
    s, e = text.find(opener), text.rfind(closer)
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except Exception:
        return None
