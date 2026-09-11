"""
Giao diện web cho streaming_chatbot / run_assistant trong template.py.

Đây KHÔNG phải bài nộp của lab — chỉ là phần mở rộng để trải nghiệm chatbot
streaming trên trình duyệt thay vì terminal. Toàn bộ logic gọi API, retry,
đếm token và tính chi phí đều tái sử dụng trực tiếp từ template.py.

Chạy:
    pip install -r requirements-web.txt
    python web_chat.py
Rồi mở http://127.0.0.1:5000 trên trình duyệt.
"""

import os
import secrets
import socket

from flask import Flask, Response, jsonify, render_template, request, session

from template import OPENAI_MODEL, count_tokens, estimate_cost, retry_with_backoff

DEFAULT_PERSONA = (
    "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# session_id -> {"history": [...], "num_turns": int, "total_tokens": int, "total_cost": float}
SESSIONS: dict[str, dict] = {}


def _get_state() -> dict:
    if "sid" not in session:
        session["sid"] = secrets.token_hex(8)
    sid = session["sid"]
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "history": [],
            "num_turns": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
    return SESSIONS[sid]


@app.route("/")
def index():
    return render_template("chat.html", default_persona=DEFAULT_PERSONA)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    persona = (data.get("persona") or DEFAULT_PERSONA).strip()
    if not user_msg:
        return jsonify({"error": "Tin nhắn rỗng"}), 400

    state = _get_state()

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = (
        [{"role": "system", "content": persona}]
        + state["history"]
        + [{"role": "user", "content": user_msg}]
    )

    def generate():
        reply = ""
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            yield delta

        state["history"].append({"role": "user", "content": user_msg})
        state["history"].append({"role": "assistant", "content": reply})
        state["history"] = state["history"][-6:]
        state["num_turns"] += 1
        state["total_tokens"] += count_tokens(user_msg) + count_tokens(reply)
        state["total_cost"] += estimate_cost(user_msg, reply)["total_cost"]

    return Response(generate(), mimetype="text/plain; charset=utf-8")


@app.route("/api/stats")
def stats():
    state = _get_state()
    return jsonify(
        {
            "num_turns": state["num_turns"],
            "total_tokens": state["total_tokens"],
            "total_cost": state["total_cost"],
        }
    )


@app.route("/api/diag")
def diag():
    """Chẩn đoán cấu hình + kết nối mạng — dùng khi không có quyền truy cập Shell."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        key_info = f"len={len(key)}, prefix={key[:6]!r}, suffix={key[-4:]!r}"
    else:
        key_info = "KHÔNG được set"

    base_url = os.getenv("OPENAI_BASE_URL") or "(mặc định: api.openai.com)"
    host = "api.openai.com"
    if os.getenv("OPENAI_BASE_URL"):
        from urllib.parse import urlparse

        host = urlparse(os.getenv("OPENAI_BASE_URL")).hostname or host

    result = {
        "openai_api_key": key_info,
        "openai_base_url": base_url,
        "resolved_host": host,
    }

    try:
        ip = socket.gethostbyname(host)
        result["dns_resolve"] = f"OK -> {ip}"
    except Exception as e:
        result["dns_resolve"] = f"FAILED: {e}"
        return jsonify(result)

    try:
        with socket.create_connection((host, 443), timeout=5):
            result["tcp_connect_443"] = "OK"
    except Exception as e:
        result["tcp_connect_443"] = f"FAILED: {e}"
        return jsonify(result)

    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"https://{host}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result["https_request"] = f"HTTP {resp.status}"
                result["https_body_snippet"] = resp.read(300).decode(
                    "utf-8", errors="replace"
                )
        except urllib.error.HTTPError as e:
            result["https_request"] = f"HTTP {e.code}"
            result["https_body_snippet"] = e.read(300).decode(
                "utf-8", errors="replace"
            )
    except Exception as e:
        result["https_request"] = f"FAILED: {type(e).__name__}: {e}"

    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset():
    state = _get_state()
    state["history"] = []
    state["num_turns"] = 0
    state["total_tokens"] = 0
    state["total_cost"] = 0.0
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Trên Render, startCommand dùng gunicorn nên khối này không chạy — chỉ
    # áp dụng khi chạy trực tiếp "python web_chat.py" (local dev).
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
