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
import time

from flask import Flask, Response, jsonify, render_template, request, session

from template import OPENAI_MODEL, count_tokens, estimate_cost, retry_with_backoff

DEFAULT_PERSONA = (
    "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
)
NEW_CHAT_TITLE = "Đoạn chat mới"

# Danh sách model cho dropdown lựa chọn trên UI. "id" phải khớp đúng tên
# model thật của nhà cung cấp (OpenAI hoặc endpoint tương thích qua
# OPENAI_BASE_URL) — nếu gõ sai tên, API sẽ báo lỗi rõ ràng khi gửi tin nhắn.
AVAILABLE_MODELS = [
    {"id": "gpt-5", "label": "GPT-5"},
    {"id": "gpt-5-mini", "label": "GPT-5 mini"},
    {"id": "gpt-4.1", "label": "GPT-4.1"},
    {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini"},
    {"id": "gpt-4o", "label": "GPT-4o"},
    {"id": "gpt-4o-mini", "label": "GPT-4o mini"},
]

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# session_id -> {
#     "conversations": {conv_id: {title, persona, model, history, num_turns,
#                                  total_tokens, total_cost, updated_at}},
#     "active_id": conv_id,
# }
SESSIONS: dict[str, dict] = {}


def _new_conversation(persona: str | None = None, model: str | None = None) -> dict:
    return {
        "title": NEW_CHAT_TITLE,
        "persona": persona or DEFAULT_PERSONA,
        "model": model or OPENAI_MODEL,
        "history": [],
        "num_turns": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "updated_at": time.time(),
    }


def _get_user_state() -> dict:
    if "sid" not in session:
        session["sid"] = secrets.token_hex(8)
    sid = session["sid"]
    if sid not in SESSIONS:
        conv_id = secrets.token_hex(6)
        SESSIONS[sid] = {
            "conversations": {conv_id: _new_conversation()},
            "active_id": conv_id,
        }
    return SESSIONS[sid]


def _get_conversation(user_state: dict, conv_id: str | None = None) -> tuple[str, dict]:
    """Trả về (conv_id, conversation) — dùng conv_id nếu hợp lệ, không thì
    dùng đoạn chat đang active; tự tạo mới nếu không còn đoạn nào."""
    conversations = user_state["conversations"]
    if conv_id and conv_id in conversations:
        user_state["active_id"] = conv_id
        return conv_id, conversations[conv_id]

    active_id = user_state.get("active_id")
    if active_id not in conversations:
        active_id = secrets.token_hex(6)
        conversations[active_id] = _new_conversation()
        user_state["active_id"] = active_id
    return active_id, conversations[active_id]


def _conversation_summary(conv_id: str, conv: dict) -> dict:
    return {
        "id": conv_id,
        "title": conv["title"],
        "num_turns": conv["num_turns"],
        "updated_at": conv["updated_at"],
    }


@app.route("/")
def index():
    return render_template(
        "chat.html",
        default_persona=DEFAULT_PERSONA,
        models=AVAILABLE_MODELS,
        default_model=OPENAI_MODEL,
    )


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    user_state = _get_user_state()
    convs = [
        _conversation_summary(cid, c) for cid, c in user_state["conversations"].items()
    ]
    convs.sort(key=lambda c: c["updated_at"], reverse=True)
    return jsonify({"conversations": convs, "active_id": user_state["active_id"]})


@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    data = request.get_json(force=True) or {}
    user_state = _get_user_state()
    conv_id = secrets.token_hex(6)
    user_state["conversations"][conv_id] = _new_conversation(
        persona=data.get("persona"), model=data.get("model")
    )
    user_state["active_id"] = conv_id
    return jsonify({"id": conv_id})


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def get_conversation(conv_id):
    user_state = _get_user_state()
    conv = user_state["conversations"].get(conv_id)
    if not conv:
        return jsonify({"error": "Không tìm thấy đoạn chat"}), 404
    user_state["active_id"] = conv_id
    return jsonify(
        {
            "id": conv_id,
            "title": conv["title"],
            "persona": conv["persona"],
            "model": conv["model"],
            "history": conv["history"],
            "num_turns": conv["num_turns"],
            "total_tokens": conv["total_tokens"],
            "total_cost": conv["total_cost"],
        }
    )


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    user_state = _get_user_state()
    user_state["conversations"].pop(conv_id, None)

    if not user_state["conversations"]:
        new_id = secrets.token_hex(6)
        user_state["conversations"][new_id] = _new_conversation()
        user_state["active_id"] = new_id
    elif user_state["active_id"] == conv_id:
        user_state["active_id"] = next(iter(user_state["conversations"]))

    return jsonify({"ok": True, "active_id": user_state["active_id"]})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Tin nhắn rỗng"}), 400

    user_state = _get_user_state()
    conv_id, conv = _get_conversation(user_state, data.get("conversation_id"))

    persona = (data.get("persona") or conv["persona"] or DEFAULT_PERSONA).strip()
    model = (data.get("model") or conv["model"] or OPENAI_MODEL).strip()
    conv["persona"] = persona
    conv["model"] = model

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = (
        [{"role": "system", "content": persona}]
        + conv["history"]
        + [{"role": "user", "content": user_msg}]
    )

    def generate():
        reply = ""
        actual_model = None
        try:
            stream = retry_with_backoff(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
            )
            for chunk in stream:
                if actual_model is None and getattr(chunk, "model", None):
                    actual_model = chunk.model
                delta = chunk.choices[0].delta.content or ""
                reply += delta
                yield delta
        except Exception as e:
            # Response đã bắt đầu stream với status 200 nên không thể đổi
            # sang mã lỗi HTTP nữa — báo lỗi ngay trong nội dung để UI hiển
            # thị, thay vì trả về rỗng im lặng (dễ gặp khi model gõ sai tên).
            yield f"⚠️ Lỗi gọi API với model '{model}': {type(e).__name__}: {e}"
            return

        # Marker để UI tách ra hiển thị model THẬT mà OpenAI đã dùng để trả
        # lời (đáng tin hơn nhiều so với việc hỏi model "bạn là ai" — model
        # tự nhận diện sai là chuyện thường gặp).
        if actual_model:
            yield f"[[MODEL::{actual_model}]]"

        conv["history"].append({"role": "user", "content": user_msg})
        conv["history"].append({"role": "assistant", "content": reply})
        conv["history"] = conv["history"][-6:]
        conv["num_turns"] += 1
        conv["total_tokens"] += count_tokens(user_msg, model) + count_tokens(reply, model)
        conv["total_cost"] += estimate_cost(user_msg, reply, model)["total_cost"]
        conv["updated_at"] = time.time()
        if conv["title"] == NEW_CHAT_TITLE:
            conv["title"] = user_msg[:36] + ("…" if len(user_msg) > 36 else "")

    return Response(generate(), mimetype="text/plain; charset=utf-8")


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

    # openai SDK dùng httpx2 nội bộ — test riêng để cô lập xem lỗi nằm ở
    # tầng mạng (đã loại ở trên), ở httpx2, hay ở chính client OpenAI.
    try:
        import httpx2

        result["httpx2_version"] = httpx2.__version__
        r = httpx2.get(
            f"https://{host}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        result["httpx2_request"] = f"HTTP {r.status_code}"
    except Exception as e:
        result["httpx2_request"] = f"FAILED: {type(e).__name__}: {e}"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL"))
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        result["openai_sdk_call"] = f"OK: {resp.choices[0].message.content!r}"
    except Exception as e:
        result["openai_sdk_call"] = f"FAILED: {type(e).__name__}: {e}"

    return jsonify(result)


if __name__ == "__main__":
    # Trên Render, startCommand dùng gunicorn nên khối này không chạy — chỉ
    # áp dụng khi chạy trực tiếp "python web_chat.py" (local dev).
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
