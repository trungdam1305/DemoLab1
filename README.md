# DemoLab1 — Streaming Chatbot Web UI

Giao diện web (Flask) cho trợ lý chatbot streaming, dựa trên
`run_assistant` / `streaming_chatbot` của bài lab K4 Ngày 1
(Khám Phá LLM API).

## Chạy local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # rồi dán API key thật vào .env
python web_chat.py
```

Mở `http://127.0.0.1:5000`.

## Deploy lên Render

### Cách 1 — Blueprint (dùng `render.yaml` có sẵn)

1. Đăng nhập [render.com](https://render.com) bằng tài khoản GitHub.
2. **New** → **Blueprint** → chọn repo này.
3. Render tự đọc `render.yaml`. Khi được hỏi, nhập giá trị cho biến môi
   trường `OPENAI_API_KEY` (dán API key thật — **không** commit key vào Git).
4. Bấm **Apply** — Render build và deploy tự động.

### Cách 2 — Tạo Web Service thủ công

1. **New** → **Web Service** → chọn repo này.
2. Cấu hình:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn web_chat:app --bind 0.0.0.0:$PORT`
3. Tab **Environment** → thêm biến `OPENAI_API_KEY` = key thật của bạn.
4. Bấm **Create Web Service**.

Sau khi deploy xong, Render cấp một URL dạng
`https://demolab1-chat.onrender.com` — mở link đó để dùng thử.

## Lưu ý

- Free tier của Render sẽ "ngủ" sau một thời gian không có traffic — lần
  truy cập đầu sau khi ngủ có thể mất 30–60 giây để khởi động lại.
- Mỗi lần Render khởi động lại instance, session chat trong bộ nhớ (`SESSIONS`
  trong `web_chat.py`) sẽ mất — đây là giới hạn chấp nhận được cho bản demo.
- Không bao giờ commit file `.env` hoặc dán API key trực tiếp vào code —
  luôn dùng biến môi trường của Render.
