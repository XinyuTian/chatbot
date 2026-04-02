# Chatbot

FastAPI backend with a browser chat UI. All chatbot-specific code, static assets, and local config live in this folder.

## Run (development)

From **this directory** (`chatbot/`):

```bash
source ../.venv/bin/activate   # optional: shared venv at repo root
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/** (chat UI). API docs: **http://127.0.0.1:8000/docs**.

### One line from repo root

```bash
source .venv/bin/activate && cd chatbot && uvicorn main:app --reload
```

## Dependencies

```bash
source ../.venv/bin/activate   # or your own venv
pip install -r requirements.txt
```

## Environment

Create **`chatbot/.env`** (not committed to git) with:

```bash
AI_BUILDERS_API_KEY=your_key_here
```

The app loads `.env` from this folder automatically.

## What `uvicorn main:app --reload` means

| Part | Meaning |
|------|--------|
| `uvicorn` | ASGI server that runs your FastAPI app |
| `main` | Python file `main.py` |
| `app` | The `app = FastAPI(...)` instance inside `main.py` |
| `--reload` | Restart the server when code changes (dev only) |
