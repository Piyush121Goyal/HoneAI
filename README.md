# Hone AI

Turns a rough goal into a structured, production-grade prompt and streams it
back token-by-token. Two parts:

- **`frontend/`** — Vite + React, the UI (was a single `App.jsx` artifact,
  now a real buildable project).
- **`backend/`** — FastAPI + Postgres/pgvector + Redis + Celery. Ships with a
  **mock LLM provider by default**, so the whole stack runs with zero API keys.

## Run it locally

```bash
# backend
cd backend
cp .env.example .env
docker compose up --build
# → http://localhost:8000  (docs at /docs, health at /health)

# frontend, in another shell
cd frontend
cp .env.example .env      # VITE_API_BASE=http://localhost:8000
npm install
npm run dev
# → http://localhost:5173
```

`docker compose` brings up four services: `db` (pgvector), `redis`, `api`,
and the Celery `worker`. Schema + the `vector` extension are created on
startup (swap for the included Alembic setup — `backend/alembic/` — before
relying on this in production).

To use a real model instead of the mock, set in `backend/.env`:

```
LLM_PROVIDER=anthropic        # or openai
ANTHROPIC_API_KEY=sk-ant-...  # or OPENAI_API_KEY=sk-...
```

## Architecture

```
React (Vite) ── REST/JSON ──▶ FastAPI (async)
  │                             ├── /optimize  → SSE stream (analyze→draft→critique→refine)
  │◀──── SSE stream ────────────┤       │
  │                             │       └── httpx → LLM provider (or mock)
  │                             ├── /auth/*    → JWT (access in body, refresh in httpOnly cookie)
  │                             └── /prompts/* → save · list · search · rate
                                │
           Postgres + pgvector ◀── embeddings (semantic search)
           Redis  ◀── cache · rate limit · Celery broker
           Celery worker ◀── embedding backfill · batch optimize
```

## API reference

| Method | Path | Auth | Purpose |
|-------|------|------|---------|
| POST | `/optimize` | optional | Stream the optimized prompt (SSE) |
| POST | `/auth/signup` | – | Create account → `{access_token, user}` |
| POST | `/auth/login` | – | Log in → `{access_token, user}` |
| POST | `/auth/refresh` | cookie | Rotate tokens |
| POST | `/auth/logout` | – | Clear refresh cookie |
| POST | `/auth/google` | – | Exchange a Google ID token |
| GET | `/auth/me` | bearer | Current user |
| GET | `/prompts` | bearer | List saved prompts |
| POST | `/prompts` | bearer | Save a prompt (v1) |
| GET | `/prompts/search?query=…&k=5` | bearer | Semantic search (pgvector) |
| GET | `/prompts/{id}` | bearer | Prompt + versions |
| POST | `/prompts/{id}/feedback` | bearer | Rate a prompt |
| DELETE | `/prompts/{id}` | bearer | Delete |

### The SSE contract (`POST /optimize`)

Request body: `{ "goal", "model", "tone", "format" }`

```
event: stage
data: {"index": 0}          # 0 Understanding · 1 Drafting · 2 Refining

event: token
data: {"text": "You are "}  # streamed repeatedly

event: done
data: {"prompt": "...", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
```

This is exactly what `frontend/src/App.jsx`'s `api.optimize()` parses.

## Deploying

**Why Render for the backend, not Vercel:** Vercel's functions are
serverless/stateless — they can't host the always-on Celery worker or hold
a long-lived SSE connection past its timeout. Render runs both as real
long-lived processes.

### 1. Backend + worker + DB + Redis → Render

Push this repo to GitHub, then in the Render dashboard: **New → Blueprint**,
point it at the repo. `render.yaml` (repo root) defines everything:
- `hone-db` — managed Postgres (pgvector extension enabled at boot by the app)
- `hone-redis` — managed Redis
- `hone-api` — the FastAPI web service (Docker, `backend/Dockerfile`)
- `hone-worker` — the Celery worker (same image, different command)

After it deploys, in the `hone-api` service's environment settings fill in
the `sync: false` vars you want (`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` for a
real model, `GOOGLE_CLIENT_ID`/`SECRET`, `SENTRY_DSN`) and set
`CORS_ORIGINS` to your Vercel URL once you have it (step 2).

### 2. Frontend → Vercel

```bash
cd frontend
vercel        # or: import the repo in the Vercel dashboard, root = frontend/
```

Set the project's env var `VITE_API_BASE` to your Render API URL
(e.g. `https://hone-api.onrender.com`), and redeploy. Then go back to step 1
and set `hone-api`'s `CORS_ORIGINS` to the Vercel URL (comma-separated if
you have a preview + prod domain) and redeploy the API.

### 3. Verify

- `GET https://<render-api>/health` → `{"status":"ok"}`
- Sign up on the deployed frontend, run an optimize, save a prompt, confirm
  it shows up under History.

## Notes

- **Auth**: argon2 password hashing; short-lived access JWT + rotating refresh
  token in an httpOnly cookie. The frontend keeps the access token in memory
  only (never localStorage) and re-derives it from the refresh cookie on load.
- **Rate limiting**: fixed-window per user/IP in Redis, applied to `/optimize`.
- **Caching**: identical `/optimize` requests are served from Redis and
  replayed as a stream.
- **pgvector**: goal embeddings power `/prompts/search`; embeddings are
  computed by the Celery worker after a prompt is saved.
- **Migrations**: `backend/alembic/` is set up and ready — run
  `alembic revision --autogenerate -m "..."` once you're past the
  `create_all`-on-boot stage, and switch `init_db()` off.
- **Google OAuth**: create an OAuth 2.0 Client ID (Web application) in
  Google Cloud Console, add your Vercel origin to "Authorized JavaScript
  origins", then set `VITE_GOOGLE_CLIENT_ID` (frontend) and
  `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` (backend) — the "Continue with
  Google" button activates automatically once the client ID is present.
- **Sentry**: set `SENTRY_DSN` on the backend to enable error reporting;
  it's a no-op without it.
