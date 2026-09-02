# SmritiYog Backend

Sync-time agent backend for SmritiYog (SIH26003, team Cipher). The mobile app
plays the games; this service takes patient context on each sync, runs it
through a LangGraph pipeline, and returns a batch of pre-generated rounds plus
caretaker alerts.

**Not here:** gameplay, live scoring, TTS/STT, offline caching. RAG data arrives
per request and is never persisted — embeddings are computed and handed back for
the app to cache.

## Setup

```bash
cd backend
uv sync
cp .env.example .env      # fill in GEMINI_API_KEY and AUTH_SECRET
uv run --env-file .env uvicorn app.main:app --reload
uv run --env-file .env pytest
```

Env vars (`.env`):

| var | notes |
|---|---|
| `GEMINI_API_KEY` | Gemini generation + embeddings |
| `DATABASE_URL` | `sqlite+aiosqlite:///./smritiyog.db`; swap to `postgresql+asyncpg://...` for Neon — schema is already Postgres-compatible |
| `AUTH_SECRET` | shared bearer token (see Auth) |
| `ENV` | `development` / `production` |

No dotenv library: `uv run --env-file .env` loads the file, `config.py` just
reads `os.getenv`.

## Endpoints

All require `Authorization: Bearer $AUTH_SECRET`.

- `POST /sync` — `SyncRequest` in, `SyncResponse` out (`batch`, `embeddings`, `alerts`)
- `POST /recall/parse` — one finished `photo_recall` round: transcript in,
  `answer_matched` + new `rag_delta`-shaped facts out (see below)
- `POST /onboard/voice-clone` — base64 audio in, `voice_model_ref` out (**stubbed**, see below)
- `GET /dashboard/{patient_id}` — per-domain trend summary + unresolved alerts
- `GET /health`

## Pipeline

`START → planner → content → monitor → END` (`app/agents/graph.py`)

- **planner** — rule-based. Weakest domain picks the game type, its score picks
  the difficulty, and the last game played is not repeated. Gemini is called
  only to break a genuine tie between two domains that map to different games.
- **content** — one batched `embed_batch` call for the whole `rag_delta` (plus
  the rerank query), then builds `BATCH_SIZE` round *slots* deterministically:
  structured filter over `rag_delta` first, cosine rerank where the round needs
  fuzzy matching. `correct_answer` always comes verbatim from a retrieved item,
  so Gemini can never invent a family fact. One batched Gemini call then writes
  the spoken `prompt_text` (and voice clues) for every slot; if it fails, each
  slot ships with its template wording.
- **monitor** — reads persisted `SessionRecord` history (not just this
  request's payload), compares the last 3 sessions per domain to the 3 before.
  Drop > 15% alerts outright; 7–15% gets one batched Gemini "real decline or
  noise?" judgment; below 7% is ignored. Alerts go to state *and* to the DB.

Gemini call budget per sync: **≤ 4** (tie-break, embeddings, content wording,
monitor judgment) — two of which usually don't fire.

## `photo_recall` — the round that feeds the RAG store

`memory_voice` at hard difficulty serves `photo_recall`: the app shows a photo
from `rag_delta` and the patient talks. The answer is who the person is, but
they also volunteer things nobody typed into the app — so the round doubles as
RAG capture.

```
/sync  -> round {subtype: "photo_recall", photo_ref: "ph1",
                 correct_answer: "Meera"}
app     shows photo, records, transcribes locally
/recall/parse {photo_ref, expected_answer, transcript}
       -> {answer_matched: true,
           facts: [{id, kind, text, metadata:{source:"photo_recall", ...}}]}
app     stores facts; they ship back up as rag_delta on the next sync
```

One Gemini call does both jobs. If it fails, `answer_matched` falls back to a
substring check and `facts` comes back empty — a round never invents a memory.
Nothing is persisted server-side; the facts are handed straight back.

## Retrieval fallbacks

- `cognitive_visual` — no RAG at all, generated from local pools.
- `memory_voice` — `photo_recall` needs a `kind: photo` in `rag_delta`; with
  none, the slot falls back to a generic `favorite_recall` round.
- `relationship` — always needs RAG and has **no** personalized fallback.
  Rounds are skipped rather than fabricated; if that empties the batch, the
  batch switches to `cognitive_visual` so the app never gets nothing.

## Known simplifications

- **Auth is one shared secret**, not per-user. `verify_patient_scope()` in
  `app/core/security.py` is the seam where real scoping drops in — today it
  returns the id unchanged, so any token holder reaches any patient.
- **Voice cloning is stubbed.** `POST /onboard/voice-clone` validates the base64
  sample and returns a deterministic `mock-voice://<hash>` ref. `TODO` in
  `app/api/onboarding.py` marks the swap point.
- `Base.metadata.create_all` on startup instead of migrations.
- Embeddings are truncated to 768 dims (from `gemini-embedding-001`'s 3072)
  to keep the sync payload small, and left unnormalized — retrieval compares
  by cosine, which normalizes anyway.

## Layout

```
app/
  main.py config.py
  db/        session.py models.py           # Patient, SessionRecord, Alert
  schemas/   sync.py onboarding.py
  api/       sync.py onboarding.py dashboard.py
  agents/    graph.py state.py planner.py content.py monitor.py
  services/  gemini.py retrieval.py
  core/      security.py
tests/       test_sync.py test_agents.py
```
