# LightNote De-Editor

AI video de-editing: take a short-form video (TikTok / Reels / Meta Ads URL, or an upload) and
break it back down into the editable components it was assembled from — scenes, captions, text
overlays, product pop-ups — plus a "clean plate" with those elements removed.

> **Core idea:** de-editing is the inverse of editing, so the deliverable is not a cleaned mp4 —
> it is an **edit project**: `clean_plate.mp4` + `project.json`. Compositing the extracted tracks
> back onto the clean plate approximately reproduces the original, which both proves the
> decomposition worked and makes "rebuild / customize" a small feature instead of a second system.

Full technical design, staged plan and rationale: **[plan.md](plan.md)**.

---

## Status

| Block | Scope | State |
|---|---|---|
| **A** | Service skeleton: FastAPI, job queue, SQLite, workspaces, SSE progress, RFC-7807 errors, health probes, Docker | ✅ **done** |
| **B** | Ingest (yt-dlp + upload), ffprobe + proxy/audio, PySceneDetect, scene clips + keyframes, frame sampling | ✅ **done** |
| **C** | RapidOCR, temporal text tracking, style extraction, caption/overlay/watermark classification | ✅ **done** |
| **D** | Mask timeline, tiered inpainting, clean video + per-scene clean clips | ✅ **done** |
| **E** | React frontend: URL/upload input, live SSE status, scene timeline, original↔clean A/B | ✅ **done** |
| **F** | Image / product pop-up detection (stability + transience + structure, per scene) | ✅ **done** |
| **G** | Whisper ASR + speech-aligned caption classification; provider-agnostic VLM pass | ✅ **done** |
| H–I | LaMa inpainting, recompose / rebuild from the edited project | ⏳ next |

**Every capability the brief asks for now works end to end.** Open the UI, paste a URL or drop a
file, watch the stages stream past, then explore the breakdown: a scene timeline with text and
overlay lanes, an original↔de-edited A/B player with detection boxes drawn from the normalized
coordinates, and a per-scene inspector. Scenes, captions, text overlays and image/product pop-ups
are detected, isolated and removed, and the pipeline is complete end to end — there are no
placeholder stages left.

Actual output for a synthetic UGC clip (3 scenes, burned-in captions, a corner watermark):

```
scenes:  sc_001 0.0→2.0   sc_002 2.0→4.0   sc_003 4.0→6.0

tx_001 [     caption]  0.00-2.00  n=6   box=(0.07,0.85,0.45,0.03)  #FFF8EC  'HELLO WORLD'
tx_002 [   watermark]  0.00-6.00  n=18  box=(0.03,0.04,0.18,0.03)  #B6EDA5  '@brandco'
tx_003 [     caption]  2.00-4.00  n=6   box=(0.19,0.85,0.30,0.03)  #F2FFED  'BUYNOW'
tx_004 [overlay_text]  4.00-6.00  n=6   box=(0.22,0.16,0.28,0.04)  #F2FDFF  '50%0FF'
```

Timings land exactly on the burned-in spans, and each element is classified by role. (`BUYNOW`,
`50%0FF` — the recogniser drops spaces and confuses `O`/`0` on the synthetic font; real footage
reads better.)

Those tracks collapse into a mask timeline, and the clean plate is rendered from it:

```
mask timeline: 5 segments, 6.0s covered (100%)
   0.00-1.92  ['tx_001:caption', 'tx_002:watermark']
   1.92-2.08  ['tx_001:caption', 'tx_002:watermark', 'tx_003:caption']   ← fade overlap
   2.08-3.92  ['tx_002:watermark', 'tx_003:caption']
   ...
outputs: clean/clean.mp4 · method=opencv · 90/90 frames inpainted
```

Pop-ups are found the same way and flow into the same masking, with no extra wiring. On a panning
fixture with a card pasted at a known box for a known window:

```
ground truth   box=(0.52,0.10,0.38,0.22)   t=1.0 → 3.0
detected       box=(0.43,0.09,0.51,0.24)   t=1.17 → 2.83   IoU 0.68   conf 0.588
               signals={stability: 0.63, transience: 0.0, detail: 0.60, solidity: 0.51}
```

One detection, no false positives, timing correct to within one sampling interval — and a
text-only video yields zero overlay tracks, so captions are never re-reported as images.

**The removal is verified by re-running OCR on the output**, not by asserting that a file exists:

```
t=1.0s   orig: ['@brandco', 'HELLO WORLD']     t=1.0s  clean: []
t=3.0s   orig: ['@brandco', 'BUY NOW']         t=3.0s  clean: []
t=5.0s   orig: ['@brandco', '50%0FF']          t=5.0s  clean: []
```

---

## The interface

Four panels, functionality over polish (per the brief):

| Panel | What it does |
|---|---|
| **Source** | URL field or drag-and-drop upload, removal-method selector |
| **Processing** | SSE-driven stage stepper — each stage shows its measured time as it completes, doubling as a live profile — plus a progress bar, log tail, degradations and cancel |
| **Timeline** | Scene strip with keyframe thumbnails, and text/overlay lanes on one shared time axis; click to select and seek |
| **Player + Inspector** | Original ↔ de-edited toggle at a shared timestamp, detection boxes overlaid, and per-scene original/clean clip previews with the recovered text, colour, box and confidence |

## Architecture

```
React (Vite)  ──POST /jobs──►  FastAPI  ──enqueue──►  worker (Celery | thread pool)
     ▲                            │                          │
     └──── SSE /jobs/{id}/events ─┘                          ▼
                                              ingest → probe → scenes → sample →
                                              ocr → asr → text_tracks → overlays →
                                              vlm → masks → inpaint → export → finalize
                                                              │
       storage/{job_id}/  ◄────── artifacts + project.json ───┘
       SQLite: job, job_event
```

**Key decisions** (the long version is in `plan.md §2–3`):

- **Worker + broker, not `BackgroundTasks`.** A 30 s clip is 1–4 min of CPU; blocking the event
  loop would kill SSE and make cancellation impossible.
- **Two queue backends behind one `dispatch` interface.** `QUEUE_MODE=celery` for the real thing,
  `QUEUE_MODE=thread` for local dev — no Redis required, which matters on Windows where Celery's
  prefork pool is unavailable.
- **Events table, not Redis pub/sub, for progress.** Works identically in both queue modes, and a
  reconnecting client replays what it missed via `Last-Event-ID`. Pub/sub is the multi-node upgrade.
- **Stages are pure, resumable and independently testable.** A stage gets a `JobContext` (workspace,
  options, progress callback) and returns a JSON-able checkpoint; the runner owns ordering,
  weight-scaled progress, timing, cancellation and failure policy (`required=False` stages degrade
  instead of failing the job).
- **Honest degradation.** Anything the pipeline had to skip or downgrade is recorded in
  `diagnostics.degradations` and surfaced through the API. If scene detection returns a single
  scene for a video long enough that an edit is near-certain, the pipeline falls back to uniform
  chunking and *says so* rather than reporting a confident wrong answer.
- **Analyse small, render big.** Every detector runs on a 720p `proxy.mp4`; the clean video is
  rendered from the full-resolution source. Cheap detection, full-quality output.
- **Detections become *tracks*, not boxes.** Two detections continue the same text track only when
  they overlap spatially **and** read the same. Either test alone is wrong: box-only merges a
  caption that changes text in place into one long track; text-only merges the same word in two
  corners. That pairing is what turns OCR noise into objects an editor can manipulate.
- **Static frames are not re-OCR'd.** Consecutive sampled frames with an identical 32×32 greyscale
  fingerprint reuse the previous result — short-form video holds a frame far more often than it
  changes.
- **Style is measured off the glyph core.** Antialiasing blends text into background over a 1 px
  rim; eroding the glyph mask before sampling colour, and reading the outline from the band just
  beyond the rim, is what makes the recovered colour usable for re-rendering.
- **Speech is what actually classifies a caption.** Position cannot: a hook line and a subtitle
  both sit in the lower third of a UGC ad. Matching each OCR track against the words spoken in its
  own time window answers the real question — *are the words on screen the words being said?* Text
  that matches is a `caption` wherever it sits; text that is never spoken is an `overlay_text` even
  when it sits exactly where a subtitle would. `kind_reason` records which rule fired, and the UI
  shows it.
- **The VLM is asked once per scene, not per frame or per candidate.** The keyframe and every
  candidate crop for that scene go up in one request, capped at 12 scenes. Per-frame calls would be
  ~900 requests; no calls at all loses every semantic. Output is constrained to a strict JSON
  schema (`output_config.format` / `response_json_schema`), so there is no prose parsing.
- **A rejected candidate is marked, not deleted.** When the model rules that a region is a poster on
  the wall rather than a pasted graphic, it stays in `overlay_tracks` for inspection but is dropped
  from the mask timeline — so a false positive stops being painted out of the footage.
- **No key is a supported configuration.** The provider factory returns a no-op, the stage records
  `vlm_unavailable`, and every CV result stands. Whisper is local and needs no key at all.
- **Pop-up detection uses two signals that are mutually exclusive, not combined.** A pasted layer
  is nailed to the screen, so when the camera pans its temporal variance collapses while the
  world's does not. On a locked-off camera that tells you nothing, so the fallback counts *how many
  times* each pixel changed — a pop-up changes once or twice, a person in nearly every frame.
  ORing them is actively harmful: panning over broad flat regions puts background transition counts
  in the same 1–2 range as a pop-up (measured: 2.66 background vs 2.74 inside a real card). The
  camera-motion estimate picks one.
- **Both signals are intersected with dilated edges.** A flat wall is perfectly stable and never
  changes, so it looks exactly like an overlay; requiring structure rejects it, and also stops flat
  background from fusing with a real overlay into one oversized blob.
- **Stability is measured over sliding windows, not whole scenes.** A pop-up that appears mid-shot
  is not stable across the scene — its region holds background, then a frozen card, then background
  again. Windows find it; template-matching the patch back across the scene recovers its true
  in/out times and rejects single-frame flickers that the spatial scores cannot distinguish from a
  real asset.
- **The mask timeline stores changes, not frames.** A 30 s clip is ~900 frames but the active box
  set changes maybe a dozen times. Segments keep the artifact small enough to live in
  `project.json`, render directly as a UI timeline, and let the inpainter binary-search a frame's
  boxes instead of loading a mask image per frame.
- **Removal is tiered and self-downgrading.** `mask_blur` → `opencv` (Telea) → `lama`. `auto` picks
  the best tier actually installed, and the method that ran is recorded in `outputs.removal_method`
  so the result never implies a quality it does not have. Only masked frames are inpainted.
- **The A/B player swaps one video's `src` rather than stacking two.** Two players would double
  the decode cost and drift apart; carrying the timestamp across the swap makes "same frame, with
  and without the edit" true by construction.
- **Detection boxes are positioned from normalized coordinates**, so they line up at any player
  size without the frontend knowing the video's resolution — the same property that lets analysis
  run on a 720p proxy and render at source resolution.
- **SSE degrades to polling.** Proxies and corporate networks drop event streams; the UI switches
  and says so rather than freezing.
- **Frames are piped into ffmpeg, not written by OpenCV.** `cv2.VideoWriter` cannot carry the
  original audio and picks whatever codec the local build has; a raw-frame pipe keeps encoding
  decisions in one place and memory flat regardless of clip length.
- **Frame-accurate scene cuts.** Clips are re-encoded rather than stream-copied: a stream copy can
  only cut on keyframes, and UGC edits routinely place cuts mid-GOP, which shows up as a frozen
  first frame.
- **Normalized 0–1 coordinates everywhere**, so analysis can run on a 720p proxy and render at
  source resolution with no conversion bugs.

---

## AI integration

| Model | Where it runs | Key needed | What it contributes |
|---|---|---|---|
| PP-OCRv4 (RapidOCR / ONNX) | local, CPU | no | on-screen text detection + recognition |
| faster-whisper (CTranslate2) | local, CPU | no | transcript → separates captions from graphics |
| Gemini 2.5 Flash **or** Claude (`claude-opus-5`) | API | **optional** | scene descriptions; names and validates pop-ups |
| LaMa (optional extra) | local | no | higher-quality inpainting |

Set `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY` with `VLM_PROVIDER=claude`) in `backend/.env` to enable
the semantic pass. Without it everything still runs — the job records `vlm_unavailable` and keeps
the computer-vision results.

Both providers implement one interface (`app/ai/base.py`) and are constrained to the same JSON
schema, so switching is a config change.

## Run it locally

Requires **ffmpeg** on `PATH`. Python **3.11** is recommended and is what the Docker image runs:
several media/AI wheels cap out below 3.13 (`rapidocr-onnxruntime` resolves to 1.2.3 on 3.13 versus
1.4.x on 3.11; `faster-whisper` and the inpainting models are stricter still — see `plan.md §3`).
Everything here is verified on 3.13 with those older pins as well.

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,media,ocr]"   # Linux/macOS: .venv/bin/python
cp .env.example .env                                 # optional; defaults work
.venv/Scripts/python -m uvicorn app.main:app --reload
```

- API docs (OpenAPI): <http://localhost:8000/docs>
- Readiness (db / storage / ffmpeg / broker): <http://localhost:8000/readyz>

Then start the UI in a second terminal:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 — proxies /api and /media to :8000
```

### With Docker (frontend + api + worker + redis)

```bash
docker compose up --build      # whole app on http://localhost:5173
```

nginx serves the built bundle and proxies `/api` and `/media` to the API container, so the app is
one origin — no CORS, and SSE and video Range requests pass through unbuffered.

### Tests

```bash
cd backend && .venv/Scripts/python -m pytest -q
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/jobs` | Submit — `application/json {url}` **or** `multipart/form-data file=` |
| `GET` | `/api/v1/jobs` | List, newest first |
| `GET` | `/api/v1/jobs/{id}` | Status, stage, progress, degradations, timings |
| `GET` | `/api/v1/jobs/{id}/events` | Live progress (SSE, resumable via `Last-Event-ID`) |
| `GET` | `/api/v1/jobs/{id}/project` | `project.json` — the editable result |
| `GET` | `/api/v1/jobs/{id}/files/{path}` | Artifact download (traversal-guarded) |
| `POST` | `/api/v1/jobs/{id}/cancel` | Cooperative cancel at the next stage boundary |
| `DELETE` | `/api/v1/jobs/{id}` | Delete job, events and workspace |
| `GET` | `/healthz`, `/readyz` | Probes |

`/media/{job_id}/...` serves the same artifacts with HTTP Range support, for `<video>` seeking.

Errors are `application/problem+json` with stable codes (`E_BAD_INPUT`, `E_UNSUPPORTED_MEDIA`,
`E_TOO_LARGE`, `E_DOWNLOAD_BLOCKED`, `E_NO_VIDEO_STREAM`, `E_STAGE_FAILED`, …) so the client can
react without string-matching. Uploads are streamed to disk and hashed as they arrive; the SHA-256
doubles as a **result cache key** — resubmitting the same bytes returns the existing job.

Quick check:

```bash
curl -X POST localhost:8000/api/v1/jobs -H 'Content-Type: application/json' \
     -d '{"url":"https://www.tiktok.com/@demo/video/12345"}'
curl -N localhost:8000/api/v1/jobs/<id>/events
curl localhost:8000/api/v1/jobs/<id>/project
```

---

## Layout

```
plan.md                     full technical plan + schedule
docker-compose.yml          api · worker · redis
backend/
  app/
    main.py config.py errors.py storage.py events.py
    api/v1/    jobs.py health.py router.py
    db/        models.py session.py
    schemas/   job.py
    services/  jobs.py                  # lifecycle rules: limits, dedupe, cancel
    pipeline/  base.py context.py runner.py stages/
    worker/    celery_app.py tasks.py dispatch.py
    utils/     ffmpeg.py style.py        # subprocess layer; text colour/outline recovery
    pipeline/  geometry.py tracking.py   # normalized boxes; OCR -> temporal tracks (pure)
    pipeline/stages/  ingest probe scenes sample ocr text_tracks masks inpaint export finalize
  tests/
frontend/
  src/api/       client.ts types.ts      # typed client, SSE + polling fallback
  src/components/ SubmitPanel StatusPanel SceneTimeline ComparePlayer Inspector
  Dockerfile nginx.conf                  # static bundle + same-origin API proxy
```

---

## Known limitations

Tracked in `plan.md §11`. Current build:

- Pop-up detection is geometric, so it reports `kind: "overlay"` with no label — it knows a
  composited layer is present, not that it is a serum bottle. Naming them (and rejecting
  lookalikes such as a picture frame on the wall) is what the VLM pass in block G adds.
  Soft-alpha gradients, animated stickers and full-frame transitions are still missed.
- Detected pop-up boxes run slightly loose (IoU ~0.7 against ground truth) because the mask is
  dilated and solidified; masking is padded anyway, so removal is unaffected, but a tight cutout
  for *replacing* an overlay needs SAM2.
- The default removal tier is OpenCV Telea, which smears on busy or moving backgrounds. LaMa
  (`pip install -e ".[inpaint]"`) is a large quality jump; ProPainter would fix temporal coherence
  but is GPU-bound and out of scope for this deadline.
- `caption` vs `overlay_text` is currently a position/lifetime heuristic. The real separation —
  is this text a subtitle of what is being *said*, or a graphic that is never spoken — needs the
  ASR alignment in block G.
- `has_bg_box` really measures "the background behind the glyphs is flat", which is a caption
  plate on real footage but also fires on a synthetically flat background.
- Scene detection handles cuts, not dissolves or whip transitions.
- Only Latin script is enabled in the OCR model pack.
- Without a VLM key, pop-ups are reported as `kind: "overlay"` with no label, and the false
  positives in the table above are not filtered.
- The first job on a fresh machine downloads the Whisper weights (~38 s measured, vs ~1 s once
  cached). Workers warm the models at boot to keep that off the first request; the Docker image
  should bake them in for a cold deploy.
