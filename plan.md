# LightNoteAI — AI Video De-Editor
## Implementation Plan (Python + FastAPI)

**Author:** data2@restthecase.com
**Date:** 2026-09-05 · **Deadline:** 2026-09-06, 18:00 (≈22 working hours left)

---

## 0. TL;DR of the approach

> **De-editing is the inverse of editing. So model the output as an edit project, not as a video.**

The whole design falls out of one decision: the deliverable is **not** "a cleaned mp4".
The deliverable is a **structured, editable EDL (Edit Decision List) + a clean plate**:

```
input video  ──►  DECOMPOSE  ──►  {  scenes[]            (cuts + keyframes + clips)
                                     text_tracks[]       (content, box, t_in/t_out, style)
                                     overlay_tracks[]    (RGBA cutout PNG, box, t_in/t_out)
                                     clean_plate.mp4     (all of the above removed)
                                     audio.wav + transcript }
                                          │
                                          ▼
                              project.json (the editable form)
                                          │
                            user edits in UI ──► RECOMPOSE ──► rebuilt.mp4
```

`clean_plate + project.json` is a **lossless-enough re-render**: compositing the tracks back
onto the clean plate approximately reproduces the original. That round-trip is the proof the
de-edit actually worked, and it is what makes "rebuild / customize" (the two bonus items)
a **1-hour feature instead of a rewrite**. Everything below serves this.

---

## 1. Time budget (hard constraint)

~22h left. The plan is **strictly tiered**. Tier 1 = submittable. Tier 2/3 = score.

| Tier | Scope | Status target |
|---|---|---|
| **T1 — must ship** | Ingest (URL+upload) → scenes → OCR text tracks → naive overlay detection → mask+inpaint (OpenCV) → clean video + scene clips → job queue → REST API → React UI → README + demo | **non-negotiable** |
| **T2 — score drivers** | Whisper ASR + caption↔speech alignment, Gemini VLM scene understanding, LaMa inpainting, SSE progress, recompose endpoint (re-add/edit captions) | do if T1 lands by hour 12 |
| **T3 — stretch** | SAM2 cutouts, GroundingDINO open-vocab pop-ups, ProPainter temporal inpaint, overlay image replacement, Docker compose | only if time remains |

**Cut rule:** every AI stage has a deterministic fallback. No stage may be able to fail the job.

---

## 2. Architecture

```
┌──────────────┐   POST /jobs (url | file)     ┌──────────────────────┐
│ React + Vite │ ────────────────────────────► │  FastAPI (api)       │
│  TypeScript  │   GET  /jobs/{id}/events SSE  │  - validation        │
│  Tailwind    │ ◄──────────────────────────── │  - enqueue           │
└──────┬───────┘   GET  /jobs/{id}/project     │  - serve artifacts   │
       │           POST /projects/{id}/render  └──────┬───────────────┘
       │ <video> preview                              │ Redis (broker + pub/sub)
       │ scene timeline                               ▼
       ▼                                       ┌──────────────────────┐
  static artifacts  ◄─────────────────────────  │  Celery worker       │
  /storage/{job}/…                              │  pipeline DAG        │
                                                └──────┬───────────────┘
                                                       │
   ┌───────────────────────────────────────────────────┴────────────────────────┐
   │ ingest → probe → scenes → sample → OCR → ASR → track → overlays → VLM →     │
   │ mask → inpaint → export(clean + clips + assets) → project.json              │
   └────────────────────────────────────────────────────────────────────────────┘

  SQLite (SQLModel) — jobs, stages, projects       Filesystem — /storage/{job_id}/
```

**Why a worker + broker and not `BackgroundTasks`:** a 30s TikTok takes 1–4 min of CPU. Blocking
an ASGI worker on that starves the event loop, kills SSE and makes cancellation impossible. Celery
gives concurrency limits, retries, cancellation and per-stage progress for free. (The assignment
explicitly lists "background processing / job queue" as a bonus.)

---

## 3. Stack + why

| Concern | Choice | Why this one |
|---|---|---|
| API | **FastAPI + Pydantic v2** | async, native OpenAPI docs = "clean API design" bonus for free, typed contracts shared with the UI |
| Queue | **Celery + Redis** | battle-tested, `task.update_state()` → progress; Redis pub/sub also backs SSE |
| DB | **SQLite via SQLModel** | zero-ops for a prototype, one file; SQLModel = same Pydantic models for DB + API |
| Download | **yt-dlp** | the only realistic answer for TikTok/IG/Reels/Meta Ads URLs; handles redirects and watermark-free variants |
| Video I/O | **ffmpeg (subprocess) + OpenCV/PyAV** | ffmpeg for cut/encode/mux (fast, exact); OpenCV for per-frame CV. **Not MoviePy** — slow and leaks temp files |
| Scenes | **PySceneDetect** (`AdaptiveDetector` + `ContentDetector`) | HSV content deltas; adaptive handles the fast whip-cuts typical of UGC ads. Deterministic, CPU-cheap, no model download |
| OCR | **RapidOCR (PP-OCRv4 ONNX)** primary, EasyOCR fallback | PP-OCRv4 quality without the PaddlePaddle install pain. Pure `onnxruntime` → installs cleanly on **Python 3.13** (Paddle wheels do not). Gives quad boxes + confidence |
| ASR | **faster-whisper** (`small`/`base`, int8) | word-level timestamps; CTranslate2 is ~4x realtime on CPU. Used to *classify* text, not just transcribe |
| VLM | **Gemini 2.5 Flash** (Claude/GPT swappable behind one interface) | cheap, fast, large context, native multi-image; used for scene semantics + overlay classification with **structured JSON output** |
| Inpaint | tier: **OpenCV Telea** → **LaMa** (`simple-lama-inpainting`) → ProPainter | graceful degradation; LaMa is the sweet spot (single model, usable without a GPU) |
| Segmentation (T3) | **SAM2 / MobileSAM** + GroundingDINO or YOLO-World | box → tight RGBA cutout so overlays become *replaceable assets* |
| Frontend | **React + Vite + TS + Tailwind** | fastest path to a timeline + inspector UI; TS types generated from the OpenAPI schema |

> ⚠️ **Env note:** local Python is 3.13.9, ffmpeg 8.1 is already on PATH. Pin the project to
> **Python 3.11** in a venv — `paddleocr`, some `torch` extras and `propainter` are still flaky on
> 3.13. If 3.13 is forced, the RapidOCR / faster-whisper / OpenCV path above is the 3.13-safe subset.

---

## 4. The pipeline, stage by stage

Each stage is a pure function `(JobContext) -> StageArtifact`, writes JSON + files into
`/storage/{job_id}/`, and is **independently resumable**. This is the core of the backend score.

### S1 · Ingest
- URL → `yt-dlp` (format `bv*+ba/b`, remux to mp4). Upload → stream to disk with size/duration caps.
- Guard rails: max 200 MB, max 180 s, container/codec whitelist, `ffprobe` before anything else.
- `sha256` of the file = **content key**. Re-submitting the same video short-circuits to the cached
  result (good demo moment, real engineering point).

### S2 · Probe & normalize
- `ffprobe` → duration, fps, resolution, rotation, has_audio.
- Normalize to a working copy: constant fps, `yuv420p`, longest side ≤ 1080. Analyse on a ≤720p
  **proxy**, render the final at source resolution. *Analyse small, render big* — 3–4x speedup.

### S3 · Scene segmentation
- `PySceneDetect` `AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=0.4s)`.
- Post-process: merge scenes < 0.35 s (UGC ads are full of flash frames), snap boundaries to the
  nearest keyframe so clips can be stream-copied.
- Export per-scene clips with `ffmpeg -ss -to -c copy` (instant) + a representative **keyframe**
  per scene (mid-scene frame, plus first/last for the VLM).
- Fallback if detection returns a single scene on an obviously edited video: uniform 3 s chunking,
  flagged `method: "fallback_uniform"` in the output. Honest degradation beats silent failure.

### S4 · Frame sampling
- Sample at **3 fps** inside each scene (5 fps for scenes < 2 s). ~90 frames for a 30 s video →
  OCR in seconds, not minutes.
- Keep the frame-index ↔ timestamp map; every downstream artifact is expressed in **seconds**,
  never in frame numbers.

### S5 · Text / caption detection (OCR)
- RapidOCR per sampled frame → `[{quad, text, conf}]`. Drop conf < 0.5 and boxes < 0.5% of frame area.
- **Temporal tracking** — this is what turns OCR noise into editable objects:
  - Group detections across consecutive samples by `IoU(box) > 0.5` **and** `rapidfuzz.ratio(text) > 80`.
  - A group becomes a **TextTrack**: `{text (modal value), t_in, t_out, box (median), style_hint}`.
  - Merge tracks separated by ≤ 1 sample gap (OCR flicker); drop tracks shorter than 0.25 s.
- **Style extraction** for rebuild: dominant text colour (k-means on pixels inside the mask),
  stroke/shadow presence, background-box presence (variance of the ring around the glyphs),
  approximate font size (box height), normalized anchor position.

### S6 · ASR + classification (the "understanding" bit)
- `faster-whisper` → word-level segments.
- For each TextTrack, compute fuzzy overlap between its text and the words spoken in `[t_in, t_out]`:
  - high similarity + bottom-centre + short lifetime → **`caption`** (burned-in subtitle)
  - no speech match + long lifetime + top/corner placement → **`overlay_text`** (hook line, CTA, price)
  - all-caps + persistent + corner → **`watermark/logo`**
- This is a genuinely useful signal that pure OCR cannot produce, and it maps directly onto the
  assignment's "captions **and** text overlays" distinction.

### S7 · Image / product pop-up detection
Three-signal ensemble — no single signal is reliable on UGC footage:

1. **Temporal-stability residual.** Per scene, build a background model (temporal median over the
   scene after coarse global-motion compensation via `cv2.estimateAffinePartial2D` on ORB features).
   Regions with low temporal variance *while the scene itself has global motion* are composited
   layers, not part of the world.
2. **Hard-edge rectangularity.** Canny → contours → `approxPolyDP`. Overlays are pasted assets, so
   they show unnaturally straight, high-contrast borders and often alpha-matte or drop-shadow edges.
   Score by edge straightness + corner count.
3. **Abrupt appearance.** A frame-difference spike localized to a region **without** a scene cut =
   something popped in.

Candidate boxes = fusion of the three (NMS + score threshold), minus any box already claimed by a TextTrack.

Then **verify with the VLM** (T2): send the scene keyframe plus candidate crops to Gemini with a
structured-output schema:

```json
{"is_overlay": true, "kind": "product_popup|sticker|logo|ui_element|none",
 "label": "serum bottle", "confidence": 0.87}
```

This is the right division of labour: **CV proposes cheap candidates, the VLM adjudicates
semantics.** Sending every frame to an LLM would be slow and expensive; sending nothing loses all
semantics.

*(T3)* Feed accepted boxes to **SAM2/MobileSAM** → precise mask → save an **RGBA cutout PNG** per
overlay. That cutout is what makes "replace the product image" trivial.

### S8 · Mask timeline
- Rasterize every TextTrack + OverlayTrack into a per-frame binary mask, dilated by ~1.5% of frame
  height (inpainting needs slack around glyph antialiasing), then temporally smoothed
  (morphological close over time) so masks don't strobe.
- Emitted as a run-length mask timeline, not 900 PNGs.

### S9 · Removal / clean plate
Tiered, chosen by config + hardware probe:

- **T1 `opencv`** — `cv2.inpaint(..., INPAINT_TELEA)` per masked frame. Instant, mediocre, always works.
- **T2 `lama`** — `simple-lama-inpainting` per masked frame. Big quality jump; only runs on frames
  that actually carry a mask (typically ~40%).
- **T2.5 `median_plate`** — for a *static* overlay over a *moving* background, the temporal-median
  background from S7 is already a near-perfect clean plate. **Prefer it over inpainting wherever
  available** — faster *and* better. Good detail to discuss in the interview.
- **T3 `propainter`** — flow-guided video inpainting, temporally coherent, GPU.
- Always also emit a `masked` mode (blur / solid fill) as a guaranteed-successful visual.

Re-encode the clean plate at source resolution and mux the original audio back (`-c:a copy`).

### S10 · Export & assemble
```
/storage/{job_id}/
  source.mp4  proxy.mp4  audio.wav
  clean/clean.mp4                      # de-edited full video
  clean/scene_003.mp4                  # de-edited per-scene clips
  scenes/scene_003.mp4                 # original per-scene clips
  frames/scene_003_key.jpg
  overlays/ov_007.png                  # RGBA cutout (replaceable asset)
  masks/timeline.json
  project.json                         # ◄── THE deliverable
```

### S11 · Recompose (bonus, and the payoff)
`POST /projects/{id}/render` takes an edited `project.json` and rebuilds:
- text → PIL-rendered RGBA strips (respecting extracted style), overlaid via the ffmpeg `overlay`
  filter with `enable='between(t,..,..)'`
- overlays → the stored PNG, or a **user-uploaded replacement image** letterboxed into the original box
- scenes → `concat`, with optional reorder / drop / trim

The same filtergraph builder serves both "re-add the original captions" (the round-trip proof) and
"customize it".

---

## 5. `project.json` — the contract

```jsonc
{
  "schema_version": "1.0",
  "job_id": "…", "source": {"kind": "url|upload", "ref": "…", "sha256": "…"},
  "media": {"duration": 28.4, "fps": 30, "width": 1080, "height": 1920, "has_audio": true},
  "scenes": [{
    "id": "sc_001", "index": 0, "t_in": 0.0, "t_out": 3.2,
    "keyframe": "frames/sc_001.jpg",
    "clip": "scenes/sc_001.mp4", "clean_clip": "clean/sc_001.mp4",
    "ai": {"description": "Woman holding serum bottle to camera, bathroom",
           "shot_type": "medium close-up", "tags": ["ugc", "talking head"]}
  }],
  "text_tracks": [{
    "id": "tx_004", "kind": "caption",            // caption | overlay_text | watermark
    "text": "this changed my skin",
    "t_in": 1.02, "t_out": 2.60, "scene_id": "sc_001",
    "box": {"x": 0.12, "y": 0.78, "w": 0.76, "h": 0.07},    // normalized 0..1
    "style": {"color": "#FFFFFF", "stroke": "#000000", "font_px_norm": 0.045,
              "align": "center", "has_bg_box": false},
    "confidence": 0.93, "source": "ocr+asr",
    "editable": true                              // UI writes back text, style, timing
  }],
  "overlay_tracks": [{
    "id": "ov_002", "kind": "product_popup", "label": "serum bottle",
    "t_in": 4.10, "t_out": 6.85, "scene_id": "sc_002",
    "box": {"x": 0.55, "y": 0.10, "w": 0.35, "h": 0.30},
    "asset": "overlays/ov_002.png", "replaceable": true,
    "confidence": 0.81, "source": "cv+vlm"
  }],
  "transcript": [{"t_in": 0.9, "t_out": 2.7, "text": "this changed my skin"}],
  "outputs": {"clean_video": "clean/clean.mp4", "removal_method": "lama"},
  "diagnostics": {"stage_timings_ms": {}, "degradations": ["asr_skipped_no_audio"]}
}
```

Normalized coordinates everywhere → resolution-independent, so proxy analysis maps onto the
full-res render with zero conversion bugs.

---

## 6. API design

```
POST   /api/v1/jobs                 multipart(file) | json{url}  → 202 {job_id, status}
GET    /api/v1/jobs                 list, paginated
GET    /api/v1/jobs/{id}            {status, stage, progress, error, degradations}
GET    /api/v1/jobs/{id}/events     SSE: stage / progress / log / done   ← live status
GET    /api/v1/jobs/{id}/project    project.json
DELETE /api/v1/jobs/{id}            cancel (revoke task) + purge storage
GET    /api/v1/jobs/{id}/files/{p}  artifact serving (path-traversal guarded)
POST   /api/v1/projects/{id}/render {edits} → 202 render job
POST   /api/v1/projects/{id}/assets replacement image for an overlay
GET    /healthz  /readyz            liveness + redis/ffmpeg/model checks
```

- `status ∈ queued | downloading | analyzing | inpainting | exporting | succeeded | failed | cancelled`
- Errors: RFC-7807 `application/problem+json` with typed codes (`E_DOWNLOAD_BLOCKED`, `E_TOO_LONG`,
  `E_NO_VIDEO_STREAM`, `E_OCR_FAILED`). Never leak stack traces.
- Idempotency: `Idempotency-Key` header + sha256 dedupe.
- SSE with a polling fallback (`GET /jobs/{id}`) — proxies eat SSE, the UI must survive that.

---

## 7. Frontend (deliberately thin)

Single page, four panels — functionality over polish, per the brief:

1. **Input** — URL field / drag-drop, options (removal method, "analyze with AI" toggle).
2. **Status** — SSE-driven stage stepper + progress bar + live log tail. This is what the demo video
   shows during the wait, so it has to look alive.
3. **Timeline** — horizontal scene strip (keyframe thumbnails, durations) with two lanes beneath:
   text tracks and overlay tracks positioned by time. Click a block → inspector.
4. **Preview / Inspector** — original vs. clean toggle (same `<video>`, swapped `src`, shared
   `currentTime` = instant A/B), per-scene clip player, editable caption/overlay list,
   "Rebuild video" button.

Types generated from `/openapi.json` via `openapi-typescript`, so the contract cannot drift.

---

## 8. Repo layout

```
lightnote-deedit/
├─ backend/
│  ├─ app/
│  │  ├─ main.py                 # FastAPI app, CORS, static, exception handlers
│  │  ├─ config.py               # pydantic-settings
│  │  ├─ api/v1/{jobs,projects,health}.py
│  │  ├─ schemas/{job,project}.py
│  │  ├─ db/{models,session}.py
│  │  ├─ worker/{celery_app,tasks}.py
│  │  ├─ pipeline/
│  │  │  ├─ context.py           # JobContext, storage paths, progress emitter
│  │  │  ├─ base.py              # Stage ABC: run(ctx), name, resumable
│  │  │  ├─ ingest.py probe.py scenes.py sample.py
│  │  │  ├─ ocr.py asr.py text_tracks.py
│  │  │  ├─ overlays.py masks.py inpaint.py
│  │  │  ├─ export.py compose.py
│  │  ├─ ai/
│  │  │  ├─ vlm.py               # provider-agnostic VLM interface
│  │  │  ├─ providers/{gemini,claude,noop}.py
│  │  │  └─ prompts/
│  │  └─ utils/{ffmpeg,geometry,color,logging}.py
│  ├─ tests/                     # unit: tracking, geometry, mask timeline; integration: 5s fixture
│  ├─ pyproject.toml  Dockerfile
├─ frontend/                     # Vite + React + TS
├─ docker-compose.yml            # api · worker · redis · frontend
├─ .env.example
└─ README.md
```

**Testability note:** the tracking / geometry / classification logic takes plain dicts, not video —
so `tests/` runs in milliseconds with no media required. That is where the code-quality points live.

---

## 9. Execution schedule (22h, checkpointed)

| Block | Hours | Deliverable | Done when |
|---|---|---|---|
| A | 0–1.5 | Repo scaffold, FastAPI + Celery + Redis + SQLite, `/healthz`, docker-compose | `POST /jobs` returns an id, worker logs it |
| B | 1.5–3.5 | S1–S3: yt-dlp + upload, ffprobe, PySceneDetect, scene clips + keyframes | scenes visible in `project.json` |
| C | 3.5–5.5 | S4–S5: sampling, RapidOCR, temporal text tracking, style extraction | text tracks with correct `t_in`/`t_out` |
| D | 5.5–7 | S8–S9 basic: mask timeline + OpenCV inpaint + `clean.mp4` export | **T1 backend complete, end-to-end** |
| E | 7–10 | Frontend: input, SSE status, scene timeline, A/B preview | **T1 SHIPPABLE — tag it, this is the safety net** |
| F | 10–12 | S7: overlay detection (3-signal CV ensemble) | overlay tracks appear on the timeline |
| G | 12–14 | S6 + VLM: faster-whisper, caption/overlay classification, Gemini scene descriptions and overlay adjudication | semantic labels visible in the UI |
| H | 14–16 | LaMa + median-plate removal; quality pass on clean output | visible quality jump in the A/B |
| I | 16–18 | Recompose: edit captions → rebuild; overlay image replacement | bonus items land |
| J | 18–20 | Error handling, cancellation, caching, tests, polish | tests green, bad URL fails gracefully |
| K | 20–22 | README, demo video, deploy (Render/Fly + Vercel) or run-locally instructions | submitted |

**Hour-12 gate:** if F–I have not started by hour 12, freeze features and go straight to J/K.
A complete, honest T1 submission beats a half-wired T3 one.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| yt-dlp blocked by TikTok/IG (auth walls, geo) | Upload path is first-class and demoed first; ship two sample videos in `samples/`; surface a clear `E_DOWNLOAD_BLOCKED` with an "upload instead" hint |
| Model downloads on first run are slow / demo is offline | Bake weights into the Docker image; `warmup()` on worker boot; `AI_ENABLED=false` degrades to pure CV |
| Inpainting too slow on CPU | Only inpaint masked frames; analyse at 720p proxy; hard per-job time budget with automatic tier downgrade (`lama → opencv → mask-blur`), recorded in `diagnostics.degradations` |
| Python 3.13 wheel breakage | Pin 3.11 in venv/Docker; ONNX-based OCR avoids PaddlePaddle entirely |
| VLM latency / cost / rate limits | One call per scene (not per frame), cap ~12 scenes, 20 s timeout, cache by frame hash, fully optional |
| Scope creep eats the deadline | The tier gates above; `git tag t1-complete` the moment block E lands |

---

## 11. Known limitations (state these up front in the README)

- Inpainting over **moving** backgrounds with large masks will smear; ProPainter is the fix but is
  GPU-bound and out of scope for this deadline.
- Overlay detection is tuned for hard-edged pasted assets — soft-alpha gradients, animated stickers
  and full-frame transitions will be missed or over-detected.
- Non-Latin scripts depend on the OCR model pack; only `en` ships enabled.
- Scene detection handles cuts, not dissolves or whip transitions (the adaptive detector reduces
  but does not eliminate this).
- Recompose approximates the original typography with the nearest bundled font; it does not
  identify the exact font.
- Single-node storage on the local filesystem; S3 plus a signed-URL layer is the obvious
  production swap.

---

## 12. Demo video script (2:30)

1. `0:00` One line on the model: *"de-editing = recovering the edit project."* Architecture slide, 5 s.
2. `0:15` Paste a TikTok URL → **De-edit**. Show the SSE stage stepper moving through
   download → scenes → OCR → inpaint.
3. `0:50` Result: scene timeline with thumbnails; text lane and overlay lane beneath.
4. `1:10` Click a caption → inspector shows text, timing, box, style, and the
   `caption` vs `overlay_text` classification (mention the ASR-alignment trick).
5. `1:30` A/B toggle: original ↔ clean plate at the same timestamp. Then a single scene clip.
6. `1:50` Edit a caption's text, swap the product overlay image, **Rebuild** → play the result.
7. `2:10` Show `/docs` (OpenAPI) and `project.json`; close on limitations and what production
   would change.

---

## 13. Submission checklist

- [ ] Video URL input **and** file upload
- [ ] Scene segmentation + per-scene clips
- [ ] Caption / text-overlay detection with timings, boxes, styles
- [ ] Image / product pop-up detection (+ RGBA cutouts)
- [ ] Real AI integration (OCR + ASR + VLM), explained and swappable
- [ ] FastAPI backend, all processing server-side, OpenAPI docs
- [ ] Job queue + live processing status
- [ ] Clean video preview + scene breakdown UI
- [ ] Bonus: caption editing, overlay replacement, rebuild
- [ ] GitHub repo, README (architecture / models / rationale / setup / limitations)
- [ ] 2–3 min demo video
- [ ] Deployed URL **or** verified `docker compose up` from a clean clone
