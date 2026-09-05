/**
 * Wire types, mirroring the backend contract.
 *
 * `Project` is the `project.json` documented in plan.md §5 -- the editable
 * deliverable, not just a status payload. Every box is normalized 0..1 of the
 * display frame, so the UI can position overlays without knowing the video's
 * resolution.
 */

export type JobStatus =
  | 'queued'
  | 'downloading'
  | 'analyzing'
  | 'inpainting'
  | 'exporting'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export const TERMINAL: JobStatus[] = ['succeeded', 'failed', 'cancelled']

export type RemovalMethod = 'auto' | 'opencv' | 'lama' | 'mask_blur'
export type TextKind = 'caption' | 'overlay_text' | 'watermark'

export interface Job {
  id: string
  status: JobStatus
  stage: string | null
  progress: number
  source_kind: 'url' | 'upload'
  source_ref: string
  error_code: string | null
  error_detail: string | null
  degradations: string[]
  stage_timings_ms: Record<string, number>
  created_at: string
  finished_at: string | null
}

export interface Box {
  x: number
  y: number
  w: number
  h: number
}

export interface Scene {
  id: string
  index: number
  t_in: number
  t_out: number
  duration: number
  keyframe: string | null
  clip: string | null
  clean_clip: string | null
  ai: { description?: string; shot_type?: string; tags?: string[] } | null
}

export interface TextStyle {
  color: string
  stroke: string | null
  bg_color: string | null
  has_bg_box: boolean
  font_px_norm: number
  align: 'left' | 'center' | 'right'
  contrast: number
  confidence: number
}

export interface TextTrack {
  id: string
  kind: TextKind
  /** Why that kind was chosen: matches_speech | not_spoken | position_heuristic | ... */
  kind_reason?: string
  /** 0-100 fuzzy match between this text and what was spoken while it was up. */
  spoken_match?: number
  text: string
  t_in: number
  t_out: number
  scene_id: string | null
  box: Box
  style: TextStyle
  confidence: number
  source: string
  samples: number
  editable: boolean
}

export interface OverlayTrack {
  id: string
  kind: string
  label: string | null
  /** The VLM judged this region to be part of the filmed scene, not a graphic. */
  rejected?: boolean
  vlm_confidence?: number
  signals?: Record<string, number>
  t_in: number
  t_out: number
  scene_id: string | null
  box: Box
  asset: string | null
  confidence: number
  source: string
}

export interface Project {
  schema_version: string
  job_id: string
  source: { kind: string; ref: string; sha256: string | null }
  media: {
    duration: number
    fps: number
    width: number
    height: number
    has_audio: boolean
    proxy: { path: string; width: number; height: number; fps: number } | null
  }
  scenes: Scene[]
  text_tracks: TextTrack[]
  overlay_tracks: OverlayTrack[]
  transcript: { t_in: number; t_out: number; text: string }[]
  outputs: {
    clean_video: string | null
    removal_method: string | null
    source_video: string | null
    audio: string | null
    mask_timeline: string | null
    frames_inpainted: number
    masked_fraction: number
  }
  diagnostics: {
    stage_timings_ms: Record<string, number>
    degradations: string[]
    options: Record<string, unknown>
  }
}

/** One SSE frame from /jobs/{id}/events. */
export interface JobEvent {
  id: number
  type: 'progress' | 'log' | 'error' | 'done'
  stage: string | null
  progress: number
  message: string | null
  status: JobStatus
}

/** RFC-7807 problem document. */
export interface ProblemDetail {
  code: string
  title: string
  detail: string
  status: number
  hint?: string
}

export class ApiError extends Error {
  constructor(readonly problem: ProblemDetail) {
    super(problem.detail || problem.title)
  }
}
