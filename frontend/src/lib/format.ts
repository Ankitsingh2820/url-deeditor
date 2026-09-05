export const cx = (...parts: (string | false | null | undefined)[]) =>
  parts.filter(Boolean).join(' ')

export function fmtTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return '--:--'
  const minutes = Math.floor(seconds / 60)
  const rest = seconds - minutes * 60
  return `${minutes}:${rest.toFixed(2).padStart(5, '0')}`
}

export function fmtDuration(seconds: number): string {
  return `${seconds.toFixed(2)}s`
}

export function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export function fmtBytes(bytes: number): string {
  return bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${(bytes / 1e3).toFixed(0)} kB`
}

/** Colour per detected element kind; shared by the timeline, chips and overlays. */
export const KIND_COLOR: Record<string, string> = {
  caption: '#6ee7b7',
  overlay_text: '#93c5fd',
  watermark: '#fca5a5',
  product_popup: '#fcd34d',
  sticker: '#fcd34d',
  logo: '#fca5a5',
}

export const kindColor = (kind: string) => KIND_COLOR[kind] ?? '#c4b5fd'

/** Turn `scene_detection_inconclusive` into `scene detection inconclusive`. */
export function humanise(token: string): string {
  return token.replace(/_/g, ' ').replace(/:/g, ': ')
}

export const STAGES = [
  'ingest',
  'probe',
  'scenes',
  'sample',
  'ocr',
  'asr',
  'text_tracks',
  'overlays',
  'vlm',
  'masks',
  'inpaint',
  'export',
  'finalize',
] as const
