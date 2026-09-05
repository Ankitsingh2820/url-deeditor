import { assetUrl } from '../api/client'
import type { Project } from '../api/types'
import { cx, fmtDuration, kindColor } from '../lib/format'

export type Selection =
  | { type: 'scene'; id: string }
  | { type: 'text'; id: string }
  | { type: 'overlay'; id: string }
  | null

interface Props {
  project: Project
  currentTime: number
  selection: Selection
  onSelect: (selection: Selection) => void
  onSeek: (seconds: number) => void
}

/**
 * Visual breakdown of the de-edited video.
 *
 * Three lanes over one shared time axis -- scenes, text tracks, overlay tracks.
 * Because every track carries absolute t_in/t_out, positioning is just
 * percentage maths: no layout engine, and the lanes stay aligned with the
 * player's playhead.
 */
export function SceneTimeline({ project, currentTime, selection, onSelect, onSeek }: Props) {
  const duration = project.media.duration || 1
  const pct = (seconds: number) => `${(seconds / duration) * 100}%`

  const seekFromClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    onSeek(((event.clientX - bounds.left) / bounds.width) * duration)
  }

  const ticks = Array.from({ length: Math.min(11, Math.ceil(duration) + 1) }, (_, i) =>
    (duration / Math.min(10, Math.ceil(duration))) * i,
  )

  return (
    <section className="rounded-xl border border-[#232936] bg-[#141821] p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-400 uppercase">
          Timeline
        </h2>
        <span className="text-xs text-neutral-600">
          {project.scenes.length} scenes · {project.text_tracks.length} text ·{' '}
          {project.overlay_tracks.length} overlays
        </span>
      </div>

      <div className="relative select-none">
        {/* playhead spans every lane */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 z-20 w-px bg-emerald-300"
          style={{ left: pct(currentTime) }}
        >
          <div className="-ml-1 h-2 w-2 rounded-full bg-emerald-300" />
        </div>

        <Lane label="scenes" onClick={seekFromClick} height="h-16">
          {project.scenes.map((scene) => {
            const thumb = assetUrl(project.job_id, scene.keyframe)
            const active = selection?.type === 'scene' && selection.id === scene.id
            return (
              <button
                key={scene.id}
                title={`${scene.id} · ${fmtDuration(scene.duration)}`}
                onClick={(e) => {
                  e.stopPropagation()
                  onSelect({ type: 'scene', id: scene.id })
                  onSeek(scene.t_in)
                }}
                style={{
                  left: pct(scene.t_in),
                  width: pct(scene.t_out - scene.t_in),
                  backgroundImage: thumb ? `url(${thumb})` : undefined,
                }}
                className={cx(
                  'absolute inset-y-0 overflow-hidden rounded border bg-cover bg-center',
                  'transition hover:brightness-125',
                  active ? 'border-emerald-300 ring-1 ring-emerald-300' : 'border-[#2b3242]',
                )}
              >
                <span
                  className="absolute inset-x-0 bottom-0 truncate bg-black/60 px-1 text-[10px]
                             text-neutral-300"
                >
                  {scene.id}
                </span>
              </button>
            )
          })}
        </Lane>

        <Lane label="text" onClick={seekFromClick}>
          {project.text_tracks.map((track) => {
            const active = selection?.type === 'text' && selection.id === track.id
            return (
              <button
                key={track.id}
                title={`${track.kind}: ${track.text}`}
                onClick={(e) => {
                  e.stopPropagation()
                  onSelect({ type: 'text', id: track.id })
                  onSeek(track.t_in + 0.05)
                }}
                style={{
                  left: pct(track.t_in),
                  width: pct(Math.max(track.t_out - track.t_in, duration * 0.004)),
                  backgroundColor: `${kindColor(track.kind)}${active ? 'ee' : '55'}`,
                  borderColor: kindColor(track.kind),
                }}
                className="absolute inset-y-0.5 overflow-hidden rounded border px-1 text-left
                           text-[10px] whitespace-nowrap text-[#0b0d12] transition
                           hover:brightness-110"
              >
                {track.text}
              </button>
            )
          })}
        </Lane>

        <Lane label="overlays" onClick={seekFromClick}>
          {project.overlay_tracks.length === 0 && (
            <span className="absolute inset-y-0 left-2 flex items-center text-[10px] text-neutral-700">
              no image/product pop-ups detected
            </span>
          )}
          {project.overlay_tracks.map((track) => {
            const active = selection?.type === 'overlay' && selection.id === track.id
            return (
              <button
                key={track.id}
                title={`${track.kind}: ${track.label ?? ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  onSelect({ type: 'overlay', id: track.id })
                  onSeek(track.t_in + 0.05)
                }}
                style={{
                  left: pct(track.t_in),
                  width: pct(Math.max(track.t_out - track.t_in, duration * 0.004)),
                  backgroundColor: `${kindColor(track.kind)}${active ? 'ee' : '55'}`,
                  borderColor: kindColor(track.kind),
                }}
                className="absolute inset-y-0.5 overflow-hidden rounded border px-1 text-left
                           text-[10px] whitespace-nowrap text-[#0b0d12]"
              >
                {track.label ?? track.kind}
              </button>
            )
          })}
        </Lane>

        <div className="relative mt-1 ml-16 h-4">
          {ticks.map((t) => (
            <span
              key={t}
              style={{ left: `${(t / duration) * 100}%` }}
              className="absolute -translate-x-1/2 text-[10px] text-neutral-600 tabular-nums"
            >
              {t.toFixed(1)}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}

function Lane({
  label,
  children,
  height = 'h-6',
  onClick,
}: {
  label: string
  children: React.ReactNode
  height?: string
  onClick: (event: React.MouseEvent<HTMLDivElement>) => void
}) {
  return (
    <div className="mb-1 flex items-stretch gap-2">
      <span className="w-14 shrink-0 pt-1 text-right text-[10px] text-neutral-600">{label}</span>
      <div
        onClick={onClick}
        className={cx('relative flex-1 cursor-crosshair rounded bg-[#0f131b]', height)}
      >
        {children}
      </div>
    </div>
  )
}
