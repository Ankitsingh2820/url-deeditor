import { useEffect, useRef, useState } from 'react'
import { assetUrl } from '../api/client'
import type { Project } from '../api/types'
import { cx, fmtTime, kindColor } from '../lib/format'

interface Props {
  project: Project
  currentTime: number
  seekTo: { t: number; nonce: number } | null
  onTime: (seconds: number) => void
}

/**
 * Original ↔ de-edited A/B preview.
 *
 * One <video> whose src is swapped, with the playhead carried across the swap.
 * Two stacked players would double the decode cost and drift out of sync; this
 * way the comparison is at an identical timestamp by construction, which is the
 * whole point of the view.
 */
export function ComparePlayer({ project, currentTime, seekTo, onTime }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [showClean, setShowClean] = useState(true)
  const [showBoxes, setShowBoxes] = useState(true)

  const jobId = project.job_id
  const original = assetUrl(jobId, project.outputs.source_video)
  const clean = assetUrl(jobId, project.outputs.clean_video)
  const hasClean = Boolean(clean)
  const src = showClean && hasClean ? clean : original

  // Keep the timestamp when swapping sources, so A/B compares the same frame.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !src) return
    const resume = !video.paused
    const at = video.currentTime
    video.src = src
    video.load()
    const restore = () => {
      video.currentTime = at
      if (resume) void video.play()
      video.removeEventListener('loadedmetadata', restore)
    }
    video.addEventListener('loadedmetadata', restore)
  }, [src])

  useEffect(() => {
    if (seekTo && videoRef.current) videoRef.current.currentTime = seekTo.t
  }, [seekTo])

  const activeText = project.text_tracks.filter(
    (t) => currentTime >= t.t_in && currentTime < t.t_out,
  )
  const activeOverlays = project.overlay_tracks.filter(
    (t) => currentTime >= t.t_in && currentTime < t.t_out,
  )

  return (
    <section className="rounded-xl border border-[#232936] bg-[#141821] p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-lg bg-[#0f131b] p-0.5 text-xs">
          <button
            onClick={() => setShowClean(false)}
            className={cx(
              'rounded-md px-3 py-1 transition',
              !showClean ? 'bg-[#232936] text-neutral-100' : 'text-neutral-500',
            )}
          >
            Original
          </button>
          <button
            onClick={() => setShowClean(true)}
            disabled={!hasClean}
            title={hasClean ? undefined : 'No clean plate was produced for this job'}
            className={cx(
              'rounded-md px-3 py-1 transition',
              showClean && hasClean ? 'bg-emerald-400/15 text-emerald-300' : 'text-neutral-500',
              !hasClean && 'cursor-not-allowed opacity-40',
            )}
          >
            De-edited
          </button>
        </div>

        <label className="flex items-center gap-1.5 text-xs text-neutral-500">
          <input
            type="checkbox"
            checked={showBoxes}
            onChange={(e) => setShowBoxes(e.target.checked)}
            className="accent-emerald-400"
          />
          show detections
        </label>
      </div>

      <div className="relative mx-auto max-h-[52vh] w-fit overflow-hidden rounded-lg bg-black">
        <video
          ref={videoRef}
          controls
          playsInline
          className="max-h-[52vh] max-w-full"
          onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
          onSeeked={(e) => onTime(e.currentTarget.currentTime)}
        />

        {/* Detected elements, drawn from normalized coordinates -- the reason
            boxes line up regardless of the rendered player size. */}
        {showBoxes &&
          [...activeText, ...activeOverlays].map((track) => (
            <div
              key={track.id}
              style={{
                left: `${track.box.x * 100}%`,
                top: `${track.box.y * 100}%`,
                width: `${track.box.w * 100}%`,
                height: `${track.box.h * 100}%`,
                borderColor: kindColor(track.kind),
              }}
              className="pointer-events-none absolute border-2 border-dashed"
            >
              <span
                style={{ backgroundColor: kindColor(track.kind) }}
                className="absolute -top-4 left-0 rounded-sm px-1 text-[9px] font-medium
                           whitespace-nowrap text-[#0b0d12]"
              >
                {track.id} · {track.kind}
              </span>
            </div>
          ))}
      </div>

      <div className="mt-2 flex items-center justify-between text-xs text-neutral-500">
        <span className="tabular-nums">
          {fmtTime(currentTime)} / {fmtTime(project.media.duration)}
        </span>
        <span>
          {project.media.width}×{project.media.height} · {project.media.fps.toFixed(2)} fps
          {project.outputs.removal_method && showClean && hasClean && (
            <span className="ml-2 text-emerald-400/70">
              removal: {project.outputs.removal_method}
            </span>
          )}
        </span>
      </div>
    </section>
  )
}
