import { useEffect, useRef } from 'react'
import type { Job, JobEvent, JobStatus } from '../api/types'
import { cx, fmtMs, humanise, STAGES } from '../lib/format'

interface Props {
  job: Job | null
  events: JobEvent[]
  streaming: boolean
  degraded: boolean
  onCancel: () => void
}

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'Queued',
  downloading: 'Fetching source',
  analyzing: 'Analyzing',
  inpainting: 'Removing elements',
  exporting: 'Exporting',
  succeeded: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export function StatusPanel({ job, events, streaming, degraded, onCancel }: Props) {
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [events.length])

  if (!job) {
    return (
      <section className="rounded-xl border border-[#232936] bg-[#141821] p-5">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-400 uppercase">
          2 · Processing
        </h2>
        <p className="mt-3 text-sm text-neutral-600">
          Submit a video to watch the pipeline run.
        </p>
      </section>
    )
  }

  const running = !['succeeded', 'failed', 'cancelled'].includes(job.status)
  const timings = job.stage_timings_ms ?? {}
  const currentIndex = STAGES.indexOf(job.stage as (typeof STAGES)[number])

  return (
    <section className="rounded-xl border border-[#232936] bg-[#141821] p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-neutral-400 uppercase">
          2 · Processing
        </h2>
        <span
          className={cx(
            'text-xs font-medium',
            job.status === 'succeeded' && 'text-emerald-300',
            job.status === 'failed' && 'text-red-300',
            job.status === 'cancelled' && 'text-amber-300',
            running && 'text-sky-300',
          )}
        >
          {STATUS_LABEL[job.status]}
        </span>
      </div>

      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#0f131b]">
        <div
          className={cx(
            'h-full rounded-full transition-[width] duration-300',
            job.status === 'failed' ? 'bg-red-400' : 'bg-emerald-400',
          )}
          style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
        />
      </div>

      {/* Stage stepper: each stage shows its measured time once it completes,
          which doubles as a profile of where the pipeline spends its budget. */}
      <ol className="mt-4 grid grid-cols-2 gap-x-3 gap-y-1 text-xs sm:grid-cols-3">
        {STAGES.map((stage, index) => {
          const done = timings[stage] !== undefined
          const active = job.stage === stage && running
          return (
            <li
              key={stage}
              className={cx(
                'flex items-center justify-between gap-2 rounded px-1.5 py-0.5',
                active && 'bg-sky-400/10 text-sky-300',
                done && !active && 'text-neutral-400',
                !done && !active && index > currentIndex && 'text-neutral-700',
              )}
            >
              <span className="truncate">{stage}</span>
              {done ? (
                <span className="shrink-0 tabular-nums text-neutral-600">
                  {fmtMs(timings[stage])}
                </span>
              ) : active ? (
                <span className="shrink-0 animate-pulse text-sky-400">···</span>
              ) : null}
            </li>
          )
        })}
      </ol>

      <div
        ref={logRef}
        className="mt-4 h-28 overflow-y-auto rounded-lg bg-[#0f131b] p-2 font-mono text-[11px]
                   leading-relaxed text-neutral-500"
      >
        {events.length === 0 && <div className="text-neutral-700">waiting for events…</div>}
        {events.map((event, i) => (
          <div key={`${event.id}-${i}`} className={cx(event.type === 'error' && 'text-red-400')}>
            <span className="text-neutral-700">
              {String(Math.round(event.progress * 100)).padStart(3)}%{' '}
            </span>
            {event.message ?? event.stage}
          </div>
        ))}
      </div>

      {degraded && (
        <p className="mt-2 text-[11px] text-amber-400/80">
          Event stream unavailable — falling back to polling.
        </p>
      )}
      {!streaming && running && !degraded && (
        <p className="mt-2 text-[11px] text-neutral-600">reconnecting…</p>
      )}

      {job.error_detail && (
        <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs">
          <div className="font-mono text-red-300">{job.error_code}</div>
          <div className="mt-1 text-neutral-300">{job.error_detail}</div>
        </div>
      )}

      {job.degradations?.length > 0 && (
        <details className="mt-3 text-xs">
          <summary className="cursor-pointer text-amber-400/80">
            {job.degradations.length} degradation{job.degradations.length > 1 ? 's' : ''}
          </summary>
          <ul className="mt-1.5 space-y-0.5 text-neutral-500">
            {job.degradations.map((item) => (
              <li key={item}>· {humanise(item)}</li>
            ))}
          </ul>
        </details>
      )}

      {running && (
        <button
          onClick={onCancel}
          className="mt-4 w-full rounded-lg border border-[#2b3242] py-1.5 text-xs text-neutral-400
                     transition hover:border-red-500/40 hover:text-red-300"
        >
          Cancel job
        </button>
      )}
    </section>
  )
}
