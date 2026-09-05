import { useCallback, useEffect, useRef, useState } from 'react'
import { api, followJob } from './api/client'
import { TERMINAL, type Job, type JobEvent, type Project } from './api/types'
import { ComparePlayer } from './components/ComparePlayer'
import { Inspector } from './components/Inspector'
import { SceneTimeline, type Selection } from './components/SceneTimeline'
import { StatusPanel } from './components/StatusPanel'
import { SubmitPanel } from './components/SubmitPanel'
import { cx } from './lib/format'

const MAX_LOG_LINES = 200

export default function App() {
  const [job, setJob] = useState<Job | null>(null)
  const [events, setEvents] = useState<JobEvent[]>([])
  const [project, setProject] = useState<Project | null>(null)
  const [recent, setRecent] = useState<Job[]>([])
  const [selection, setSelection] = useState<Selection>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [seekTo, setSeekTo] = useState<{ t: number; nonce: number } | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [degraded, setDegraded] = useState(false)
  const stopRef = useRef<(() => void) | null>(null)

  const refreshList = useCallback(() => {
    api
      .listJobs()
      .then((page) => setRecent(page.items))
      .catch(() => undefined)
  }, [])

  useEffect(refreshList, [refreshList])

  // Always detach the previous stream before starting a new one, and on unmount.
  useEffect(() => () => stopRef.current?.(), [])

  const loadProject = useCallback(async (jobId: string) => {
    try {
      const loaded = await api.getProject(jobId)
      setProject(loaded)
      setSelection(loaded.scenes.length ? { type: 'scene', id: loaded.scenes[0].id } : null)
    } catch {
      setProject(null) // failed jobs have no project.json; the status panel explains why
    }
  }, [])

  const watch = useCallback(
    (target: Job) => {
      stopRef.current?.()
      setJob(target)
      setEvents([])
      setProject(null)
      setSelection(null)
      setCurrentTime(0)
      setDegraded(false)

      if (TERMINAL.includes(target.status)) {
        if (target.status === 'succeeded') void loadProject(target.id)
        return
      }

      setStreaming(true)
      stopRef.current = followJob(target.id, {
        onEvent: (event) => {
          setEvents((previous) => [...previous, event].slice(-MAX_LOG_LINES))
          setJob((previous) =>
            previous && previous.id === target.id
              ? { ...previous, status: event.status, stage: event.stage, progress: event.progress }
              : previous,
          )
        },
        onFallback: () => setDegraded(true),
        onDone: async () => {
          setStreaming(false)
          // Re-read the job for the authoritative timings/degradations, which
          // the event stream only carries in summary form.
          const finished = await api.getJob(target.id).catch(() => null)
          if (finished) setJob(finished)
          if (finished?.status === 'succeeded') await loadProject(target.id)
          refreshList()
        },
      })
    },
    [loadProject, refreshList],
  )

  const submitted = useCallback(
    (created: Job) => {
      refreshList()
      watch(created)
    },
    [refreshList, watch],
  )

  const cancel = useCallback(async () => {
    if (!job) return
    const cancelled = await api.cancel(job.id).catch(() => null)
    if (cancelled) setJob(cancelled)
  }, [job])

  const seek = useCallback((seconds: number) => {
    setCurrentTime(seconds)
    setSeekTo({ t: seconds, nonce: Date.now() })
  }, [])

  const busy = Boolean(job && !TERMINAL.includes(job.status))

  return (
    <div className="mx-auto max-w-[1500px] p-4 lg:p-6">
      <header className="mb-5 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold text-neutral-100">LightNote De-Editor</h1>
          <p className="text-xs text-neutral-500">
            Decompose a short-form video into scenes, captions, overlays — and a clean plate.
          </p>
        </div>
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          className="text-xs text-neutral-500 underline-offset-2 hover:text-emerald-300 hover:underline"
        >
          API docs ↗
        </a>
      </header>

      <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-4">
          <SubmitPanel onSubmitted={submitted} busy={busy} />
          <StatusPanel
            job={job}
            events={events}
            streaming={streaming}
            degraded={degraded}
            onCancel={cancel}
          />
          <RecentJobs jobs={recent} activeId={job?.id} onOpen={watch} />
        </div>

        <div className="space-y-4">
          {project ? (
            <>
              <ComparePlayer
                project={project}
                currentTime={currentTime}
                seekTo={seekTo}
                onTime={setCurrentTime}
              />
              <SceneTimeline
                project={project}
                currentTime={currentTime}
                selection={selection}
                onSelect={setSelection}
                onSeek={seek}
              />
              <Inspector project={project} selection={selection} />
            </>
          ) : (
            <Placeholder busy={busy} failed={job?.status === 'failed'} />
          )}
        </div>
      </div>
    </div>
  )
}

function RecentJobs({
  jobs,
  activeId,
  onOpen,
}: {
  jobs: Job[]
  activeId?: string
  onOpen: (job: Job) => void
}) {
  if (jobs.length === 0) return null
  return (
    <section className="rounded-xl border border-[#232936] bg-[#141821] p-4">
      <h2 className="mb-2 text-sm font-semibold tracking-wide text-neutral-400 uppercase">
        Recent
      </h2>
      <ul className="space-y-1">
        {jobs.map((entry) => (
          <li key={entry.id}>
            <button
              onClick={() => onOpen(entry)}
              className={cx(
                'flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs transition',
                entry.id === activeId ? 'bg-[#1b2130]' : 'hover:bg-[#1b2130]',
              )}
            >
              <span
                className={cx(
                  'h-1.5 w-1.5 shrink-0 rounded-full',
                  entry.status === 'succeeded' && 'bg-emerald-400',
                  entry.status === 'failed' && 'bg-red-400',
                  entry.status === 'cancelled' && 'bg-amber-400',
                  !TERMINAL.includes(entry.status) && 'animate-pulse bg-sky-400',
                )}
              />
              <span className="truncate text-neutral-300">{entry.source_ref}</span>
              <span className="ml-auto shrink-0 text-neutral-600">{entry.source_kind}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Placeholder({ busy, failed }: { busy: boolean; failed?: boolean }) {
  return (
    <section
      className="flex min-h-[420px] items-center justify-center rounded-xl border border-dashed
                 border-[#232936] p-10 text-center"
    >
      <div>
        <div className="mb-2 text-4xl">{failed ? '⚠' : busy ? '⏳' : '🎬'}</div>
        <p className="text-sm text-neutral-500">
          {failed
            ? 'This job failed — see the reason on the left.'
            : busy
              ? 'De-editing… the breakdown appears here when the pipeline finishes.'
              : 'Paste a video URL or drop a file to begin.'}
        </p>
        {!busy && !failed && (
          <p className="mt-2 text-xs text-neutral-700">
            Scenes, captions, text overlays and product pop-ups are detected, isolated and removed.
          </p>
        )}
      </div>
    </section>
  )
}
