import { useCallback, useRef, useState } from 'react'
import { api, type SubmitOptions } from '../api/client'
import { ApiError, type Job, type RemovalMethod } from '../api/types'
import { cx, fmtBytes } from '../lib/format'

interface Props {
  onSubmitted: (job: Job) => void
  busy: boolean
}

const METHODS: { value: RemovalMethod; label: string; hint: string }[] = [
  { value: 'auto', label: 'Auto', hint: 'Best tier installed (LaMa if present, else OpenCV)' },
  { value: 'opencv', label: 'OpenCV', hint: 'Telea inpainting — fast, no model' },
  { value: 'lama', label: 'LaMa', hint: 'Higher quality; falls back if unavailable' },
  { value: 'mask_blur', label: 'Blur', hint: 'Blur through the mask — always works' },
]

export function SubmitPanel({ onSubmitted, busy }: Props) {
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [method, setMethod] = useState<RemovalMethod>('auto')
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<{ detail: string; hint?: string } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const submit = useCallback(async () => {
    setError(null)
    const options: SubmitOptions = { removal_method: method }
    try {
      const job = file
        ? await api.submitFile(file, options)
        : await api.submitUrl(url.trim(), options)
      setUrl('')
      setFile(null)
      onSubmitted(job)
    } catch (err) {
      if (err instanceof ApiError) {
        setError({ detail: err.problem.detail, hint: err.problem.hint })
      } else {
        setError({ detail: String(err) })
      }
    }
  }, [file, url, method, onSubmitted])

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) {
      setFile(dropped)
      setUrl('')
    }
  }

  const canSubmit = !busy && (file !== null || url.trim().length > 0)

  return (
    <section className="rounded-xl border border-[#232936] bg-[#141821] p-5">
      <h2 className="mb-4 text-sm font-semibold tracking-wide text-neutral-400 uppercase">
        1 · Source
      </h2>

      <label className="mb-2 block text-xs text-neutral-500">Video URL</label>
      <input
        value={url}
        disabled={busy || file !== null}
        onChange={(e) => setUrl(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && canSubmit && submit()}
        placeholder="https://www.tiktok.com/@user/video/123…"
        className="w-full rounded-lg border border-[#2b3242] bg-[#0f131b] px-3 py-2 text-sm
                   outline-none placeholder:text-neutral-600 focus:border-emerald-500/60
                   disabled:opacity-40"
      />

      <div className="my-3 flex items-center gap-3 text-xs text-neutral-600">
        <div className="h-px flex-1 bg-[#232936]" /> or <div className="h-px flex-1 bg-[#232936]" />
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !busy && inputRef.current?.click()}
        className={cx(
          'cursor-pointer rounded-lg border border-dashed px-4 py-6 text-center text-sm transition',
          dragging ? 'border-emerald-400 bg-emerald-400/5' : 'border-[#2b3242] hover:border-[#3a4356]',
          busy && 'pointer-events-none opacity-40',
        )}
      >
        {file ? (
          <span className="text-emerald-300">
            {file.name} <span className="text-neutral-500">· {fmtBytes(file.size)}</span>
          </span>
        ) : (
          <span className="text-neutral-500">Drop a video file, or click to browse</span>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => {
          const chosen = e.target.files?.[0]
          if (chosen) {
            setFile(chosen)
            setUrl('')
          }
        }}
      />

      <div className="mt-4">
        <label className="mb-2 block text-xs text-neutral-500">Removal method</label>
        <div className="flex flex-wrap gap-1.5">
          {METHODS.map((option) => (
            <button
              key={option.value}
              title={option.hint}
              disabled={busy}
              onClick={() => setMethod(option.value)}
              className={cx(
                'rounded-md px-2.5 py-1 text-xs transition',
                method === option.value
                  ? 'bg-emerald-400/15 text-emerald-300 ring-1 ring-emerald-400/40'
                  : 'bg-[#1b2130] text-neutral-400 hover:text-neutral-200',
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={submit}
        disabled={!canSubmit}
        className="mt-5 w-full rounded-lg bg-emerald-400 py-2.5 text-sm font-semibold text-[#0b0d12]
                   transition hover:bg-emerald-300 disabled:cursor-not-allowed
                   disabled:bg-[#232936] disabled:text-neutral-500"
      >
        {busy ? 'Processing…' : 'De-edit'}
      </button>

      {error && (
        <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs">
          <div className="text-red-300">{error.detail}</div>
          {error.hint && <div className="mt-1 text-neutral-400">{error.hint}</div>}
        </div>
      )}

      {file && !busy && (
        <button
          onClick={() => setFile(null)}
          className="mt-2 w-full text-xs text-neutral-500 hover:text-neutral-300"
        >
          clear file
        </button>
      )}
    </section>
  )
}
