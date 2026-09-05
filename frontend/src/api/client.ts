/**
 * Backend client.
 *
 * Two things worth noting:
 *  - errors come back as RFC-7807 problem+json, so they are unwrapped into a
 *    typed ApiError the UI can branch on (E_DOWNLOAD_BLOCKED -> "upload
 *    instead") instead of matching on message strings.
 *  - progress uses SSE with a polling fallback. Proxies and corporate networks
 *    drop event streams; the UI must keep working when that happens.
 */

import { ApiError, type Job, type JobEvent, type Project, type ProblemDetail } from './types'

const API = '/api/v1'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) {
    let problem: ProblemDetail
    try {
      problem = (await response.json()) as ProblemDetail
    } catch {
      problem = {
        code: 'E_NETWORK',
        title: 'Request failed',
        detail: `${response.status} ${response.statusText}`,
        status: response.status,
      }
    }
    throw new ApiError(problem)
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

export interface SubmitOptions {
  removal_method?: string
  sample_fps?: number
  force?: boolean
}

export const api = {
  submitUrl(url: string, options: SubmitOptions = {}): Promise<Job> {
    return request<Job>('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, options }),
    })
  },

  submitFile(file: File, options: SubmitOptions = {}): Promise<Job> {
    const form = new FormData()
    form.append('file', file)
    form.append('options', JSON.stringify(options))
    // No Content-Type header: the browser must set the multipart boundary.
    return request<Job>('/jobs', { method: 'POST', body: form })
  },

  getJob(id: string): Promise<Job> {
    return request<Job>(`/jobs/${id}`)
  },

  listJobs(limit = 12): Promise<{ items: Job[]; total: number }> {
    return request(`/jobs?limit=${limit}`)
  },

  getProject(id: string): Promise<Project> {
    return request<Project>(`/jobs/${id}/project`)
  },

  cancel(id: string): Promise<Job> {
    return request<Job>(`/jobs/${id}/cancel`, { method: 'POST' })
  },

  remove(id: string): Promise<void> {
    return request<void>(`/jobs/${id}`, { method: 'DELETE' })
  },
}

/** Absolute URL for a workspace-relative artifact path. */
export function assetUrl(jobId: string, relative: string | null | undefined): string | undefined {
  return relative ? `/media/${jobId}/${relative}` : undefined
}

export interface StreamHandlers {
  onEvent: (event: JobEvent) => void
  onDone: (status: string) => void
  onFallback?: () => void
}

/**
 * Follow a job to completion.
 *
 * Returns a disposer. If the event stream fails before completing, this
 * degrades to polling rather than leaving the UI frozen.
 */
export function followJob(jobId: string, handlers: StreamHandlers): () => void {
  let closed = false
  let poller: number | undefined
  const source = new EventSource(`${API}/jobs/${jobId}/events`)

  const stop = () => {
    closed = true
    source.close()
    if (poller) window.clearInterval(poller)
  }

  const handle = (raw: MessageEvent) => {
    if (closed) return
    try {
      handlers.onEvent(JSON.parse(raw.data) as JobEvent)
    } catch {
      /* a malformed frame is not worth tearing the stream down for */
    }
  }

  source.addEventListener('progress', handle)
  source.addEventListener('log', handle)
  source.addEventListener('error', handle as EventListener)

  source.addEventListener('done', (raw) => {
    if (closed) return
    stop()
    handlers.onDone((raw as MessageEvent).data)
  })

  // EventSource reports transport failures on 'error' too; distinguish by
  // readyState, then fall back to polling.
  source.onerror = () => {
    if (closed || source.readyState !== EventSource.CLOSED) return
    handlers.onFallback?.()
    source.close()
    poller = window.setInterval(async () => {
      try {
        const job = await api.getJob(jobId)
        handlers.onEvent({
          id: 0,
          type: 'progress',
          stage: job.stage,
          progress: job.progress,
          message: job.stage,
          status: job.status,
        })
        if (['succeeded', 'failed', 'cancelled'].includes(job.status)) {
          stop()
          handlers.onDone(job.status)
        }
      } catch {
        /* keep polling; a transient failure should not kill the view */
      }
    }, 1000)
  }

  return stop
}
