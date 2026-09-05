import { assetUrl } from '../api/client'
import type { Project, Scene, TextTrack } from '../api/types'
import { fmtDuration, fmtTime, kindColor } from '../lib/format'
import type { Selection } from './SceneTimeline'

interface Props {
  project: Project
  selection: Selection
}

/** Detail view for whatever is selected in the timeline. */
export function Inspector({ project, selection }: Props) {
  if (!selection) {
    return (
      <Panel title="Inspector">
        <p className="text-sm text-neutral-600">
          Select a scene or a detected element in the timeline.
        </p>
        <Summary project={project} />
      </Panel>
    )
  }

  if (selection.type === 'scene') {
    const scene = project.scenes.find((s) => s.id === selection.id)
    return scene ? <SceneDetail project={project} scene={scene} /> : null
  }

  if (selection.type === 'text') {
    const track = project.text_tracks.find((t) => t.id === selection.id)
    return track ? <TextDetail track={track} /> : null
  }

  const overlay = project.overlay_tracks.find((t) => t.id === selection.id)
  if (!overlay) return null
  return (
    <Panel title={`${overlay.id} · ${overlay.kind}`}>
      <Field label="label">{overlay.label ?? '—'}</Field>
      <Field label="time">
        {fmtTime(overlay.t_in)} → {fmtTime(overlay.t_out)}
      </Field>
      {overlay.asset && (
        <img
          src={assetUrl(project.job_id, overlay.asset)}
          alt={overlay.label ?? 'overlay'}
          className="mt-2 max-h-32 rounded border border-[#232936]"
        />
      )}
    </Panel>
  )
}

function SceneDetail({ project, scene }: { project: Project; scene: Scene }) {
  const original = assetUrl(project.job_id, scene.clip)
  const clean = assetUrl(project.job_id, scene.clean_clip)
  const inScene = project.text_tracks.filter((t) => t.scene_id === scene.id)

  return (
    <Panel title={`${scene.id} · scene ${scene.index + 1}/${project.scenes.length}`}>
      <Field label="time">
        {fmtTime(scene.t_in)} → {fmtTime(scene.t_out)} ({fmtDuration(scene.duration)})
      </Field>
      {scene.ai?.description && <Field label="ai">{scene.ai.description}</Field>}

      <div className="mt-3 grid grid-cols-2 gap-2">
        <ClipPreview label="original" src={original} />
        <ClipPreview label="de-edited" src={clean} />
      </div>

      {inScene.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-[10px] tracking-wide text-neutral-600 uppercase">
            text in this scene
          </div>
          <ul className="space-y-1">
            {inScene.map((track) => (
              <li key={track.id} className="flex items-center gap-2 text-xs">
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: kindColor(track.kind) }}
                />
                <span className="truncate text-neutral-300">{track.text}</span>
                <span className="ml-auto shrink-0 text-neutral-600">{track.kind}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}

function TextDetail({ track }: { track: TextTrack }) {
  return (
    <Panel title={`${track.id} · ${track.kind}`}>
      <div
        className="mb-3 rounded-lg border border-[#232936] p-3 text-center"
        style={{
          backgroundColor: track.style.has_bg_box
            ? (track.style.bg_color ?? '#0f131b')
            : '#0f131b',
        }}
      >
        <span
          style={{
            color: track.style.color,
            textShadow: track.style.stroke ? `0 0 3px ${track.style.stroke}` : undefined,
          }}
          className="text-lg font-semibold"
        >
          {track.text}
        </span>
      </div>

      <Field label="time">
        {fmtTime(track.t_in)} → {fmtTime(track.t_out)} (
        {fmtDuration(track.t_out - track.t_in)})
      </Field>
      <Field label="scene">{track.scene_id ?? '—'}</Field>
      <Field label="box">
        x {track.box.x.toFixed(3)} · y {track.box.y.toFixed(3)} · w {track.box.w.toFixed(3)} · h{' '}
        {track.box.h.toFixed(3)}
      </Field>
      <Field label="style">
        <span className="inline-flex items-center gap-1.5">
          <Swatch color={track.style.color} />
          {track.style.color}
          {track.style.stroke && (
            <>
              <span className="text-neutral-600">/ stroke</span>
              <Swatch color={track.style.stroke} />
            </>
          )}
          {track.style.has_bg_box && <span className="text-neutral-500">· plate</span>}
        </span>
      </Field>
      <Field label="confidence">
        {(track.confidence * 100).toFixed(0)}% over {track.samples} sampled frame
        {track.samples === 1 ? '' : 's'}
      </Field>
      <Field label="source">{track.source}</Field>
    </Panel>
  )
}

function Summary({ project }: { project: Project }) {
  const kinds = project.text_tracks.reduce<Record<string, number>>((acc, track) => {
    acc[track.kind] = (acc[track.kind] ?? 0) + 1
    return acc
  }, {})

  return (
    <dl className="mt-4 space-y-1.5 border-t border-[#232936] pt-3 text-xs">
      <Field label="duration">{fmtDuration(project.media.duration)}</Field>
      <Field label="scenes">{project.scenes.length}</Field>
      {Object.entries(kinds).map(([kind, count]) => (
        <Field key={kind} label={kind}>
          {count}
        </Field>
      ))}
      <Field label="masked">{(project.outputs.masked_fraction * 100).toFixed(0)}% of timeline</Field>
      <Field label="inpainted">{project.outputs.frames_inpainted} frames</Field>
    </dl>
  )
}

function ClipPreview({ label, src }: { label: string; src?: string }) {
  return (
    <div>
      <div className="mb-1 text-[10px] tracking-wide text-neutral-600 uppercase">{label}</div>
      {src ? (
        <video src={src} controls playsInline className="w-full rounded bg-black" />
      ) : (
        <div className="rounded bg-[#0f131b] p-4 text-center text-[11px] text-neutral-700">
          not produced
        </div>
      )}
    </div>
  )
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-[#232936] bg-[#141821] p-4">
      <h2 className="mb-3 text-sm font-semibold tracking-wide text-neutral-400 uppercase">
        {title}
      </h2>
      {children}
    </section>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 text-xs">
      <dt className="w-20 shrink-0 text-neutral-600">{label}</dt>
      <dd className="min-w-0 flex-1 break-words text-neutral-300">{children}</dd>
    </div>
  )
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      className="inline-block h-3 w-3 rounded-sm border border-[#2b3242]"
      style={{ backgroundColor: color }}
    />
  )
}
