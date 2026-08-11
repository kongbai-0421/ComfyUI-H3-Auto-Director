---
name: h3-auto-director-json
description: Generate strict JSON segment plans for the MiniMax H3 Auto Director ComfyUI plugin. Use when a user asks for H3 multi-segment prompts, durations, continuation settings, audio restart points, or multimodal reference assignments. Default to no references and output only a valid JSON array.
---

# H3 Auto Director JSON

Generate the `segments_json` value consumed by `H3AutoDirectorPlan`. Follow the
MiniMax H3 official full-reference prompt format, but return only the JSON
array. Do not add Markdown fences, explanations, headings, or comments outside
the JSON.

## Output Contract

Return a non-empty JSON array. Every array item must contain exactly these
plugin-facing fields unless the user explicitly requests another supported
field:

```json
{
  "prompt": "subject_definitions:\n...\n\nsummary:\n...\n\nretention_analysis:\n...\n\ndetailed_description:\n...\n\noverall_soundscape:\n...\n\nnon_diegetic_music:\n...",
  "duration": 4,
  "audio_restart": false,
  "continue_video": false,
  "references": []
}
```

Use these defaults:

- `duration`: `5` unless the user gives a duration. Keep every value between `4` and `15` seconds.
- `audio_restart`: `false`; set `true` only at a user-requested or clearly justified audio reset point.
- `continue_video`: `false` for the first segment; `true` for later segments unless the user requests independent scenes or disables continuation.
- `references`: use `[]` unless the user supplies an exact local file path for the asset. Never invent filenames, paths, URLs, reference images, videos, audio, labels, or numbers.
- Image references have no independent duration. Never emit `duration` or `image_duration` inside an image reference; the segment-level `duration` controls the complete generated clip. Video and audio metadata may retain `duration` only for display or source-media bookkeeping.

The plugin's reference arrays and ComfyUI sockets are zero-based, but MiniMax H3
prompt labels are one-based. Labels are scoped to each segment and are assigned
separately by type: `<Picture 1>`, `<Picture 2>`, `<Video 1>`, and `<Audio 1>`.
Do not carry a label from one segment into another unless that exact reference is
also present in the other segment. Preserve the user's explicit label-to-file
mapping exactly. If the user supplies paths but does not specify the material
numbering or mapping, stop and ask which file is `<Picture N>`, `<Video N>`, or
`<Audio N>`; do not infer numbering from filenames, upload order, descriptions,
attachments, or visual appearance.

For every referenced material, use its concrete material label in all six
sections: define it in `subject_definitions`, list it as its own line in
`retention_analysis`, and cite it at the point where it acts in `summary`,
`detailed_description`, `overall_soundscape`, and/or `non_diegetic_music` as
applicable. A `<Subject N>` may describe content derived from a material, but it
never replaces the corresponding `<Picture N>`, `<Video N>`, or `<Audio N>` label.
If a section has no applicable use for a material, state that relationship
explicitly instead of silently omitting the label.

A video reference creates an audio label only when the user explicitly confirms
that it has an audio track or supplies an already-confirmed mapping. Otherwise
do not mention a soundtrack label. When present, enabled video soundtracks are
numbered first in H3 presentation order, followed by standalone audio files;
disabling one removes its label and shifts later audio labels down. Do not invent
`has_audio` metadata. If the user provides no exact local file paths, every
segment's `references` must remain `[]`; if the prompt still refers to an
unidentified material, ask the user for its exact path and one-based label before
returning JSON. Do not create a reference entry from a visual description, an
attachment name, “reference” wording, an uploaded preview, a URL, or inferred
asset mapping.

## Prompt Requirements

Write every `prompt` in English except dialogue, lyrics, and visible scene text,
which must stay in their original language. Use the six sections in this exact
order:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

Use the official Ref2VA relationship markers in `retention_analysis`:
`fully_preserved`, `partially_preserved`, `attribute_transfer`, or
`weak_reference`. Use `fully_copy`, `partially_copy`, `reference`, or
`weak_reference` for audio entries.

If the user provides no local file paths and no material is requested, write a
text-only plan with `references: []` and no material labels. If the request
mentions any missing material, ask for its exact local path and explicit one-based
label before generating JSON. When references exist, every material label must be
defined before use, remain stable across all six sections, and state precisely
what is preserved versus transferred.

## Segment Detail Rule

Do not reduce a segment to a short action label such as “the character runs” or
“the character walks”, unless the user explicitly requests that minimal
description. Each ordinary segment must describe a complete mini-shot:

- composition, framing, subject position, and environment;
- the subject's appearance and continuity from the previous segment;
- an ordered action progression with at least two or three state changes;
- body direction, weight shift, hand/leg motion, and interaction with props when relevant;
- camera type, movement direction, amplitude, speed, and the intended ending pose;
- physical sound events that correspond to the visible motion;
- lighting, atmosphere, and important background continuity.

For a 4-6 second segment, describe an opening state, a developing action, a
peak or transition, and a final state. Use natural timing phrases such as
`during the first second`, `then`, `midway through the shot`, and `end on`.
Do not invent extra plot events merely to make a prompt longer. Preserve the
user's requested action, but expand its visual and temporal execution.

Every segment must be self-contained enough to run by itself while explicitly
continuing the previous segment when `continue_video` is true. The first
sentence of a continued segment should identify the previous ending state and
the next motion. Do not make all segments identical: vary the shot, movement
phase, camera behavior, and sound according to the user's story or supplied
timeline.

## Continuity and Audio

Keep identity, costume, props, lighting direction, environment, and camera
logic consistent across segments unless the user requests a change. Use
`audio_restart: true` at a deliberate musical, scene, or sound-design boundary;
otherwise describe continuous ambience and music across the cut.

When the user supplies no local audio file path, generate `overall_soundscape` and
`non_diegetic_music` descriptions but keep `references` empty. Do not add an
audio reference just because the prompt mentions music or sound effects.

## Validation Before Returning

Before emitting the result:

1. Confirm the result parses as one JSON array.
2. Confirm every segment has a non-empty `prompt`, valid duration, boolean audio and continuation flags, and a list-valued `references` field.
3. Confirm `references` is `[]` for every segment unless the user supplied exact local file paths and explicit material labels. Confirm every non-empty reference has an exact user-provided local file path, the user-confirmed one-based label, and a valid type (`image`, `video`, or `audio`). For video references, `video_audio_enabled` is optional and must be boolean when present; omit it for the default enabled behavior.
4. Confirm no segment exceeds 12 total references, 9 images, 3 videos, or 3 independent audios.
5. Confirm every prompt has all six sections in the required order.
6. Confirm no prose, Markdown fence, or trailing comma appears outside the JSON.

If the required material path or numbering clarification is missing, do not
return a partial JSON array. Ask the user a concise clarification question first.

If the user supplies only a story idea and no segment count, infer a sensible
number of 4-6 second segments from the requested duration, but do not add
references. If the user explicitly asks for a deliberately simple action, obey
that request while still identifying the shot, subject, camera, timing, and
ending state.
