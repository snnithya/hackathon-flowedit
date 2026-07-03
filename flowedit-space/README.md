---
title: FlowEdit
emoji: "\U0001F3B5"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.3.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# FlowEdit (Stable Audio 3)

Gradio interface for FlowEdit audio editing, running on a pinned build of the
`stable_audio_3` package.

## How the environment is pinned

- **Library:** `stable_audio_3` is installed from an immutable git tag in
  `requirements.txt` (`stable-audio-3[ui] @ git+https://github.com/snnithya/hackathon-flowedit.git@flowedit-v0.1.0`).
  Bump the tag there to upgrade.
- **Runtime:** `sdk_version: 6.3.0` and `python_version: "3.10"` in the header
  above match the dev environment (`pyproject.toml` / `.python-version`).
- **System libs:** `packages.txt` installs `libsndfile1` and `ffmpeg` for audio I/O.

## Configuration

Set these in the Space **Settings**:

- **Secret** `HF_TOKEN` — a token with access to the gated
  `stabilityai/stable-audio-3-*` models (accept each model's license first).
- **Hardware** — a GPU (e.g. A10G) is required for the default `medium` model.
- **Variable** `SAO_MODEL` (optional) — override the model, e.g. `small-music`
  to run on CPU hardware. Defaults to `medium`.

## Local run

```bash
pip install -r requirements.txt
export HF_TOKEN=...
python app.py
```
