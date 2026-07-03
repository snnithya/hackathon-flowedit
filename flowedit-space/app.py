"""Hugging Face Space entrypoint for FlowEdit (deterministic inverse + LFE only).

This is a self-contained copy of the standalone FlowEdit Gradio app, adapted to
run on a Hugging Face Space:
  - the model is loaded once at import time (Spaces has no CLI args),
  - the interface binds to the host/port that HF injects via env vars,
  - the optional /generate FastAPI route is mounted after launch.

The `stable_audio_3` package is installed from a pinned git tag (see
requirements.txt), so the FlowEdit inference code is version-locked.

Runs a single FlowEdit pass and shows NUM_COLS=10 outputs arranged in two
rows of COLS_PER_ROW=5:
  - (NUM_COLS - 1) = 9 intermediate latents decoded at steps chosen by:
      • t=0.95 forced (always the second step, regardless of sigma/center), and
      • 8 additional steps drawn from a Gaussian centred at `sample_center`
        (default 0.5) with spread `sample_std` along the t-schedule
        (t=1 noisy -> t=0 clean).
  - 1 final decoded output (rightmost column, second row).
"""

import gc
import json
import os

import gradio as gr
import numpy as np
import torch
import torchaudio
from einops import rearrange
from torchaudio import transforms as T

from stable_audio_3.inference.sampling import generate_diffusion_latent_flowedit
from stable_audio_3.interface.aeiou import audio_spectrogram_image
from stable_audio_3 import StableAudioModel
from scipy.stats import norm as scipy_norm


model = None
sample_rate = 44100
sample_size = 2097152
model_half = True

# Which pretrained model to load. Override via the SAO_MODEL Space variable.
# "medium" requires GPU hardware; "small-music"/"small-sfx" run on CPU.
PRETRAINED_NAME = os.environ.get("SAO_MODEL", "medium")

# Number of output columns: (NUM_COLS - 1) intermediates + 1 final.
# Rendered as two rows of COLS_PER_ROW each.
NUM_COLS = 10
COLS_PER_ROW = 5


def _pad(lst, n, fill=None):
    lst = list(lst)
    return lst[:n] if len(lst) >= n else lst + [fill] * (n - len(lst))


def _t_to_step(t: float, lfe_steps: int) -> int:
    """Convert a t-value in [0, 1] to the nearest LFE step index."""
    return int(np.clip(np.round((1.0 - t) * (lfe_steps - 1)), 0, lfe_steps - 1))


def _aligned_sample_size(num_samples: int) -> int:
    """Round an audio length (in samples) up to a size the autoencoder can handle.

    The latent length must be a multiple of the pretransform downsampling ratio, and
    for chunked encoder attention it must additionally align to chunk_size//stride in
    latent space. Result is clamped to the global `sample_size` cap.
    """
    ds_ratio = model.model.pretransform.downsampling_ratio

    target = ((int(num_samples) + ds_ratio - 1) // ds_ratio) * ds_ratio

    try:
        enc_cfg = model.model_config["model"]["pretransform"]["config"]["encoder"]["config"]
        chunk_size = enc_cfg.get("chunk_size", 32)
        stride = enc_cfg["strides"][0]
        latent_align = max(chunk_size // stride, 1)
        align = ds_ratio * latent_align
        target = ((target + align - 1) // align) * align
    except (KeyError, TypeError, IndexError):
        pass

    return int(min(max(target, ds_ratio), sample_size))


def _gaussian_step_indices(
    lfe_steps: int,
    n_samples: int,
    center: float = 0.5,
    std: float = 0.15,
    forced_t_values: tuple = (0.95,),
):
    """Return a sorted list of LFE step indices.

    Always includes one step per entry in `forced_t_values` (default t=0.95).
    The remaining (n_samples - len(forced)) indices are drawn via evenly-spaced
    Gaussian quantiles centred at `center` with spread `std`.

    The t-schedule runs t=1 (noisy) -> t=0 (clean).
    t -> step_index:  ind = round((1 - t) * (lfe_steps - 1))
    """
    if lfe_steps <= 0:
        return []

    # --- forced indices (always included) ---
    forced_set = set()
    for t in forced_t_values:
        forced_set.add(_t_to_step(float(np.clip(t, 0.0, 1.0)), lfe_steps))

    # --- Gaussian-sampled indices for the remaining slots ---
    n_gaussian = max(0, n_samples - len(forced_set))
    gaussian_set = set()
    if n_gaussian > 0:
        quantiles = np.linspace(1 / (n_gaussian + 1), n_gaussian / (n_gaussian + 1), n_gaussian)
        t_values = scipy_norm.ppf(quantiles, loc=center, scale=std)
        t_values = np.clip(t_values, 0.0, 1.0)
        for t in t_values:
            gaussian_set.add(_t_to_step(float(t), lfe_steps))

    return sorted(forced_set | gaussian_set)


def load_model(
    pretrained_name=None,
    device="cuda",
    in_model_half=False,
):
    global model, sample_rate, sample_size, model_half

    if pretrained_name is not None:
        model = StableAudioModel.from_pretrained(pretrained_name, device=device, model_half=in_model_half)
    else:
        raise ValueError("No pretrained model provided")

    print("Done loading model")
    return model


def _prepare_init_audio(init_audio_input):
    if init_audio_input is None:
        return None

    print(f"Preparing init audio: {init_audio_input}")
    in_sr, audio = init_audio_input

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio dtype: {audio.dtype}")

    if model_half:
        audio = audio.to(torch.float16)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1)

    if in_sr != sample_rate:
        resample_tf = T.Resample(in_sr, sample_rate).to(audio.device).to(audio.dtype)
        audio = resample_tf(audio)

    if audio.shape[-1] > sample_size:
        audio = audio[:, :sample_size]

    return (sample_rate, audio)


def _sampled_to_outputs(sampled, save_path=None, spec_figsize=(3, 2), max_len=None):
    if max_len is not None:
        sampled = sampled[..., :max_len]
    audio = rearrange(sampled, "b d n -> d (b n)").to(torch.float32).cpu()
    peak = audio.abs().max().clamp(min=1e-8)
    audio_int16 = audio.div(peak).clamp(-1, 1).mul(32767).to(torch.int16)

    if save_path is not None:
        torchaudio.save(save_path, audio_int16, sample_rate)

    spectrogram = audio_spectrogram_image(audio_int16, sample_rate=sample_rate, figsize=spec_figsize)
    audio_np = audio_int16.numpy().T
    return (sample_rate, audio_np), [spectrogram]


def _run_flowedit(
    src_conditioning_inputs,
    tar_conditioning_inputs,
    init_audio,
    device,
    seed,
    src_inv_cfg_scale,
    tar_inv_cfg_scale,
    lfe_steps,
    n_avg,
    intermediate_latents_steps,
    run_sample_size,
    batch_size=1,
):

    print(src_conditioning_inputs)
    sampled, intermediate_sampled = generate_diffusion_latent_flowedit(
        model=model.model,
        steps=int(lfe_steps),
        src_cfg_scale=float(src_inv_cfg_scale),
        tar_cfg_scale=float(tar_inv_cfg_scale),
        src_conditioning_inputs=src_conditioning_inputs,
        tar_conditioning_inputs=tar_conditioning_inputs,
        init_audio=init_audio,
        device=device,
        n_avg=int(n_avg),
        batch_size=batch_size,
        sample_size=run_sample_size,
        seed=int(seed),
        inv_steps=0,
        lfe_steps=int(lfe_steps),
        return_intermediate_latents=True,
        intermediate_latents_steps=intermediate_latents_steps,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return sampled, intermediate_sampled


def generate_edit(
    init_audio_input,
    src_inv_cfg_scale,
    tar_inv_cfg_scale,
    lfe_steps,
    n_avg,
    sample_center=0.5,
    sample_std=0.15,
    seed=-1,
    src_prompt="",
    tar_prompt="",
    seconds_total=30,
):
    batch_size = 1
    if init_audio_input is None:
        raise gr.Error("Please provide an init audio file.")

    def _to_float(v, default):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _to_int(v, default):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    src_inv_cfg_scale  = _to_float(src_inv_cfg_scale,  1.0)
    tar_inv_cfg_scale  = _to_float(tar_inv_cfg_scale,  5.0)
    lfe_steps          = _to_int(lfe_steps,             20)
    n_avg              = _to_int(n_avg,                 10)
    sample_center      = _to_float(sample_center,       0.5)
    sample_std         = _to_float(sample_std,          0.15)
    seconds_total      = _to_int(seconds_total,         30)

    device = next(model.model.parameters()).device

    # Prepare init audio first so we can size the whole run to the input's duration.
    print(f"Preparing init audio: {init_audio_input}")
    init_audio = _prepare_init_audio(init_audio_input)
    input_len = init_audio[1].shape[-1]
    run_sample_size = _aligned_sample_size(input_len)
    print(
        f"[gradio_flowedit] input_len={input_len} "
        f"({input_len / sample_rate:.2f}s) -> run_sample_size={run_sample_size} "
        f"({run_sample_size / sample_rate:.2f}s)"
    )

    src_conditioning_inputs, _ = model._build_conditioning_dicts(
                src_prompt, None, seconds_total, batch_size
            )
    tar_conditioning_inputs, _ = model._build_conditioning_dicts(
                tar_prompt, None, seconds_total, batch_size
            )

    latent_sample_size = run_sample_size // model.model.pretransform.downsampling_ratio
    io_channels = model.model.io_channels
    inpaint_mask = torch.zeros(batch_size, 1, latent_sample_size, device=device)
    inpaint_masked_input = torch.zeros(batch_size, io_channels, latent_sample_size, device=device)

    src_conditioning_inputs = model.model.conditioner(src_conditioning_inputs, device)
    src_conditioning_inputs["inpaint_mask"] = [inpaint_mask]
    src_conditioning_inputs["inpaint_masked_input"] = [inpaint_masked_input]
    src_conditioning_inputs = model.model.get_conditioning_inputs(src_conditioning_inputs)
    tar_conditioning_inputs = model.model.conditioner(tar_conditioning_inputs, device)
    tar_conditioning_inputs["inpaint_mask"] = [inpaint_mask]
    tar_conditioning_inputs["inpaint_masked_input"] = [inpaint_masked_input]
    tar_conditioning_inputs = model.model.get_conditioning_inputs(tar_conditioning_inputs)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    seed = int(seed) if seed and seed.strip() else -1
    if seed == -1:
        seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
    print(f"[gradio_flowedit] seed={seed}")

    # Compute which LFE step indices to capture (NUM_COLS - 1 intermediates).
    n_intermediates = NUM_COLS - 1
    step_indices = _gaussian_step_indices(
        lfe_steps=lfe_steps,
        n_samples=n_intermediates,
        center=sample_center,
        std=sample_std,
    )
    # Corresponding t-values for display (t = 1 - ind/(lfe_steps-1))
    t_labels = [round(1.0 - idx / max(lfe_steps - 1, 1), 3) for idx in step_indices]
    print(f"[gradio_flowedit] capturing steps {step_indices} (t={t_labels})")

    sampled, intermediate_sampled = _run_flowedit(
        src_conditioning_inputs=src_conditioning_inputs,
        tar_conditioning_inputs=tar_conditioning_inputs,
        init_audio=init_audio,
        device=device,
        seed=seed,
        src_inv_cfg_scale=src_inv_cfg_scale,
        tar_inv_cfg_scale=tar_inv_cfg_scale,
        lfe_steps=lfe_steps,
        n_avg=n_avg,
        intermediate_latents_steps=step_indices,
        run_sample_size=run_sample_size,
    )

    audio_list = []
    specs_list = []
    for col_idx, (intermediate, t_val) in enumerate(zip(intermediate_sampled, t_labels)):
        audio, specs = _sampled_to_outputs(
            intermediate,
            save_path=f"flowedit_intermediate_{col_idx}_t{t_val}.wav",
            max_len=input_len,
        )
        audio_list.append(audio)
        specs_list.append(specs)

    final_audio, final_specs = _sampled_to_outputs(
        sampled, save_path="flowedit_final.wav", max_len=input_len
    )
    audio_list.append(final_audio)
    specs_list.append(final_specs)

    del sampled
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pad to exactly NUM_COLS so return arity always matches the outputs list.
    audio_list = _pad(audio_list, NUM_COLS)
    specs_list = _pad(specs_list, NUM_COLS)

    # Labels for each column (intermediates + "Final")
    col_labels = [f"t={t}" for t in t_labels] + ["Final"]
    col_labels = _pad(col_labels, NUM_COLS, fill="")

    return (*audio_list, *specs_list, *col_labels)


def create_edit_ui(gradio_title=""):
    col_min_width = 160
    gallery_height = 120

    with gr.Blocks(theme=gr.themes.Base()) as ui:
        if gradio_title:
            gr.Markdown(f"### {gradio_title}")

        with gr.Row():
            with gr.Column(scale=6):
                src_prompt = gr.Textbox(label="Source prompt", placeholder="Describe the input audio")
                tar_prompt = gr.Textbox(label="Target prompt", placeholder="Describe the desired edit")
            generate_button = gr.Button("Generate", variant="primary", scale=1)

        with gr.Row(equal_height=False):
            with gr.Column():
                init_audio_input = gr.Audio(label="Init audio")
            with gr.Column():
                gr.Markdown("**CFG scales**")
                with gr.Row():
                    src_inv_cfg = gr.Slider(0.0, 25.0, value=1.0, step=0.1, label="src_inv_cfg_scale")
                    tar_inv_cfg = gr.Slider(0.0, 25.0, value=5.0, step=0.1, label="tar_inv_cfg_scale")

                with gr.Accordion("Edit params", open=False):
                    with gr.Row():
                        lfe_steps = gr.Slider(1, 200, value=20, step=1, label="fe_steps")
                        n_avg = gr.Slider(1, 20, value=10, step=1, label="n_avg")
                        seconds_total = gr.Slider(1, 300, value=30, step=20, label="seconds_total")
                    with gr.Row():
                        sample_center = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.01,
                            label="Intermediate sample center (t-value; 1=noisy, 0=clean)",
                        )
                        sample_std = gr.Slider(
                            0.01, 0.5, value=0.15, step=0.01,
                            label="Intermediate sample spread (sigma; small=clustered, large=spread)",
                        )
                    seed_textbox = gr.Textbox(label="Seed (-1 for random)", value="-1")

        gr.Markdown("### FlowEdit outputs (intermediates + final)")
        audio_outputs = []
        spec_galleries = []
        col_label_components = []
        for row_start in range(0, NUM_COLS, COLS_PER_ROW):
            with gr.Row():
                for i in range(row_start, min(row_start + COLS_PER_ROW, NUM_COLS)):
                    with gr.Column(scale=1, min_width=col_min_width):
                        col_label_components.append(gr.Markdown(value=f"col {i}"))
                        audio_outputs.append(gr.Audio(
                            interactive=False,
                            show_label=False,
                            container=False,
                        ))
                        spec_galleries.append(gr.Gallery(
                            show_label=False,
                            columns=1,
                            height=gallery_height,
                            object_fit="contain",
                            container=False,
                            preview=False,
                        ))

        generate_button.click(
            fn=generate_edit,
            inputs=[
                init_audio_input,
                src_inv_cfg,
                tar_inv_cfg,
                lfe_steps,
                n_avg,
                sample_center,
                sample_std,
                seed_textbox,
                src_prompt,
                tar_prompt,
                seconds_total,
            ],
            outputs=[
                *audio_outputs,
                *spec_galleries,
                *col_label_components,
            ],
            api_name="generate_edit",
        )

    return ui


def _mount_api(interface):
    """Mount /generate on the Gradio app's FastAPI instance.

    Mirrors the logic in generate_edit() but accepts/returns base64 audio
    so the FastAPI backend can call it through any tunnel without file serving.
    """
    import base64
    import io as _io
    import traceback as _tb

    from fastapi import Request
    from fastapi.responses import JSONResponse

    @interface.app.post("/generate")
    async def api_generate(request: Request):
        import asyncio
        try:
            body = await request.json()

            src_prompt   = body["src_prompt"]
            tar_prompt   = body["tar_prompt"]
            lfe_steps    = int(body.get("lfe_steps", 20))
            n_avg        = int(body.get("n_avg", 10))
            src_cfg      = float(body.get("src_lfe_cfg_scale", 1.0))
            tar_cfg      = float(body.get("tar_lfe_cfg_scale", 3.0))
            num_inter    = int(body.get("num_intermediates", 9))
            center       = float(body.get("sample_center", 0.5))
            std          = float(body.get("sample_std", 0.15))
            seed         = int(body.get("seed", -1))
            seconds_total = int(body.get("seconds_total", 30))

            # Decode audio and build init_audio_input tuple
            audio_bytes = base64.b64decode(body["audio_b64"])
            waveform, in_sr = torchaudio.load(_io.BytesIO(audio_bytes))
            # _prepare_init_audio expects (sr, numpy array) with shape (samples, channels)
            init_audio_input = (in_sr, waveform.numpy().T)

            device = next(model.model.parameters()).device
            batch_size = 1

            init_audio = _prepare_init_audio(init_audio_input)
            input_len = init_audio[1].shape[-1]
            run_sample_size = _aligned_sample_size(input_len)

            # Build conditioning (same as generate_edit)
            src_cond_raw, _ = model._build_conditioning_dicts(src_prompt, None, seconds_total, batch_size)
            tar_cond_raw, _ = model._build_conditioning_dicts(tar_prompt, None, seconds_total, batch_size)

            latent_sample_size = run_sample_size // model.model.pretransform.downsampling_ratio
            io_channels = model.model.io_channels
            inpaint_mask = torch.zeros(batch_size, 1, latent_sample_size, device=device)
            inpaint_masked_input = torch.zeros(batch_size, io_channels, latent_sample_size, device=device)

            src_cond = model.model.conditioner(src_cond_raw, device)
            src_cond["inpaint_mask"] = [inpaint_mask]
            src_cond["inpaint_masked_input"] = [inpaint_masked_input]
            src_cond = model.model.get_conditioning_inputs(src_cond)

            tar_cond = model.model.conditioner(tar_cond_raw, device)
            tar_cond["inpaint_mask"] = [inpaint_mask]
            tar_cond["inpaint_masked_input"] = [inpaint_masked_input]
            tar_cond = model.model.get_conditioning_inputs(tar_cond)

            if seed == -1:
                seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))

            step_indices = _gaussian_step_indices(
                lfe_steps=lfe_steps, n_samples=num_inter, center=center, std=std,
            )
            t_labels = [round(1.0 - idx / max(lfe_steps - 1, 1), 3) for idx in step_indices]

            sampled, intermediate_sampled = await asyncio.to_thread(
                _run_flowedit,
                src_conditioning_inputs=src_cond,
                tar_conditioning_inputs=tar_cond,
                init_audio=init_audio,
                device=device,
                seed=seed,
                src_inv_cfg_scale=src_cfg,
                tar_inv_cfg_scale=tar_cfg,
                lfe_steps=lfe_steps,
                n_avg=n_avg,
                intermediate_latents_steps=step_indices,
                run_sample_size=run_sample_size,
            )

            def to_b64(tensor):
                t = tensor[..., :input_len]
                audio = rearrange(t, "b d n -> d (b n)").to(torch.float32).cpu()
                peak = audio.abs().max().clamp(min=1e-8)
                audio_int16 = audio.div(peak).clamp(-1, 1).mul(32767).to(torch.int16)
                buf = _io.BytesIO()
                torchaudio.save(buf, audio_int16, sample_rate, format="wav")
                return base64.b64encode(buf.getvalue()).decode()

            return JSONResponse({
                "intermediates": [
                    {"label": f"t={t}", "audio_b64": to_b64(inter)}
                    for t, inter in zip(t_labels, intermediate_sampled)
                ],
                "final": {"label": "Final", "audio_b64": to_b64(sampled)},
                "seed": seed,
            })
        except Exception as e:
            tb = _tb.format_exc()
            print(f"[api] /generate ERROR:\n{tb}")
            return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)

    print("[api] /generate mounted on Gradio app")


# --- Space entrypoint: load once at import, then launch ---------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
load_model(pretrained_name=PRETRAINED_NAME, in_model_half=True, device=device)

demo = create_edit_ui(gradio_title=os.environ.get("SAO_TITLE", "FlowEdit"))
demo.queue()

# HF Spaces sets GRADIO_SERVER_NAME/GRADIO_SERVER_PORT; launch picks them up.
# prevent_thread_lock lets us mount the extra FastAPI route, then we block.
demo.launch(prevent_thread_lock=True)
_mount_api(demo)
demo.block_thread()
