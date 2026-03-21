"""Gradio demo for qualitative testing of WeLT models."""

import argparse
import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from welt.attention import get_attention_mask_for_packed_sequence
from welt.model import WordLatentTransformerForCausalLM
from welt.processor import TextImageProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AUTOCAST_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float16

STRATEGY_GREEDY = "Greedy"
STRATEGY_BEAM = "Beam search"
STRATEGY_SAMPLING = "Sampling (top-k / top-p)"


def load_model_and_processor(model_path: str | Path):
    model_path = str(model_path)
    checkpoint_path = model_path
    if os.path.isdir(model_path):
        last_ckpt = get_last_checkpoint(model_path)
        if last_ckpt is not None:
            checkpoint_path = last_ckpt
            print(f"Using last checkpoint: {checkpoint_path}")

    model: WordLatentTransformerForCausalLM = (
        WordLatentTransformerForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map=DEVICE,
        )
    )
    model.enable_optimizations()
    model.eval()

    processor = TextImageProcessor.from_pretrained(model_path)

    print(f"Model loaded on {DEVICE}")
    return model, processor


def build_generation_config(strategy, num_beams, top_k, top_p, temperature, repetition_penalty):
    kwargs = {}

    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty

    if strategy == STRATEGY_BEAM:
        kwargs["num_beams"] = int(num_beams)
    elif strategy == STRATEGY_SAMPLING:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        if top_k > 0:
            kwargs["top_k"] = int(top_k)
        if top_p < 1.0:
            kwargs["top_p"] = top_p

    if not kwargs:
        return None
    return GenerationConfig(**kwargs)


def create_entropy_plot(entropies: list[float], byte_labels: list[str], prompt_byte_count: int = 0):
    """Create a matplotlib figure showing per-byte entropy."""
    if not entropies:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No bytes generated", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Bytes")
        ax.set_ylabel("Entropy (bits)")
        plt.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(max(8, len(entropies) * 0.35), 4))
    x = np.arange(len(entropies))
    max_ent = max(entropies) + 1e-8
    colors = plt.cm.RdYlGn_r(np.array(entropies) / max_ent)

    # Fade prompt bars to distinguish from generated
    if prompt_byte_count > 0:
        alphas = [0.4] * prompt_byte_count + [1.0] * (len(entropies) - prompt_byte_count)
        for i, (xi, h, c, a) in enumerate(zip(x, entropies, colors, alphas)):
            ax.bar(xi, h, color=c, alpha=a, edgecolor="none", width=0.8)
        # Separator line between prompt and generated
        ax.axvline(prompt_byte_count - 0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    else:
        ax.bar(x, entropies, color=colors, edgecolor="none", width=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(byte_labels, fontfamily="monospace", fontsize=8, rotation=0, ha="center")
    ax.set_ylabel("Entropy (bits)")
    ax.set_xlabel("Bytes (prompt | generated)")
    ax.set_title("Per-byte entropy")
    ax.set_xlim(-0.5, len(entropies) - 0.5)

    plt.tight_layout()
    return fig


@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=AUTOCAST_DTYPE, enabled=DEVICE != "cpu")
def generate(prompt, max_words, strategy, num_beams, top_k, top_p, temperature, repetition_penalty, model, processor):
    if not prompt.strip():
        return "", create_entropy_plot([], [])

    gen_config = build_generation_config(strategy, num_beams, top_k, top_p, temperature, repetition_penalty)

    words = processor.pretokenize(prompt)
    tokenized = processor.tokenize_words(words, device=DEVICE)
    input_images, input_images_dimensions = processor.render_texts(words, device=DEVICE)
    attention_mask = get_attention_mask_for_packed_sequence([len(words)], words=words)

    inputs = {
        "input_ids": tokenized.input_ids.unsqueeze(0),
        "input_attention_mask": tokenized.attention_mask.unsqueeze(0),
        "input_images": input_images.unsqueeze(0),
        "input_images_dimensions": input_images_dimensions.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0).to(device=DEVICE),
    }

    texts, entropies, byte_labels, prompt_byte_count = model.generate(
        **inputs,
        processor=processor,
        max_generated_words=int(max_words),
        bytes_generation_config=gen_config,
        return_entropy=True,
        prompt_words=words,
    )

    fig = create_entropy_plot(entropies, byte_labels, prompt_byte_count)
    return texts[0], fig


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for WeLT model")
    parser.add_argument("model_path", type=str, help="Path to a local WeLT model/checkpoint directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_path)

    def on_generate(prompt, max_words, strategy, num_beams, top_k, top_p, temperature, repetition_penalty):
        return generate(
            prompt, max_words, strategy, num_beams, top_k, top_p, temperature, repetition_penalty, model, processor)

    with gr.Blocks(title="WeLT Demo") as demo:
        gr.Markdown("# WeLT – Qualitative Demo")
        gr.Markdown(f"**Model:** `{args.model_path}` &nbsp; **Device:** `{DEVICE}`")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Enter text here…")
                max_words = gr.Slider(minimum=1, maximum=200, value=32, step=1, label="Max generated words")
                strategy = gr.Radio(
                    [STRATEGY_GREEDY, STRATEGY_BEAM, STRATEGY_SAMPLING],
                    value=STRATEGY_GREEDY, label="Decoding strategy")

                with gr.Group(visible=False) as beam_params:
                    num_beams = gr.Slider(minimum=2, maximum=16, value=4, step=1, label="Number of beams")

                with gr.Group(visible=False) as sampling_params:
                    temperature = gr.Slider(minimum=0.01, maximum=2.0, value=1.0, step=0.01, label="Temperature")
                    top_k = gr.Slider(minimum=0, maximum=200, value=50, step=1,
                                      label="Top-k (0 = disabled)")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.01,
                                      label="Top-p (1.0 = disabled)")

                repetition_penalty = gr.Slider(
                    minimum=1.0, maximum=3.0, value=1.0, step=0.05, label="Repetition penalty (1.0 = off)")

                btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output = gr.Textbox(label="Generated output", lines=8, interactive=False)
                entropy_plot = gr.Plot(label="Per-byte entropy")

        # Show/hide strategy-specific parameters
        def on_strategy_change(choice):
            return (
                gr.update(visible=choice == STRATEGY_BEAM),
                gr.update(visible=choice == STRATEGY_SAMPLING),
            )

        strategy.change(fn=on_strategy_change, inputs=strategy, outputs=[beam_params, sampling_params])

        all_inputs = [prompt, max_words, strategy, num_beams, top_k, top_p, temperature, repetition_penalty]
        btn.click(fn=on_generate, inputs=all_inputs, outputs=[output, entropy_plot])
        prompt.submit(fn=on_generate, inputs=all_inputs, outputs=[output, entropy_plot])

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
