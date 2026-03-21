"""Gradio demo for qualitative testing of sub-word causal language models (GPT-2, Pythia, etc.)."""

import argparse
import os
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AUTOCAST_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float16

STRATEGY_GREEDY = "Greedy"
STRATEGY_BEAM = "Beam search"
STRATEGY_SAMPLING = "Sampling (top-k / top-p)"


def load_model_and_tokenizer(model_path: str | Path):
    model_path = str(model_path)
    checkpoint_path = model_path
    if os.path.isdir(model_path):
        last_ckpt = get_last_checkpoint(model_path)
        if last_ckpt is not None:
            checkpoint_path = last_ckpt
            print(f"Using last checkpoint: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()

    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


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


@torch.inference_mode()
@torch.autocast(device_type=DEVICE, dtype=AUTOCAST_DTYPE, enabled=DEVICE != "cpu")
def generate(prompt, max_new_tokens, strategy, num_beams, top_k, top_p, temperature, repetition_penalty, model, tokenizer):
    if not prompt.strip():
        return ""

    gen_config = build_generation_config(strategy, num_beams, top_k, top_p, temperature, repetition_penalty)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": int(max_new_tokens),
    }
    if gen_config is not None:
        gen_kwargs["generation_config"] = gen_config

    output_ids = model.generate(**gen_kwargs)
    # Decode only the newly generated tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for sub-word causal LMs")
    parser.add_argument("model_path", type=str, help="Path to a local model/checkpoint directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    def on_generate(prompt, max_new_tokens, strategy, num_beams, top_k, top_p, temperature, repetition_penalty):
        return generate(
            prompt, max_new_tokens, strategy, num_beams, top_k, top_p, temperature, repetition_penalty, model, tokenizer)

    with gr.Blocks(title="CLM Demo") as demo:
        gr.Markdown("# Sub-word CLM – Qualitative Demo")
        gr.Markdown(f"**Model:** `{args.model_path}` &nbsp; **Device:** `{DEVICE}`")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Enter text here…")
                max_new_tokens = gr.Slider(minimum=1, maximum=512, value=64, step=1, label="Max new tokens")
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
                output = gr.Textbox(label="Generated output", lines=12, interactive=False)

        # Show/hide strategy-specific parameters
        def on_strategy_change(choice):
            return (
                gr.update(visible=choice == STRATEGY_BEAM),
                gr.update(visible=choice == STRATEGY_SAMPLING),
            )

        strategy.change(fn=on_strategy_change, inputs=strategy, outputs=[beam_params, sampling_params])

        all_inputs = [prompt, max_new_tokens, strategy, num_beams, top_k, top_p, temperature, repetition_penalty]
        btn.click(fn=on_generate, inputs=all_inputs, outputs=output)
        prompt.submit(fn=on_generate, inputs=all_inputs, outputs=output)

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
