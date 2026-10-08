"""
WeLT generation server: an exported model on one GPU, served over HTTP (run in the container, one per GPU).

    welt-serve <export_dir> [--port 8080]

GET  /health   -> {"status": "ok", "version": MODEL_TAG}
POST /generate {"texts": [...], "max_generated_words": 50, "temperature": 0.0, "seed": null}
               -> {"outputs": [...], "generated_words": N}
One generation runs at a time (vLLM batches all texts of a request); a busy server answers 503 with Retry-After.
`generate()` is the client, used by welt-generate (`welt-generate "<prompt>" ... --url URL`) and welt-evaluate.
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

from flask import Flask, jsonify, request

MODEL_TAG_HEADER = "X-Model-Tag"  # Which model revision served a response, from the MODEL_TAG environment variable
EXPORT_DIR_ENV = "WELT_EXPORT_DIR"


def create_app(generator=None, model_tag: str | None = None) -> Flask:
    """generator: a WeLTGenerator, by default of the export directory in WELT_EXPORT_DIR."""
    if generator is None:
        from welt.inference import WeLTGenerator  # vLLM, only in the serving process
        generator = WeLTGenerator(os.environ[EXPORT_DIR_ENV])
    model_tag = model_tag if model_tag is not None else os.environ.get("MODEL_TAG")
    busy = threading.Lock()  # vLLM's offline engines run one call at a time

    app = Flask(__name__)

    @app.after_request
    def add_model_tag(response):
        if model_tag:
            response.headers[MODEL_TAG_HEADER] = model_tag
        return response

    @app.get("/health")
    def health():
        return jsonify(status="ok", version=model_tag)

    @app.post("/generate")
    def generate_route():
        body = request.get_json(silent=True)
        texts = body.get("texts") if isinstance(body, dict) else None
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            return jsonify(message="'texts' must be a list of strings"), 400
        try:
            options = dict(max_generated_words=int(body.get("max_generated_words", 50)),
                           temperature=float(body.get("temperature", 0.0)),
                           seed=None if body.get("seed") is None else int(body["seed"]))
        except (TypeError, ValueError) as error:
            return jsonify(message=f"Invalid option: {error}"), 400
        if not busy.acquire(blocking=False):
            return jsonify(message="Busy"), 503, {"Retry-After": "1"}
        try:
            outputs = generator.generate(texts, **options)
        except ValueError as error:  # e.g. a context longer than the model's
            return jsonify(message=str(error)), 400
        finally:
            busy.release()
        generated_words = sum(len(generator.processor.pretokenize(output)) - 1 for output in outputs)  # Without BOS
        return jsonify(outputs=outputs, generated_words=generated_words)

    return app


def generate(url: str, texts: list[str], max_generated_words: int = 50, temperature: float = 0.0,
             seed: int | None = None, timeout: float = 3600) -> dict:
    """Generate with a running welt-serve at url: {"outputs": [...], "generated_words": N}. Waits while busy."""
    payload = json.dumps({"texts": texts, "max_generated_words": max_generated_words,
                          "temperature": temperature, "seed": seed}).encode()
    while True:
        http_request = urllib.request.Request(f"{url.rstrip('/')}/generate", data=payload,
                                              headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 503:
                raise RuntimeError(f"welt-serve answered {error.code}: {error.read().decode()}") from error
            time.sleep(float(error.headers.get("Retry-After", 1)))


def generate_main():
    parser = argparse.ArgumentParser(description="Generate with a running welt-serve.")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--max_generated_words", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    outputs = generate(args.url, args.prompts, args.max_generated_words, args.temperature)["outputs"]
    for prompt, output in zip(args.prompts, outputs, strict=True):
        print(repr(prompt), "->", repr(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", help="Directory exported by welt-export")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    args = parser.parse_args()
    os.environ[EXPORT_DIR_ENV] = os.path.abspath(args.export_dir)
    # One worker holds the model (and the GPU); threads answer health checks (and 503s) while it generates
    os.execvp("gunicorn", ["gunicorn", "--bind", f":{args.port}", "--workers", "1", "--threads", "8",
                           "--timeout", "0", "welt.server:create_app()"])


if __name__ == "__main__":
    main()
