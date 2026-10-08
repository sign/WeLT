import threading

import pytest

pytest.importorskip("flask")

from werkzeug.serving import make_server  # noqa: E402

from welt.server import MODEL_TAG_HEADER, create_app, generate  # noqa: E402


class FakeGenerator:
    def __init__(self):
        self.release = threading.Event()
        self.release.set()
        self.processor = type("Processor", (), {"pretokenize": staticmethod(lambda text: ["\x02", *text.split()])})

    def generate(self, texts, max_generated_words, temperature, seed):
        if max_generated_words > 100:
            raise ValueError("Prompt words + max_generated_words exceed the latent's context")
        self.release.wait()
        return [f"{text}!" * max_generated_words for text in texts]


@pytest.fixture
def generator():
    return FakeGenerator()


@pytest.fixture
def client(generator):
    return create_app(generator, model_tag="v1").test_client()


def test_health(client):
    response = client.get("/health")
    assert response.json == {"status": "ok", "version": "v1"}
    assert response.headers[MODEL_TAG_HEADER] == "v1"


def test_generate(client):
    response = client.post("/generate", json={"texts": ["a b", "c"], "max_generated_words": 2})
    assert response.json == {"outputs": ["a b!a b!", "c!c!"], "generated_words": 4}


@pytest.mark.parametrize("body", [None, [], {"texts": "a"}, {"texts": [1]}, {"texts": ["a"], "seed": "x"},
                                  {"texts": ["a"], "max_generated_words": "x"},
                                  {"texts": ["a"], "max_generated_words": 1000}])
def test_generate_rejects_invalid_requests(client, body):
    assert client.post("/generate", json=body).status_code == 400


def test_busy_server_answers_503(client, generator):
    generator.release.clear()
    first = threading.Thread(target=client.post, args=("/generate",), kwargs={"json": {"texts": ["a"]}})
    first.start()
    try:
        for _ in range(100):  # Until the first request holds the generator
            response = client.post("/generate", json={"texts": ["b"]})
            if response.status_code == 503:
                break
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "1"
    finally:
        generator.release.set()
        first.join()


def test_client(generator):
    server = make_server("localhost", 0, create_app(generator), threaded=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        result = generate(f"http://localhost:{server.port}", ["hi"], max_generated_words=1)
        assert result == {"outputs": ["hi!"], "generated_words": 1}
    finally:
        server.shutdown()
        thread.join()
