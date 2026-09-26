import xml.etree.ElementTree as ET
import pytest
import requests
import socket
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


def test_server_multiple_addresses(monkeypatch):
    # The CLI value replaces the environment value, including an unavailable address.
    monkeypatch.setenv("LLAMA_ARG_HOST", "192.0.2.1")
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")  # ty: ignore[too-many-positional-arguments]

    server.server_host = "127.0.0.1,::1"
    server.api_key = "test-multiple-addresses"
    server.start()

    def check_address(host):
        res = server.make_request("GET", "/health", host=host)
        assert res.status_code == 200
        res = server.make_request("POST", "/v1/completions", data={}, host=host)
        assert res.status_code == 401
        events = list(server.make_stream_request("POST", "/v1/completions", data={
            "prompt": "Once upon a time",
            "max_tokens": 8,
            "stream": True,
        }, headers={"Authorization": f"Bearer {server.api_key}"}, host=host))
        assert len(events) > 1
        return True

    # parallel_function_calls swallows exceptions, a failed check leaves None in the results
    results = parallel_function_calls([(check_address, (host,)) for host in ["127.0.0.1", "[::1]"]])
    assert all(results)


def test_server_props():
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == server.n_slots
    default_val = res.body["default_generation_settings"]
    assert server.n_ctx is not None and server.n_slots is not None
    assert default_val["n_ctx"] == server.n_ctx / server.n_slots
    assert default_val["params"]["seed"] == server.seed


def test_server_models():
    global server
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias


def test_server_slots():
    global server

    # without slots endpoint enabled, this should return error
    server.server_slots = False
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 501 # ERROR_TYPE_NOT_SUPPORTED
    assert "error" in res.body
    server.stop()

    # with slots endpoint enabled, this should return slots info
    server.server_slots = True
    server.n_slots = 2
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert len(res.body) == server.n_slots
    assert server.n_ctx is not None and server.n_slots is not None
    assert res.body[0]["n_ctx"] == server.n_ctx / server.n_slots
    assert "params" not in res.body[0]


def test_load_split_model():
    global server
    server.offline = False
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf"
    server.model_alias = "tinyllama-split"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl)+", res.body["content"])


def test_no_ui():
    global server
    # default: UI enabled
    server.start()
    url = f"http://{server.server_host}:{server.server_port}"
    res = requests.get(url)
    assert res.status_code == 200
    assert "<!doctype html>" in res.text
    server.stop()

    # with --no-ui, the UI should be disabled
    server.no_ui = True
    server.start()
    res = requests.get(url)
    assert res.status_code == 404
    assert server.make_request("GET", "/opensearch.xml").status_code == 404


@pytest.mark.parametrize("api_prefix", ["", "/llama", "/llama&chat"])
def test_opensearch(api_prefix: str, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_API_PREFIX", api_prefix)
    make_url = server.make_url
    monkeypatch.setattr(server, "make_url", lambda path, host=None: make_url(api_prefix + path, host))
    server.api_key = "test-key"
    server.start()
    namespace = {"o": "http://a9.com/-/spec/opensearch/1.1/"}

    for host, proto in [("chat.example:8080", ""), ("chat.example:8443", "https"), ("[::1]:8080", "http")]:
        res = server.make_request("GET", "/opensearch.xml", headers={"Host": host, "X-Forwarded-Proto": proto})
        assert res.status_code == 200
        assert res.headers["Content-Type"] == "application/opensearchdescription+xml"
        assert res.headers["Cache-Control"] == "no-store"
        document = ET.fromstring(res.body)
        base = f"{proto or 'http'}://{host}{api_prefix}"
        assert document.find("o:ShortName", namespace).text == "llama.cpp"
        assert document.find("o:Url", namespace).attrib == {"type": "text/html", "method": "GET", "template": base + "/?q={searchTerms}"}
        assert document.find("o:Image", namespace).text == base + "/favicon.ico"

    for headers in [{"Host": ""}, {"Host": "good.example@evil.example"}, {"Host": 'bad"host'}, {"X-Forwarded-Proto": "https, http"}, {"X-Forwarded-Proto": "invalid"}]:
        assert server.make_request("GET", "/opensearch.xml", headers=headers).status_code == 400

    assert server.make_request("GET", "/props").status_code == 401
    assert server.make_request("GET", "/opensearch.xml/extra").status_code == 401
    assert server.make_request("GET", "/opensearchXxml", headers={"Authorization": "Bearer test-key"}).status_code == 404


def test_server_model_aliases_and_tags():
    global server
    server.model_alias = "tinyllama-2,fim,code"
    server.model_tags = "chat,fim,small"
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    model = res.body["data"][0]
    # aliases field must contain all aliases
    assert set(model["aliases"]) == {"tinyllama-2", "fim", "code"}
    # tags field must contain all tags
    assert set(model["tags"]) == {"chat", "fim", "small"}
    # id is derived from first alias (alphabetical order from std::set)
    assert model["id"] == "code"
