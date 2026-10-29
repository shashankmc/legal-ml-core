"""docs/openapi.yaml must describe exactly the routes init_ml_feature registers.

Fails when a blueprint gains (or loses) a route or method without the spec
being updated, so the published API description cannot drift from the code.
"""

import re
from pathlib import Path

import yaml

SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "openapi.yaml"
HTTP_METHODS = {"get", "put", "post", "delete", "patch"}


def _load_spec():
    with SPEC_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _registered_operations(app):
    ops = set()
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        # Flask `<converter:name>` / `<name>` -> OpenAPI `{name}`.
        path = re.sub(r"<(?:[^:<>]+:)?([^<>]+)>", r"{\1}", rule.rule)
        for method in rule.methods - {"HEAD", "OPTIONS"}:
            ops.add((path, method.lower()))
    return ops


def _spec_operations(spec):
    return {
        (path, method)
        for path, item in spec["paths"].items()
        for method in item
        if method in HTTP_METHODS
    }


def test_spec_is_openapi_3_1():
    spec = _load_spec()
    assert spec["openapi"].startswith("3.1")
    assert "bearerAuth" in spec["components"]["securitySchemes"]


def test_every_registered_route_is_in_the_spec(app):
    missing = _registered_operations(app) - _spec_operations(_load_spec())
    assert not missing, f"routes missing from docs/openapi.yaml: {sorted(missing)}"


def test_spec_describes_no_unregistered_routes(app):
    extra = _spec_operations(_load_spec()) - _registered_operations(app)
    assert not extra, f"docs/openapi.yaml describes unknown routes: {sorted(extra)}"


def test_spec_path_params_are_declared():
    spec = _load_spec()
    for path, item in spec["paths"].items():
        names = set(re.findall(r"{([^}]+)}", path))
        declared = set()
        for p in item.get("parameters", []):
            if "$ref" in p:
                p = spec["components"]["parameters"][p["$ref"].rsplit("/", 1)[1]]
            if p["in"] == "path":
                declared.add(p["name"])
        assert names == declared, f"{path}: path params {names} vs declared {declared}"


def test_spec_refs_resolve():
    spec = _load_spec()
    refs = set(re.findall(r"\$ref: \"#/([^\"]+)\"", SPEC_PATH.read_text(encoding="utf-8")))
    for ref in refs:
        node = spec
        for part in ref.split("/"):
            assert part in node, f"unresolved $ref #/{ref}"
            node = node[part]
