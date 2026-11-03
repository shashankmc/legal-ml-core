"""The API process must not load the worker's training stack.

Hosts install `legal-ml-core` for the API and `legal-ml-core[compute]` for the
worker, so the API image can leave out torch, transformers, shap, lime and
nltk. Importing the package or mounting it on an app must not pull them in.
Each check runs in a fresh interpreter, since this test session has already
imported them for other tests.
"""
import subprocess
import sys
import textwrap

COMPUTE_MODULES = ("torch", "transformers", "accelerate", "shap", "lime", "nltk", "spacy")


def _loaded_compute_modules(code):
    probe = textwrap.dedent(code) + textwrap.dedent(
        f"""
        import sys
        print(",".join(m for m in {COMPUTE_MODULES!r} if m in sys.modules))
        """
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    return [m for m in result.stdout.strip().split(",") if m]


def test_importing_the_package_loads_no_training_stack():
    assert _loaded_compute_modules("import legal_ml_core") == []


def test_mounting_the_api_loads_no_training_stack():
    code = """
        import fakeredis
        from flask import Flask
        from legal_ml_core import db, init_ml_feature

        app = Flask(__name__)
        app.config.update(
            SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
            AUTH_ISSUER="https://keycloak.test/realms/biss",
            AUTH_AUDIENCE="baigle",
        )
        db.init_app(app)
        init_ml_feature(app, fakeredis.FakeStrictRedis(decode_responses=True), "/tmp", db=db)
    """
    assert _loaded_compute_modules(code) == []


def test_the_worker_entrypoint_is_still_exported():
    from legal_ml_core import run_ml_worker
    from legal_ml_core.worker import run_ml_worker as from_module

    assert run_ml_worker is from_module
