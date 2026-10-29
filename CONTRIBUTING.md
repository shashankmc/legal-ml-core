# Contributing

Thanks for helping improve legal-ml-core. Issues and pull requests are welcome
at <https://github.com/shashankmc/legal-ml-core>.

## Running the tests

Python 3.11 or newer.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # optional: avoids the CUDA build
pip install -e ".[test]"
pytest
```

The suite uses SQLite and fakeredis, so no Postgres, Redis or Keycloak is
needed. CI (`.github/workflows/ci.yml`) runs the same commands on Python 3.11
and 3.12 for every push and pull request.

If you add, remove or change an HTTP route, update `docs/openapi.yaml`;
`tests/test_openapi.py` fails when the spec and the registered routes differ.
If you change accepted upload columns or the model bundle, update
`docs/data-formats.md`.

## Cutting a release

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version
   heading with the date, following [Semantic Versioning](https://semver.org/).
2. Set the same version in `pyproject.toml`, `CITATION.cff` (`version`,
   `date-released`), `codemeta.json` (`version`, `dateModified`) and
   `.zenodo.json` (`version`).
3. Commit as `Release X.Y.Z`, tag `vX.Y.Z`, push the tag and create a GitHub
   release from it. Once the repository is connected to Zenodo, each GitHub
   release is archived and receives a DOI.

## How to cite

Cite the software using [`CITATION.cff`](CITATION.cff) (GitHub shows a
"Cite this repository" button for it). See the README for details.

## Citation metadata the owner still has to fill in

- ORCID: uncomment and fill the `orcid` line in `CITATION.cff`, and add the
  same URL as `"@id"` of the author in `codemeta.json` and as `"orcid"`
  (bare `XXXX-XXXX-XXXX-XXXX`) of the creator in `.zenodo.json`.
- DOI: after the first Zenodo release, add `doi` to `CITATION.cff`,
  `"identifier"` to `codemeta.json`, and the DOI badge to the README.
