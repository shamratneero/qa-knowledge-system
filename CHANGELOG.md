# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Minimal browser UI at `/ui` connected to the `/ask` API.
- Expanded benchmark dataset in `data/eval_set.json`.
- Benchmark metrics export to `artifacts/benchmark_results.json`.
- Deployment manifests: `render.yaml` and `railway.toml`.
- Release workflow on version tags (`.github/workflows/release.yml`).
- Release checklist in README.

### Changed
- Enhanced API and search logging with request IDs and elapsed timing.
- Improved API schemas and endpoint documentation.
- Strengthened tests with route/contract/performance smoke coverage.

## [1.0.0] - 2026-07-12

### Added
- FastAPI backend with `/`, `/health`, `/ask`, and `/docs`.
- Hybrid retrieval engine (keyword, fuzzy, semantic).
- Confidence scores and ranked result payloads.
- Docker and Docker Compose support.
- CI workflow for automated test execution.
- Professional README with architecture and deployment guidance.
