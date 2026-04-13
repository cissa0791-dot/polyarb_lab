# External Pack

This folder holds isolated external research tooling checkouts only.

## What was installed here

- `autoresearch`
  - shallow Git checkout only
  - source kept isolated under `research_tools/external_pack/autoresearch`
- `lightpanda-browser`
  - shallow Git checkout only
  - source kept isolated under `research_tools/external_pack/lightpanda-browser`
- `cli-anything`
  - shallow Git checkout only
  - source kept isolated under `research_tools/external_pack/cli-anything`

## What was intentionally not installed here

- `Cognee`
  - recorded in [`docs/external_tools.md`](D:\Issac\polyarb_lab\docs\external_tools.md)
  - intentionally not installed in phase 1

## Basic integrity/help/version checks

- checkout identity
  - `git -C D:\Issac\polyarb_lab\research_tools\external_pack\autoresearch rev-parse --short HEAD`
  - `git -C D:\Issac\polyarb_lab\research_tools\external_pack\lightpanda-browser rev-parse --short HEAD`
  - `git -C D:\Issac\polyarb_lab\research_tools\external_pack\cli-anything rev-parse --short HEAD`
- autoresearch
  - source-only for now; upstream quickstart uses `uv`
  - if you later want an isolated manual check inside that repo: `uv --version`
- Lightpanda browser
  - source-only for now; upstream README points to nightly binaries / Docker / WSL-style usage
  - if you later add an upstream binary separately: `.\lightpanda --help`
- CLI-Anything
  - source-only for now; not installed into PATH
  - if you later create an isolated venv inside that repo and install it there: `cli-anything --help`

## Intentionally not wired into mainline

- no imports into current `src/` runtime path
- no strategy behavior changes
- no background services
- no PATH/global installs
- no automatic attachment to current trading or validation scripts
