# GithubDownloader

Downloads markdown from GitHub repo URLs into a target directory (default `source_docs`). Used to fetch docs from GitHub for the RAG app. The list of repos to download is read from a JSON file.

Note:

Unlike `git clone`, it downloads only matching files (e.g. `.md` only, excluding `README.md`) — no full repo or history.
With Git you can do something similar using sparse checkout, but that is path-based, not filter-by-extension or exclude-by-filename like this tool.

Git alternative (path-based):
```sh
git clone --filter=blob:none --sparse <repo_url>
cd <repo>
git sparse-checkout set <path_in_repo>
```

## Requirements

- Python 3.10+
- Dependencies: managed by [uv](https://github.com/astral-sh/uv). From repo root run `uv sync --extra dev`, then activate the venv (`source .venv/bin/activate`) so you can use `python` without `uv run`. See the main [README](../README.md#installation).

## Input: JSON file

The downloader reads repo URLs from **`urls.json`** in this package directory (`github_downloader/urls.json`). The repository includes an **example** [urls.json](./urls.json) that you can edit for your needs, or replace with your own list.

**Format:** A JSON array of objects with `name` and `url`:

```json
[
  {
    "name": "Top10",
  "url": "https://github.com/OWASP/Top10/blob/master/2025/docs/en/"
  },
  {
    "name": "Proactive_Controls",
    "url": "https://github.com/OWASP/www-project-proactive-controls/tree/master/docs/the-top-10"}
]
```

- `name`: Used as the subdirectory under `COLLECTION_PATH` (e.g. `source_docs/Top10`).
- `url`: GitHub tree or blob URL (e.g. `.../tree/master/...` or `.../blob/master/...`).

If the file is missing or invalid, the program logs an error and exits.

## Run

From repo root with the venv activated:

```sh
python -m github_downloader
```

If the venv is not activated, use `uv run python -m github_downloader`.

## Configuration

- **Env:** `GITHUB_TOKEN` (recommended for rate limits), `COLLECTION_PATH` (default `source_docs`).
- **Constructor:** `config` (see class docstring).
- **Input:** [urls.json](./urls.json) — list of repos (example included; edit or replace for your use). See [Input: JSON file](#input-json-file) above.

---

## Issues and Learnings

- URLs for listing directory contents (API) and for file contents (raw) are different.
- GitHub API is rate limited; use a `GITHUB_TOKEN` for higher limits.

**Config:** Extensions (e.g. `.md`) and exclude patterns (e.g. `README.md`) are already constructor args; could be driven by env later if you want different defaults.

---
