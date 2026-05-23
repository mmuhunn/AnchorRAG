# Artifact Schema

`artifacts/paper_2026_cikm/MANIFEST.csv` uses paths relative to the public repository root.

Columns:

- `artifact`: human-readable artifact name.
- `path`: repo-root-relative path.
- `version`: experiment or paper version.
- `used_in_paper`: `yes`, `limited`, `supporting`, or `no`.
- `section_or_table`: paper location or usage.
- `source_or_derived`: whether the file is an input/source artifact or derived output.
- `notes`: short provenance note.
- `object_type`: `file`, `directory`, or `missing`.
- `file_size_bytes`: byte size for files.
- `row_count`: CSV rows excluding header or non-empty JSONL lines when applicable.
- `sha256`: SHA-256 digest for files.
- `inventory_status`: expected to be `ok` for release artifacts.
