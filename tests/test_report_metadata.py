from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from src.live import report_metadata


def test_writer_metadata_includes_runtime_and_inputs(tmp_path: Path) -> None:
    writer = tmp_path / "writer.py"
    source = tmp_path / "source.py"
    input_report = tmp_path / "input.json"
    writer.write_text("print('writer')\n", encoding="utf-8")
    source.write_text("print('source')\n", encoding="utf-8")
    input_report.write_text("{}\n", encoding="utf-8")

    payload: dict = {"report_type": "example"}
    report_metadata.attach_writer_metadata(
        payload,
        writer_script=writer,
        report_schema_version="example.v1",
        input_reports_used=[input_report],
        source_files=[source],
        root=tmp_path,
    )

    assert payload["writer_script"] == "writer.py"
    assert payload["generated_at_utc"]
    assert payload["process_pid"] == os.getpid()
    assert payload["process_start_time_utc"]
    assert payload["report_schema_version"] == "example.v1"
    assert payload["source_file_mtime_utc"]
    assert payload["input_reports_used"] == [
        {
            "path": "input.json",
            "exists": True,
            "mtime_utc": payload["input_reports_used"][0]["mtime_utc"],
            "size_bytes": input_report.stat().st_size,
        }
    ]


def test_writer_metadata_warns_when_source_is_newer_than_process(tmp_path: Path, monkeypatch) -> None:
    writer = tmp_path / "writer.py"
    writer.write_text("print('writer')\n", encoding="utf-8")
    source_mtime = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc).timestamp()
    os.utime(writer, (source_mtime, source_mtime))
    monkeypatch.setattr(
        report_metadata,
        "_process_start_time_utc",
        lambda: datetime(2026, 5, 4, 11, 0, tzinfo=timezone.utc),
    )

    payload: dict = {}
    report_metadata.attach_writer_metadata(
        payload,
        writer_script=writer,
        report_schema_version="example.v1",
        root=tmp_path,
    )

    assert payload["stale_process_possible"] is True
    assert "STALE_PROCESS_POSSIBLE" in payload["writer_warnings"]
