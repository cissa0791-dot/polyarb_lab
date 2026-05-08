from __future__ import annotations

import os

from src.live.env_file import load_env_file, resolve_env_file


def test_load_env_file_sets_missing_values_without_exposing_values(tmp_path, monkeypatch) -> None:
    source = tmp_path / ".env"
    source.write_text(
        "\n".join(
            [
                "POLYMARKET_API_KEY=secret-key",
                "export POLYMARKET_CHAIN_ID=137",
                "QUOTED='quoted value'",
                "INLINE=value # comment",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("POLYMARKET_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_CHAIN_ID", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("INLINE", raising=False)

    status = load_env_file(source)

    assert status["loaded"] is True
    assert sorted(status["keys"]) == ["INLINE", "POLYMARKET_API_KEY", "POLYMARKET_CHAIN_ID", "QUOTED"]
    assert "secret-key" not in str(status)
    assert os.environ["POLYMARKET_API_KEY"] == "secret-key"
    assert os.environ["POLYMARKET_CHAIN_ID"] == "137"
    assert os.environ["QUOTED"] == "quoted value"
    assert os.environ["INLINE"] == "value"


def test_load_env_file_preserves_existing_values_by_default(tmp_path, monkeypatch) -> None:
    source = tmp_path / ".env"
    source.write_text("POLYMARKET_API_KEY=new\n", encoding="utf-8")
    monkeypatch.setenv("POLYMARKET_API_KEY", "existing")

    load_env_file(source)

    assert os.environ["POLYMARKET_API_KEY"] == "existing"


def test_resolve_env_file_prefers_explicit_path(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "explicit.env"
    explicit.write_text("POLYMARKET_CHAIN_ID=137\n", encoding="utf-8")
    root_env = tmp_path / ".env"
    root_env.write_text("POLYMARKET_CHAIN_ID=80002\n", encoding="utf-8")
    monkeypatch.delenv("POLYARB_LIVE_ENV_FILE", raising=False)

    assert resolve_env_file(explicit, root=tmp_path) == explicit


def test_resolve_env_file_uses_override_before_root_env(tmp_path, monkeypatch) -> None:
    override = tmp_path / "override.env"
    override.write_text("POLYMARKET_CHAIN_ID=137\n", encoding="utf-8")
    root_env = tmp_path / ".env"
    root_env.write_text("POLYMARKET_CHAIN_ID=80002\n", encoding="utf-8")
    monkeypatch.setenv("POLYARB_LIVE_ENV_FILE", str(override))

    assert resolve_env_file(None, root=tmp_path) == override


def test_resolve_env_file_falls_back_to_root_env(tmp_path, monkeypatch) -> None:
    root_env = tmp_path / ".env"
    root_env.write_text("POLYMARKET_CHAIN_ID=137\n", encoding="utf-8")
    monkeypatch.delenv("POLYARB_LIVE_ENV_FILE", raising=False)
    monkeypatch.setattr("src.live.env_file.DEFAULT_ENV_FILE_CANDIDATES", ())

    assert resolve_env_file(None, root=tmp_path) == root_env
