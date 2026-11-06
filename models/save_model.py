from __future__ import annotations

import json
import os
import platform
import re
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import joblib

DEFAULT_SAVE_DIR = Path(__file__).resolve().parents[1] / "saved_models"
_FORMAT_VERSION = 1
TRACKS = ("off_market", "list_aware")
_ALLOWED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _ALLOWED_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "Model name must contain only letters, numbers, underscores, "
            "hyphens or periods; it cannot include a path."
        )
    return name


def _check_track(track: str | None) -> str | None:
    if track is not None and track not in TRACKS:
        raise ValueError(
            f"track must be one of {TRACKS}; omitted track selects legacy flat storage"
        )
    return track


def _directory(save_dir: str | Path | None, track: str | None = None) -> Path:
    root = Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR
    track = _check_track(track)
    return root / track if track is not None else root


def _describe(obj: Any, *, depth: int = 0) -> dict[str, Any]:
    """Record the shape/types of a model collection, without serializing it to JSON."""
    if isinstance(obj, (list, tuple)):
        result: dict[str, Any] = {"collection": type(obj).__name__, "length": len(obj)}
        if depth < 3:
            result["items"] = [_describe(item, depth=depth + 1) for item in obj]
        return result
    cls = type(obj)
    return {"type": f"{cls.__module__}.{cls.__qualname__}"}


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("joblib", "catboost", "scikit-learn", "pandas", "numpy", "statsmodels"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            pass
    return versions


def save_model(
    model: Any,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
    compress: int = 3,
    save_dir: str | Path | None = None,
    track: str | None = None,
) -> Path:
    """Persist one fitted model, a list, or a nested list of fitted models.

    Parameters
    ----------
    model: A fitted model or arbitrary (nested) collection of fitted models.
    name:  Unique experiment name (no path separators).
    metadata: Optional JSON-compatible run details (parameters, tracks, cutoffs).
    overwrite: Replace a previously saved run with the same name if True.
    compress: joblib compression level, 0 (fast/larger) through 9 (slow/smaller).
    save_dir: Base saved_models directory (defaults beside this module).
    track: 'off_market' or 'list_aware' for new saves. Omitting track
           preserves the legacy flat layout; never put a mixed-track model
           collection inside a track-specific folder.

    Returns the folder containing the model and its metadata.
    """
    name = _check_name(name)
    if not isinstance(compress, int) or not 0 <= compress <= 9:
        raise ValueError("compress must be an integer from 0 to 9")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    track = _check_track(track)
    metadata = dict(metadata or {})
    if track is not None:
        if metadata.get("track", track) != track:
            raise ValueError("metadata['track'] conflicts with storage track")
        if "tracks" in metadata and list(metadata["tracks"]) != [track]:
            raise ValueError("Track-specific snapshots must contain only their own track")
        metadata["track"] = track

    root = _directory(save_dir, track)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists() and not overwrite:
        raise FileExistsError(f"Already saved: {target} (choose a new name or overwrite=True)")

    info = {
        "format_version": _FORMAT_VERSION,
        "name": name,
        "storage_track": track if track is not None else "legacy",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "model_structure": _describe(model),
        "metadata": metadata,
    }
    # Validate metadata *before* beginning the potentially expensive model dump.
    json_text = json.dumps(info, indent=2, default=str)

    # Stage the complete run in a temporary folder. An unsuccessful dump does
    # not leave a partially saved model at its final location.
    with tempfile.TemporaryDirectory(prefix=".saving_", dir=root) as staging:
        staged = Path(staging)
        joblib.dump(model, staged / "model.joblib", compress=compress)
        (staged / "metadata.json").write_text(json_text + "\n", encoding="utf-8")

        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"A non-directory entry already exists: {target}")
            # Only remove an existing run when the user explicitly opts in.
            shutil.rmtree(target)
        os.replace(staged, target)

    return target


def load_model(name: str, *, save_dir: str | Path | None = None,
               track: str | None = None) -> Any:
    """Load one track-scoped or legacy model, without cross-track fallback."""
    folder = _directory(save_dir, track) / _check_name(name)
    path = folder / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"No saved model at {path}")
    # SECURITY: joblib.load can execute code. Never load untrusted files.
    return joblib.load(path)


def model_info(name: str, *, save_dir: str | Path | None = None,
               track: str | None = None) -> dict[str, Any]:
    """Read metadata from the selected track, without loading the model."""
    path = _directory(save_dir, track) / _check_name(name) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"No model metadata at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_saved_models(*, save_dir: str | Path | None = None,
                      track: str | None = None) -> list[str]:
    """List models from one namespace; omitted track means legacy only."""
    root = _directory(save_dir, track)
    if not root.is_dir():
        return []
    return sorted(
        folder.name
        for folder in root.iterdir()
        if folder.is_dir() and (folder / "model.joblib").is_file()
    )


def list_model_catalog(*, save_dir: str | Path | None = None) -> list[dict[str, str]]:
    """List both track folders and legacy snapshots without loading binaries."""
    rows = []
    for track in (None, *TRACKS):
        namespace = "legacy" if track is None else track
        for name in list_saved_models(save_dir=save_dir, track=track):
            rows.append({
                "track": namespace, "name": name,
                "relative_path": f"{name}/" if track is None else f"{track}/{name}/",
            })
    return rows


def save_validation_tracks(
    monthly_models: Any,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
    compress: int = 3,
    save_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Split [month][off_market, list_aware] validation fits into track folders.

    Models keep their chronological order. Does not remove or modify a legacy
    snapshot. Caller must ensure [off_market, list_aware] ordering.
    """
    _check_name(name)
    if not isinstance(monthly_models, (list, tuple)) or not monthly_models:
        raise ValueError("Expected nonempty monthly [off_market, list_aware] model pairs")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2
           for pair in monthly_models):
        raise ValueError("Each validation month must contain exactly two ordered models")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    base = dict(metadata or {})
    if "tracks" in base and list(base["tracks"]) != list(TRACKS):
        raise ValueError("Validation track order must be off_market, list_aware")
    if "validation_months" in base and len(base["validation_months"]) != len(monthly_models):
        raise ValueError("validation_months count does not match monthly models")
    root = _directory(save_dir)
    if not overwrite:
        existing = [
            str(root / track / name) for track in TRACKS
            if (root / track / name).exists()
        ]
        if existing:
            raise FileExistsError("Already saved: " + ", ".join(existing))
    paths = {}
    for idx, track in enumerate(TRACKS):
        track_metadata = {
            **base, "track": track, "tracks": [track],
            "split_from_multi_track": True,
            "validation_model_order": "chronological",
        }
        paths[track] = save_model(
            [pair[idx] for pair in monthly_models], name,
            metadata=track_metadata, overwrite=overwrite,
            compress=compress, save_dir=save_dir, track=track,
        )
    return paths


# Optional plural spellings; both functions also support single models.
save_models = save_model
load_models = load_model
