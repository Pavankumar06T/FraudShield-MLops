"""DVC tracking for data that does not live in the repository.

The IEEE-CIS raw files and the split parquet sit on Google Drive at
``MyDrive/fraudshield/data``; the repo finds them through
``FRAUDSHIELD_DATA``. DVC, however, only tracks paths inside the workspace
-- ``dvc add`` on an outside path is an error, and the ``--external`` escape
hatch was removed in DVC 3.

So the bridge is a link: ``<repo>/data`` points at whatever
``FRAUDSHIELD_DATA`` resolves to, and DVC tracks ``data/raw`` and
``data/splits`` as ordinary workspace paths. On Colab that is a symlink to
the mounted Drive; on Windows a directory junction, which needs no
administrator rights where a symlink would.

What this buys, and what it does not. The ``.dvc`` files committed to git
carry an md5 of each tracked directory, so "which bytes produced this
model" has an answer that survives the data being moved, re-split, or
overwritten -- and a changed hash is visible in a diff. It does NOT by
itself copy the data anywhere; ``dvc push`` to a remote is optional and
doubles the storage, which is a real cost against ~700 MB on Drive. The
hashes are the point.

    python -m src.data.version --link          # set up <repo>/data
    python -m src.data.version                 # link, then dvc add
    python -m src.data.version --status        # what DVC thinks changed
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.common.config import DATA_ROOT, REPO_ROOT

#: Paths DVC tracks, relative to the repo root.
TRACKED: tuple[str, ...] = ("data/raw", "data/splits")

#: Where <repo>/data must point.
LINK_PATH: Path = REPO_ROOT / "data"

#: Default remote name and its location relative to the data root. Kept
#: beside the data on Drive rather than inside it, so `dvc push` never
#: recurses into its own cache.
REMOTE_NAME: str = "drive"


def dvc(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a dvc subcommand from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "dvc", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def link_status() -> str:
    """Describe what ``<repo>/data`` currently is."""
    if not LINK_PATH.exists() and not LINK_PATH.is_symlink():
        return "missing"
    if LINK_PATH.is_symlink() or _is_junction(LINK_PATH):
        return "link"
    return "directory"


def _is_junction(path: Path) -> bool:
    """Windows directory junctions are not symlinks to ``is_symlink``."""
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except (OSError, ValueError):
        return False


def _resolved_target() -> Path | None:
    """Where the link points, or None if it is not a link."""
    try:
        return Path(os.readlink(LINK_PATH)).resolve()
    except (OSError, ValueError):
        return None


def ensure_link(force: bool = False) -> str:
    """Make ``<repo>/data`` resolve to DATA_ROOT. Returns what happened.

    Refuses to replace a real directory holding files unless ``force``.
    Deleting someone's data to make a tool happy is not a trade this should
    make on its own -- and if DATA_ROOT already *is* ``<repo>/data``, there
    is nothing to do at all.
    """
    if DATA_ROOT == LINK_PATH:
        return "data root is already <repo>/data; nothing to link"

    status = link_status()
    if status == "link":
        if _resolved_target() == DATA_ROOT:
            return f"link already points at {DATA_ROOT}"
        LINK_PATH.unlink()
    elif status == "directory":
        contents = [p for p in LINK_PATH.rglob("*") if p.is_file()]
        if contents and not force:
            raise RuntimeError(
                f"{LINK_PATH} is a real directory holding {len(contents)} file(s), "
                f"and FRAUDSHIELD_DATA points elsewhere ({DATA_ROOT}).\n"
                "Refusing to replace it. Move or delete it yourself, or pass "
                "--force if you are certain it is disposable."
            )
        # Empty directory (the scaffold's data/raw, data/splits) is safe to
        # remove: nothing is lost and the link is what the tooling needs.
        for directory in sorted(
            (p for p in LINK_PATH.rglob("*") if p.is_dir()), reverse=True
        ):
            directory.rmdir()
        LINK_PATH.rmdir()

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"FRAUDSHIELD_DATA resolves to {DATA_ROOT}, which does not exist. "
            "Mount Drive first, or point it somewhere real."
        )

    if os.name == "nt":
        # Junction rather than symlink: symlinks need either Developer Mode
        # or elevation on Windows, junctions need neither.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(LINK_PATH), str(DATA_ROOT)],
            check=True,
            capture_output=True,
            text=True,
        )
        return f"junction {LINK_PATH} -> {DATA_ROOT}"

    LINK_PATH.symlink_to(DATA_ROOT, target_is_directory=True)
    return f"symlink {LINK_PATH} -> {DATA_ROOT}"


def configure_remote(location: Path | None = None) -> str:
    """Point a DVC remote at a directory beside the data on Drive.

    Optional. Without it the hashes still work; with it ``dvc push`` can
    make the cache survive a Colab runtime being recycled.
    """
    target = location or (DATA_ROOT.parent / "dvcstore")
    dvc("remote", "add", "--default", "--force", REMOTE_NAME, str(target))
    return f"remote {REMOTE_NAME!r} -> {target}"


def add_tracked() -> list[str]:
    """``dvc add`` each tracked path, returning the lines DVC reported."""
    missing = [p for p in TRACKED if not (REPO_ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Not present under the data root: {', '.join(missing)}. "
            f"DATA_ROOT is {DATA_ROOT}."
        )
    result = dvc("add", *TRACKED)
    return [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track the data splits with DVC.")
    parser.add_argument("--link", action="store_true", help="set up <repo>/data and stop")
    parser.add_argument("--status", action="store_true", help="show dvc status and stop")
    parser.add_argument("--remote", action="store_true", help="configure the Drive remote")
    parser.add_argument(
        "--force", action="store_true", help="replace a non-empty <repo>/data"
    )
    args = parser.parse_args(argv)

    print(f"  FRAUDSHIELD_DATA -> {DATA_ROOT}")
    print(f"  repo data path   -> {LINK_PATH} ({link_status()})")

    if args.status:
        result = dvc("status", check=False)
        print((result.stdout + result.stderr).strip() or "  (nothing tracked yet)")
        return 0

    print(f"  {ensure_link(force=args.force)}")
    if args.link:
        return 0

    if args.remote:
        print(f"  {configure_remote()}")

    print(f"\n  dvc add {' '.join(TRACKED)}  (hashing may take a while on ~700 MB)")
    for line in add_tracked():
        print(f"    {line}")

    print(
        "\n  Commit the pointers, never the data:\n"
        f"    git add {' '.join(f'{p}.dvc' for p in TRACKED)} data/.gitignore\n"
        "    git commit -m 'Track data splits with DVC'\n"
        "\n  Optional, doubles storage on Drive:\n"
        "    python -m src.data.version --remote\n"
        "    dvc push"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
