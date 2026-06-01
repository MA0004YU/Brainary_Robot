"""Bootstrap sys.path for standalone cerebellum scripts inside ntu_jinao_repo."""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap_isaaclab_paths(script_file: str | Path) -> None:
    """Add cerebellum module and Isaac Lab source packages to ``sys.path``.

    Walks upward from ``script_file`` until it finds an Isaac Lab root
    (directory containing ``isaaclab.sh`` and ``source/isaaclab``).
    Supports both layouts:

    - ``IsaacLab/source/standalone/franka_state_machine_cerebellum/``
    - ``IsaacLab/ntu_jinao_repo/source/standalone/franka_state_machine_cerebellum/``
    """

    script_dir = Path(script_file).resolve().parent
    script_dir_str = script_dir.as_posix()
    if script_dir_str not in sys.path:
        sys.path.insert(0, script_dir_str)

    isaaclab_root: Path | None = None
    for candidate in script_dir.parents:
        if (candidate / "isaaclab.sh").exists() and (candidate / "source" / "isaaclab").exists():
            isaaclab_root = candidate
            break

    if isaaclab_root is None:
        return

    for package_dir in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
        source_path = isaaclab_root / "source" / package_dir
        source_path_str = source_path.as_posix()
        if source_path.exists() and source_path_str not in sys.path:
            sys.path.insert(0, source_path_str)
