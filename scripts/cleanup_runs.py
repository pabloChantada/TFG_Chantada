"""Limpieza de carpetas de runs de RL.

Conserva todo lo que sirve para la memoria (metrics.jsonl, steps.jsonl,
metrics.png, config.json) y el último checkpoint utilizable de cada run.
Elimina los checkpoints intermedios *_ep<N>.pth que ocupan la mayor parte
del espacio, y los directorios de run que quedaron vacíos.

Uso:
    python scripts/cleanup_runs.py                # dry-run, no borra nada
    python scripts/cleanup_runs.py --apply        # ejecuta la limpieza
    python scripts/cleanup_runs.py --apply -v     # con detalle por archivo

Reglas de conservación dentro de cada run:
    KEEP siempre:  metrics.jsonl, steps.jsonl, metrics.png, config.json,
                   cualquier .json/.csv/.txt/.md
    KEEP uno:      el checkpoint *_ep<N>.pth con N más alto
    KEEP si está:  *_interrupted.pth  (modelo final si se cortó el entreno)
    DELETE:        el resto de *_ep<N>.pth intermedios

A nivel raíz de cada carpeta runs/ también se limpian los .pth sueltos
(p.ej. bc_hybrid_ep1.pth, bc_hybrid_ep2.pth -> se queda bc_hybrid.pth o el
último epN).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Carpetas con runs que se barren por defecto.
DEFAULT_RUN_ROOTS = [
    Path("src/rl/runs"),
    Path("src/rl/visual/runs"),
]

# Extensiones/archivos que NUNCA se borran dentro de un run.
KEEP_EXTENSIONS = {".jsonl", ".json", ".png", ".jpg", ".jpeg", ".csv",
                   ".txt", ".md", ".log", ".tfevents", ".yaml", ".yml"}
KEEP_FILENAMES = {"metrics.jsonl", "steps.jsonl", "metrics.png",
                  "config.json"}

EP_PATTERN = re.compile(r"^(?P<stem>.+)_ep(?P<num>\d+)\.pth$")


@dataclass
class Stats:
    files_deleted: int = 0
    files_kept: int = 0
    dirs_deleted: int = 0
    bytes_freed: int = 0
    deleted_paths: list[Path] = field(default_factory=list)
    kept_checkpoints: list[Path] = field(default_factory=list)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def is_empty_dir(d: Path) -> bool:
    try:
        return not any(d.iterdir())
    except OSError:
        return False


def classify_checkpoints(pth_files: list[Path]) -> tuple[set[Path], set[Path]]:
    """Devuelve (a_conservar, a_borrar) entre los .pth dados.

    Agrupa por stem (parte antes de _ep<N>) y deja el N más alto.
    Conserva siempre los *_interrupted.pth y los .pth sin sufijo _epN.
    """
    keep: set[Path] = set()
    drop: set[Path] = set()
    by_stem: dict[str, list[tuple[int, Path]]] = {}

    for f in pth_files:
        if f.name.endswith("_interrupted.pth"):
            keep.add(f)
            continue
        m = EP_PATTERN.match(f.name)
        if not m:
            # .pth sin patrón epN -> se conserva (suele ser el modelo final).
            keep.add(f)
            continue
        by_stem.setdefault(m.group("stem"), []).append(
            (int(m.group("num")), f))

    for stem, items in by_stem.items():
        items.sort(key=lambda t: t[0])
        # nos quedamos con el de N más alto
        keep.add(items[-1][1])
        for _, f in items[:-1]:
            drop.add(f)

    return keep, drop


def clean_run_dir(run_dir: Path, stats: Stats, apply: bool, verbose: bool) -> None:
    pth_files = [p for p in run_dir.iterdir() if p.is_file() and p.suffix == ".pth"]
    other_files = [p for p in run_dir.iterdir()
                   if p.is_file() and p.suffix != ".pth"]

    keep, drop = classify_checkpoints(pth_files)

    for f in other_files:
        if f.suffix.lower() in KEEP_EXTENSIONS or f.name in KEEP_FILENAMES:
            stats.files_kept += 1
        else:
            # archivo desconocido y no .pth -> conservador: lo dejamos.
            stats.files_kept += 1

    for f in keep:
        stats.files_kept += 1
        stats.kept_checkpoints.append(f)

    for f in drop:
        size = safe_size(f)
        stats.files_deleted += 1
        stats.bytes_freed += size
        stats.deleted_paths.append(f)
        if verbose:
            print(f"  - {f.relative_to(run_dir.parent)}  ({human(size)})")
        if apply:
            f.unlink(missing_ok=True)


def clean_root(root: Path, stats: Stats, apply: bool, verbose: bool) -> None:
    if not root.exists():
        print(f"[skip] {root} no existe")
        return
    print(f"\n== {root} ==")

    # 1. .pth sueltos a nivel raíz (p.ej. bc_hybrid_ep1.pth ...)
    loose_pth = [p for p in root.iterdir() if p.is_file() and p.suffix == ".pth"]
    if loose_pth:
        keep, drop = classify_checkpoints(loose_pth)
        for f in keep:
            stats.files_kept += 1
            stats.kept_checkpoints.append(f)
        for f in drop:
            size = safe_size(f)
            stats.files_deleted += 1
            stats.bytes_freed += size
            stats.deleted_paths.append(f)
            if verbose:
                print(f"  - {f.relative_to(root)}  ({human(size)})")
            if apply:
                f.unlink(missing_ok=True)

    # 2. cada subcarpeta = un run
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_empty_dir(d):
            stats.dirs_deleted += 1
            if verbose:
                print(f"  [rmdir vacío] {d.relative_to(root)}")
            if apply:
                try:
                    d.rmdir()
                except OSError:
                    pass
            continue
        clean_run_dir(d, stats, apply=apply, verbose=verbose)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Ejecuta los borrados. Sin esto, solo simula (dry-run).")
    ap.add_argument("--root", action="append", type=Path,
                    help="Carpeta runs/ a limpiar. Se puede repetir. "
                         "Por defecto: src/rl/runs y src/rl/visual/runs")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Imprime cada archivo afectado.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    roots = [repo_root / r for r in (args.root or DEFAULT_RUN_ROOTS)]

    mode = "APPLY" if args.apply else "DRY-RUN (no se borra nada, usa --apply)"
    print(f"Modo: {mode}")
    print(f"Repo: {repo_root}")

    stats = Stats()
    for r in roots:
        clean_root(r, stats, apply=args.apply, verbose=args.verbose)

    print("\n== Resumen ==")
    print(f"  Archivos a borrar      : {stats.files_deleted}")
    print(f"  Archivos conservados   : {stats.files_kept}")
    print(f"  Carpetas vacías        : {stats.dirs_deleted}")
    print(f"  Espacio liberado       : {human(stats.bytes_freed)}")
    print(f"  Checkpoints finales    : {len(stats.kept_checkpoints)} conservados")
    if not args.apply:
        print("\n(dry-run) ejecuta de nuevo con --apply para borrar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
