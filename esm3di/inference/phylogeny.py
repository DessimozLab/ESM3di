"""FoldMason fast-mode alignment and FastTree NJ + NNI inference."""

import gzip
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

logger = logging.getLogger("esm3di")


def resolve_binary(explicit: str | None, variable: str, default: str) -> str:
    name = explicit or os.environ.get(variable) or default
    binary = shutil.which(name)
    if binary is None:
        raise FileNotFoundError(
            f"Cannot find {name}. Install with "
            "'conda install -c conda-forge -c bioconda foldmason fasttree', "
            f"or set {variable} / the corresponding --*-bin option."
        )
    return binary


def validate_database(database: Path) -> None:
    for component in (database, Path(f"{database}_ss"), Path(f"{database}_h")):
        for suffix in (".index", ".dbtype"):
            path = Path(f"{component}{suffix}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing database component: {path}")
        if not component.is_file() and not Path(f"{component}.0").is_file():
            raise FileNotFoundError(f"Missing database data: {component} (or {component}.0)")


def run_foldmason(database: Path, output_dir: Path, threads: int, binary: str) -> Path:
    validate_database(database)
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix = output_dir / "alignment"
    command = [
        binary, "structuremsa", str(database.resolve()), str(prefix.resolve()),
        "--threads", str(threads), "--fast", "1", "--refine-iters", "0",
    ]
    (output_dir / "run.txt").write_text(shlex.join(command) + "\n")
    log = output_dir / "foldmason.log"
    logger.info("Aligning with FoldMason; progress: %s", log)
    with log.open("x") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"FoldMason exited with status {result.returncode}; see {log}")
    alignment = output_dir / "alignment_aa.fa"
    if not alignment.is_file() or not alignment.stat().st_size:
        raise RuntimeError(f"FoldMason did not produce an amino acid alignment; see {log}")
    return alignment


def run_fasttree(alignment: Path, output_tree: Path, threads: int, binary: str) -> None:
    if not alignment.is_file() or not alignment.stat().st_size:
        raise FileNotFoundError(f"Cannot read nonempty alignment: {alignment}")
    log = Path(f"{output_tree}.log")
    for path in (output_tree, log):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}; choose a new filename")
    output_tree.parent.mkdir(parents=True, exist_ok=True)
    command = [binary, "-fastest", "-noml", "-spr", "0", "-nosupport"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    logger.info("Running FastTree NJ + NNI; progress: %s", log)
    with tempfile.NamedTemporaryFile(dir=output_tree.parent) as tree, log.open("x") as handle:
        handle.write(f"Command: {shlex.join(command)}\nOMP_NUM_THREADS={threads}\n")
        handle.flush()
        with tempfile.TemporaryFile() as decoded:
            if alignment.suffix == ".gz":
                with gzip.open(alignment, "rb") as source:
                    shutil.copyfileobj(source, decoded)
                decoded.seek(0)
                result = subprocess.run(command, stdin=decoded, stdout=tree, stderr=handle, env=env)
            else:
                with alignment.open("rb") as source:
                    result = subprocess.run(command, stdin=source, stdout=tree, stderr=handle, env=env)
        if result.returncode:
            raise RuntimeError(f"FastTree exited with status {result.returncode}; see {log}")
        tree.flush()
        if not Path(tree.name).stat().st_size:
            raise RuntimeError(f"FastTree did not produce a tree; see {log}")
        os.link(tree.name, output_tree)
    logger.info("Tree written to %s", output_tree)
