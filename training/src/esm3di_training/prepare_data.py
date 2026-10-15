"""Turn a set of protein structures into 3Di training FASTAs.

Input: directories and/or files with PDB / mmCIF structures (optionally
``.gz``), e.g. AlphaFold/ColabFold models. Output (in ``--out-dir``)::

    train_aa.fasta  train_3di.fasta  train_plddt_bins.fasta
    val_aa.fasta    val_3di.fasta    val_plddt_bins.fasta
    test_aa.fasta   test_3di.fasta   test_plddt_bins.fasta     (if --test-fraction > 0)
    all_aa.fasta    all_3di.fasta    all_plddt_bins.fasta
    manifest.json

* AA and 3Di sequences come from ``foldseek createdb`` (so they are exactly
  what FoldSeek would index).
* pLDDT is read from the B-factor column of each structure's CA atoms and
  stored as one digit per residue (``min(int(pLDDT // 10), 9)``). Values in
  [0, 1] (e.g. ESMFold) are rescaled to [0, 100]. Use ``--no-plddt`` for
  experimental structures, where B-factors are not confidences.
* Splits are random by default; ``--cluster-split`` keeps FoldSeek clusters
  together so validation/test structures are not near-duplicates of training.

Example::

    python -m esm3di.prepare_data --structures /data/afdb_models/ \
        --out-dir data/my_3di --val-fraction 0.02 --test-fraction 0.05 \
        --foldseek-bin foldseek
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

STRUCTURE_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".ent.gz", ".cif.gz", ".mmcif.gz")
THREEDI_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------

def _is_structure(name: str) -> bool:
    return name.lower().endswith(STRUCTURE_SUFFIXES)


def find_structures(inputs: Sequence[str]) -> List[str]:
    """Expand directories (recursively) and list files (one path per line in ``.txt``)."""
    found: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                found.extend(os.path.join(root, f) for f in files if _is_structure(f))
        elif item.endswith(".txt") and not _is_structure(item):
            with open(item) as fh:
                found.extend(line.strip() for line in fh if line.strip())
        elif os.path.isfile(item):
            found.append(item)
        else:
            raise FileNotFoundError(f"Structure input not found: {item}")
    found = sorted(set(os.path.abspath(p) for p in found))
    if not found:
        raise SystemExit("No structure files found (expected .pdb/.cif[.gz]).")
    return found


def structure_stem(path: str) -> str:
    """FoldSeek's entry name for a file: basename without structure suffixes."""
    name = os.path.basename(path)
    if name.lower().endswith(".gz"):
        name = name[:-3]
    for suffix in (".pdb", ".ent", ".cif", ".mmcif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


# ---------------------------------------------------------------------------
# pLDDT extraction
# ---------------------------------------------------------------------------

def read_chain_plddts(path: str) -> Dict[str, List[float]]:
    """Return ``{chain_id: [CA B-factor per residue]}`` for the first model."""
    from Bio.PDB import MMCIFParser, PDBParser

    is_cif = ".cif" in os.path.basename(path).lower()
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        structure = parser.get_structure("s", fh)
    model = next(iter(structure))
    chains: Dict[str, List[float]] = {}
    for chain in model:
        values = []
        for residue in chain:
            hetflag = residue.id[0]
            if hetflag == "W" or "CA" not in residue:
                continue
            ca = residue["CA"]
            if hetflag.startswith("H_") and ca.element != "C":
                continue  # calcium ions etc.
            values.append(float(ca.get_bfactor()))
        if values:
            chains[chain.id] = values
    return chains


def _plddt_worker(path: str) -> Tuple[str, Dict[str, List[float]], Optional[str]]:
    try:
        return path, read_chain_plddts(path), None
    except Exception as exc:  # noqa: BLE001 - report and continue
        return path, {}, f"{type(exc).__name__}: {exc}"


def plddt_to_bins(values: Sequence[float]) -> str:
    scale = 100.0 if values and max(values) <= 1.0 else 1.0
    return "".join(str(max(0, min(9, int(v * scale // 10)))) for v in values)


def collect_plddt_bins(paths: Sequence[str], threads: int) -> Tuple[Dict[str, str], List[str]]:
    """Map FoldSeek-style entry names to pLDDT bin strings."""
    bins: Dict[str, str] = {}
    errors: List[str] = []
    with mp.Pool(max(1, threads)) as pool:
        for i, (path, chains, err) in enumerate(pool.imap_unordered(_plddt_worker, paths, chunksize=64), 1):
            if err:
                errors.append(f"{path}: {err}")
                continue
            stem = structure_stem(path)
            if len(chains) == 1:
                bins[stem] = plddt_to_bins(next(iter(chains.values())))
            for chain_id, values in chains.items():
                bins[f"{stem}_{chain_id}"] = plddt_to_bins(values)
            if i % 5000 == 0:
                print(f"  parsed pLDDT for {i}/{len(paths)} structures", flush=True)
    return bins, errors


# ---------------------------------------------------------------------------
# FoldSeek
# ---------------------------------------------------------------------------

def find_foldseek(foldseek_bin: Optional[str]) -> str:
    candidate = foldseek_bin or os.environ.get("FOLDSEEK_BIN") or "foldseek"
    resolved = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
    if not resolved:
        raise SystemExit(
            f"FoldSeek binary not found ({candidate!r}). Install it (e.g. `conda install -c bioconda foldseek`) "
            "or pass --foldseek-bin /path/to/foldseek."
        )
    return resolved


def _run(cmd: List[str]) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr[-4000:]}")


def _read_db_sequences(db: str) -> List[str]:
    """Read sequences from an MMseqs-style DB in index order."""
    entries = []
    with open(db + ".index") as fh:
        for line in fh:
            key, offset, length = line.split("\t")
            entries.append((int(key), int(offset), int(length)))
    entries.sort()
    out = []
    with open(db, "rb") as fh:
        for _, offset, length in entries:
            fh.seek(offset)
            out.append(fh.read(length).rstrip(b"\x00").rstrip(b"\n").decode())
    return out


def _read_lookup(db: str) -> List[str]:
    rows = []
    with open(db + ".lookup") as fh:
        for line in fh:
            key, name, _ = line.rstrip("\n").split("\t")
            rows.append((int(key), name))
    rows.sort()
    return [name for _, name in rows]


def foldseek_extract(paths: Sequence[str], workdir: str, foldseek: str, threads: int) -> List[Tuple[str, str, str]]:
    """Run ``foldseek createdb``; return ``[(name, aa, 3di)]``."""
    list_file = os.path.join(workdir, "structures.tsv")  # foldseek detects lists by .tsv suffix
    with open(list_file, "w") as fh:
        fh.write("\n".join(paths) + "\n")
    db = os.path.join(workdir, "db")
    _run([foldseek, "createdb", list_file, db, "--threads", str(threads), "-v", "1"])
    names = _read_lookup(db)
    aa = _read_db_sequences(db)
    ss = _read_db_sequences(db + "_ss")
    if not (len(names) == len(aa) == len(ss)):
        raise RuntimeError(f"Inconsistent FoldSeek DB: {len(names)} names, {len(aa)} AA, {len(ss)} 3Di")
    return [(n, a.upper(), s.upper()) for n, a, s in zip(names, aa, ss)]


def foldseek_clusters(workdir: str, foldseek: str, threads: int, min_seq_id: float, coverage: float) -> Dict[str, str]:
    """Cluster the createdb output; return ``{member_name: representative_name}``."""
    db = os.path.join(workdir, "db")
    clu = os.path.join(workdir, "clu")
    _run([
        foldseek, "cluster", db, clu, os.path.join(workdir, "clu_tmp"), "--threads", str(threads),
        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "-v", "1",
    ])
    tsv = os.path.join(workdir, "clu.tsv")
    _run([foldseek, "createtsv", db, db, clu, tsv, "--threads", str(threads), "-v", "1"])
    mapping = {}
    with open(tsv) as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")[:2]
            mapping[member.split()[0]] = rep.split()[0]
    return mapping


# ---------------------------------------------------------------------------
# Splitting and writing
# ---------------------------------------------------------------------------

def split_names(
    names: Sequence[str], val_fraction: float, test_fraction: float, seed: int, groups: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    """Random split by entry or, with ``groups``, by cluster representative."""
    keys = sorted({(groups or {}).get(n, n) for n in names})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test = int(round(len(keys) * test_fraction))
    n_val = int(round(len(keys) * val_fraction))
    if val_fraction > 0 and n_val == 0 and len(keys) > 1:
        n_val = 1
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test : n_test + n_val])
    out = {"train": [], "val": [], "test": []}
    for n in names:
        k = (groups or {}).get(n, n)
        out["test" if k in test_keys else "val" if k in val_keys else "train"].append(n)
    return out


def _write(path: str, records: Iterable[Tuple[str, str]]) -> int:
    n = 0
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")
            n += 1
    return n


def prepare(
    structures: Sequence[str],
    out_dir: str,
    foldseek_bin: Optional[str] = None,
    use_plddt: bool = True,
    val_fraction: float = 0.02,
    test_fraction: float = 0.0,
    seed: int = 42,
    min_length: int = 20,
    max_length: Optional[int] = None,
    cluster_split: bool = False,
    cluster_min_seq_id: float = 0.3,
    cluster_coverage: float = 0.8,
    threads: Optional[int] = None,
    keep_workdir: bool = False,
) -> dict:
    """Run the whole structures → FASTA pipeline. Returns the manifest dict."""
    threads = threads or max(1, (os.cpu_count() or 2) - 1)
    foldseek = find_foldseek(foldseek_bin)
    paths = find_structures(structures)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Found {len(paths)} structure files")

    workdir = tempfile.mkdtemp(prefix="foldseek_", dir=out_dir)
    try:
        print("Extracting AA + 3Di with foldseek createdb ...")
        entries = foldseek_extract(paths, workdir, foldseek, threads)
        print(f"  {len(entries)} chains")

        bins: Dict[str, str] = {}
        if use_plddt:
            print("Reading pLDDT from B-factors ...")
            bins, errors = collect_plddt_bins(paths, threads)
            for err in errors[:10]:
                print(f"  warning: could not parse {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more parse errors")

        kept, dropped = [], {"length": 0, "alphabet": 0, "plddt": 0}
        for name, aa, ss in entries:
            if len(aa) != len(ss) or len(aa) < min_length or (max_length and len(aa) > max_length):
                dropped["length"] += 1
                continue
            if set(ss) - THREEDI_ALPHABET:
                dropped["alphabet"] += 1
                continue
            pl = None
            if use_plddt:
                pl = bins.get(name)
                if pl is None or len(pl) != len(aa):
                    dropped["plddt"] += 1
                    continue
            kept.append((name, aa, ss, pl))
        print(f"Kept {len(kept)} chains; dropped {dropped}")
        if not kept:
            raise SystemExit("No usable chains left after filtering.")

        groups = None
        if cluster_split:
            print("Clustering with foldseek for a leakage-aware split ...")
            groups = foldseek_clusters(workdir, foldseek, threads, cluster_min_seq_id, cluster_coverage)
        split = split_names([k[0] for k in kept], val_fraction, test_fraction, seed, groups)
        by_name = {k[0]: k for k in kept}

        files: Dict[str, Dict[str, str]] = {}
        counts: Dict[str, int] = {}
        for split_name, names in [("all", [k[0] for k in kept])] + list(split.items()):
            if not names:
                continue
            prefix = os.path.join(out_dir, split_name)
            files[split_name] = {
                "aa": prefix + "_aa.fasta",
                "3di": prefix + "_3di.fasta",
            }
            counts[split_name] = _write(files[split_name]["aa"], ((n, by_name[n][1]) for n in names))
            _write(files[split_name]["3di"], ((n, by_name[n][2]) for n in names))
            if use_plddt:
                files[split_name]["plddt_bins"] = prefix + "_plddt_bins.fasta"
                _write(files[split_name]["plddt_bins"], ((n, by_name[n][3]) for n in names))

        manifest = {
            "format": "esm3di-data-v1",
            "structures": list(structures),
            "num_structure_files": len(paths),
            "counts": counts,
            "dropped": dropped,
            # stored relative to the data dir so the directory can be moved
            "files": {sp: {k: os.path.basename(v) for k, v in kinds.items()} for sp, kinds in files.items()},
            "use_plddt": use_plddt,
            "split": {
                "val_fraction": val_fraction,
                "test_fraction": test_fraction,
                "seed": seed,
                "cluster_split": cluster_split,
                "cluster_min_seq_id": cluster_min_seq_id if cluster_split else None,
                "cluster_coverage": cluster_coverage if cluster_split else None,
            },
            "min_length": min_length,
            "max_length": max_length,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Wrote {out_dir}/manifest.json: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return manifest
    finally:
        if keep_workdir:
            print(f"FoldSeek working directory kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _resolve_in_dir(data_dir: str, path: str) -> str:
    """Resolve a manifest entry: relative to ``data_dir``; fall back to the bare file name."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(data_dir, path)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(data_dir, os.path.basename(path))


def data_files_from_dir(data_dir: str) -> Dict[str, Optional[str]]:
    """Return train/val FASTA paths (train_3di config keys) from a prepared data directory."""
    manifest_path = os.path.join(data_dir, "manifest.json")
    files: Dict[str, Dict[str, str]] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            files = json.load(fh).get("files", {})
        files = {sp: {k: _resolve_in_dir(data_dir, v) for k, v in kinds.items()} for sp, kinds in files.items()}
    else:
        for split_name in ("train", "val", "test", "all"):
            entry = {}
            for kind in ("aa", "3di", "plddt_bins"):
                p = os.path.join(data_dir, f"{split_name}_{kind}.fasta")
                if os.path.exists(p):
                    entry[kind] = p
            if entry:
                files[split_name] = entry
    train = files.get("train") or files.get("all")
    if not train or "aa" not in train or "3di" not in train:
        raise FileNotFoundError(f"{data_dir} does not contain train_aa.fasta/train_3di.fasta (or a manifest.json)")
    val = files.get("val", {})
    return {
        "aa_fasta": train["aa"],
        "three_di_fasta": train["3di"],
        "plddt_bins_fasta": train.get("plddt_bins"),
        "val_aa_fasta": val.get("aa"),
        "val_three_di_fasta": val.get("3di"),
        "val_plddt_bins_fasta": val.get("plddt_bins") if train.get("plddt_bins") else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_prepare_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--structures", "-s", nargs="+", required=True,
                   help="Structure directories/files (.pdb/.cif[.gz]) or .txt lists of paths.")
    p.add_argument("--out-dir", "-o", required=True)
    p.add_argument("--foldseek-bin", help="FoldSeek binary (default: $FOLDSEEK_BIN or `foldseek` on PATH).")
    p.add_argument("--no-plddt", action="store_true", help="Do not read pLDDT from B-factors (experimental structures).")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-length", type=int, default=20)
    p.add_argument("--max-length", type=int)
    p.add_argument("--cluster-split", action="store_true", help="Split by FoldSeek clusters instead of per chain.")
    p.add_argument("--cluster-min-seq-id", type=float, default=0.3)
    p.add_argument("--cluster-coverage", type=float, default=0.8)
    p.add_argument("--threads", type=int)
    p.add_argument("--keep-workdir", action="store_true", help="Keep the intermediate FoldSeek DB.")
    return p


def run_prepare(ns: argparse.Namespace) -> dict:
    return prepare(
        ns.structures, ns.out_dir, foldseek_bin=ns.foldseek_bin, use_plddt=not ns.no_plddt,
        val_fraction=ns.val_fraction, test_fraction=ns.test_fraction, seed=ns.seed,
        min_length=ns.min_length, max_length=ns.max_length, cluster_split=ns.cluster_split,
        cluster_min_seq_id=ns.cluster_min_seq_id, cluster_coverage=ns.cluster_coverage,
        threads=ns.threads, keep_workdir=ns.keep_workdir,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_prepare_args(parser)
    run_prepare(parser.parse_args(argv))


if __name__ == "__main__":
    main()
