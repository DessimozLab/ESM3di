"""Write a FoldSeek database directly from amino-acid + 3Di sequences.

FoldSeek databases are MMseqs2-style flat files. This writer produces the same
layout as ``foldseek createdb`` for the parts needed to search with predicted
3Di sequences (``DB``, ``DB_ss``, ``DB_h`` with their ``.index``/``.dbtype``
files and ``DB.lookup``), so no FoldSeek binary is required.

Search it like any FoldSeek DB::

    foldseek easy-search query.pdb my_db result.m8 tmp/   # or
    foldseek search queryDB my_db alnDB tmp/
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence, Tuple

Record = Tuple[str, str]


def write_foldseek_db(aa_records: Sequence[Record], three_di_records: Sequence[Record], output_db: str) -> int:
    """Write ``output_db`` (+ ``_ss``/``_h`` siblings). Returns the number of entries.

    ``aa_records`` and ``three_di_records`` must be in the same order with
    identical headers and equal sequence lengths.
    """
    if len(aa_records) != len(three_di_records):
        raise ValueError(f"{len(aa_records)} AA records but {len(three_di_records)} 3Di records")
    out_dir = os.path.dirname(os.path.abspath(output_db))
    os.makedirs(out_dir, exist_ok=True)

    with open(output_db + ".dbtype", "wb") as fh:
        fh.write(b"\x00\x00\x00\x00")  # amino acids
    with open(output_db + "_ss.dbtype", "wb") as fh:
        fh.write(b"\x00\x00\x00\x00")  # 3Di uses the AA alphabet type
    with open(output_db + "_h.dbtype", "wb") as fh:
        fh.write(b"\x00\x0c\x00\x00")  # generic headers

    with open(output_db, "wb") as aa_h, open(output_db + "_ss", "wb") as ss_h, open(output_db + "_h", "wb") as hd_h, \
         open(output_db + ".index", "wb") as aa_i, open(output_db + "_ss.index", "wb") as ss_i, \
         open(output_db + "_h.index", "wb") as hd_i, open(output_db + ".lookup", "wb") as lookup:
        for n, ((aa_header, aa_seq), (tdi_header, tdi_seq)) in enumerate(zip(aa_records, three_di_records)):
            if aa_header.split()[0] != tdi_header.split()[0]:
                raise ValueError(f"Entry {n}: AA header {aa_header!r} != 3Di header {tdi_header!r}")
            if len(aa_seq) != len(tdi_seq):
                raise ValueError(f"Entry {n} ({aa_header.split()[0]}): AA length {len(aa_seq)} != 3Di length {len(tdi_seq)}")
            for data_h, index_h, payload in (
                (aa_h, aa_i, aa_seq.upper()),
                (ss_h, ss_i, tdi_seq.upper()),
                (hd_h, hd_i, aa_header),
            ):
                start = data_h.tell()
                data_h.write(payload.encode() + b"\n\x00")
                index_h.write(f"{n}\t{start}\t{data_h.tell() - start}\n".encode())
            lookup.write(f"{n}\t{aa_header.split()[0]}\t{n}\n".encode())
    return len(aa_records)


def fasta2foldseek(aa_fasta: str, three_di_fasta: str, output_db: str) -> int:
    """File-based wrapper around :func:`write_foldseek_db`."""
    from .predict import read_fasta_records

    count = write_foldseek_db(read_fasta_records(aa_fasta), read_fasta_records(three_di_fasta), output_db)
    print(f"Wrote FoldSeek database {output_db} with {count} entries")
    return count
