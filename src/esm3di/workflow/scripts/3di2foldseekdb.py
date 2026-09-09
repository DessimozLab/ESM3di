import sys
import string
from typing import Union, List, Tuple

# Redirect stdout and stderr to the Snakemake log file
sys.stdout = open(snakemake.log[0], "w")
sys.stderr = sys.stdout

aa_input = snakemake.input.aa_fast
tdi_input = snakemake.input.tdi_fast
output_basename = snakemake.params.db_prefix


def _open_if_is_name(filename_or_handle, mode="r"):
    out = filename_or_handle
    input_type = "handle"
    try:
        out = open(filename_or_handle, mode)
        input_type = "name"
    except TypeError:
        pass
    return (out, input_type)


class CleanSeq:
    def __init__(self, clean=None):
        self.clean = clean
        if clean == 'delete':
            deletekeys = dict.fromkeys(string.ascii_lowercase)
            deletekeys["."] = None
            deletekeys["*"] = None
            self.remove_insertions = lambda x: x.translate(str.maketrans(deletekeys))
        elif clean == 'upper':
            deletekeys = {'*': None, ".": "-"}
            self.remove_insertions = lambda x: x.upper().translate(str.maketrans(deletekeys))
        elif clean == 'unalign':
            deletekeys = {'*': None, ".": None, "-": None}
            self.remove_insertions = lambda x: x.upper().translate(str.maketrans(deletekeys))
        else:
            self.remove_insertions = lambda x: x.upper()

    def __call__(self, seq):
        return self.remove_insertions(seq)


def iter_fasta(filename, clean=None, full_name=True):
    """Generator-based FASTA parser."""
    prev_name = None
    prev_seq = []
    input_handle, input_type = _open_if_is_name(filename)
    seq_cleaner = CleanSeq(clean)

    for line in input_handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if prev_name is not None:
                yield prev_name, seq_cleaner("".join(prev_seq))
            prev_name = line[1:] if full_name else line.split(None, 1)[0][1:]
            prev_seq = []
        else:
            prev_seq.append(line)
            
    if prev_name is not None:
        yield prev_name, seq_cleaner("".join(prev_seq))

    if input_type == "name":
        input_handle.close()


def fasta2foldseek(
    aa_input: Union[str, List[Tuple[str, str]]],
    tdi_input: Union[str, List[Tuple[str, str]]],
    output_basename: str
):
    """Compiles binary Foldseek database files from AA and 3Di inputs."""
    # Write native Foldseek dbtype headers
    with open(f"{output_basename}.dbtype", "wb") as f:
        f.write(b'\x00\x00\x00\x00')  # 0 = Amino Acid sequence type
    with open(f"{output_basename}_ss.dbtype", "wb") as f:
        f.write(b'\x0d\x00\x00\x00')  # 13 = 3Di secondary structure type
    with open(f"{output_basename}_h.dbtype", "wb") as f:
        f.write(b'\x0c\x00\x00\x00')  # 12 = Header type

    with open(f"{output_basename}", "wb") as aa_h, \
         open(f"{output_basename}_ss", "wb") as tdi_h, \
         open(f"{output_basename}_h", "wb") as header_h, \
         open(f"{output_basename}.index", "wb") as aa_index_h, \
         open(f"{output_basename}_ss.index", "wb") as tdi_index_h, \
         open(f"{output_basename}_h.index", "wb") as header_index_h, \
         open(f"{output_basename}.lookup", "wb") as lookup_h:

        pep_iterator = iter_fasta(aa_input, full_name=True) if isinstance(aa_input, str) else aa_input
        tdi_iterator = iter_fasta(tdi_input, full_name=True) if isinstance(tdi_input, str) else tdi_input

        tdi_iter = iter(tdi_iterator)
        seq_index = -1

        for pep_header, pep_seq in pep_iterator:
            try:
                tdi_header, tdi_seq = next(tdi_iter)
            except StopIteration:
                raise ValueError("3Di FASTA has fewer entries than Amino Acid FASTA.")

            pep_name = pep_header.split()[0]
            tdi_name = tdi_header.split()[0]

            assert pep_name == tdi_name, f"Header ID mismatch: '{pep_name}' vs '{tdi_name}'"
            assert len(pep_seq) == len(tdi_seq), f"Sequence length mismatch for {pep_name}: {len(pep_seq)} vs {len(tdi_seq)}"

            seq_index += 1

            # 1. Primary AA database entry
            aa_start_pos = aa_h.tell()
            aa_h.write(pep_seq.encode('utf-8') + b'\x0a\x00')
            aa_size = aa_h.tell() - aa_start_pos
            aa_index_h.write(f"{seq_index}\t{aa_start_pos}\t{aa_size}\n".encode('utf-8'))

            # 2. Secondary Structure 3Di database entry (_ss)
            tdi_start_pos = tdi_h.tell()
            tdi_h.write(tdi_seq.encode('utf-8') + b'\x0a\x00')
            tdi_size = tdi_h.tell() - tdi_start_pos
            tdi_index_h.write(f"{seq_index}\t{tdi_start_pos}\t{tdi_size}\n".encode('utf-8'))

            # 3. Header database entry (_h)
            header_start_pos = header_h.tell()
            header_h.write(pep_header.encode('utf-8') + b'\x0a\x00')
            header_size = header_h.tell() - header_start_pos
            header_index_h.write(f"{seq_index}\t{header_start_pos}\t{header_size}\n".encode('utf-8'))

            # 4. Lookup database table
            lookup_h.write(f"{seq_index}\t{pep_name}\t{seq_index}\n".encode('utf-8'))

    print(f"Successfully created Foldseek database entries for {seq_index + 1} sequences.")


# Run conversion
fasta2foldseek(aa_input, tdi_input, output_basename)