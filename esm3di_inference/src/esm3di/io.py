import string
from typing import List, Tuple, Union
import sys
from pathlib import Path


# Placeholder imports matching your existing workflow
# def fasta2foldseek(aa_input, tdi_input, output_basename): pass

def resolve_user_path(path_str: str) -> Path:
    """Resolves a user-provided file or folder path argument.

    Ensures that if the IDE runs inside 'src' or 'src/esm3di', it steps
    out to the repository workspace root automatically.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd = Path.cwd()
    if cwd.name == "src":
        cwd = cwd.parent
    elif cwd.parent.name == "src":
        cwd = cwd.parent.parent

    return (cwd / path).resolve()


def resolve_output_path(path_str: str, default_dir: str = "outputs") -> Path:
    """Resolves output target paths.

    If the user inputs just a plain filename without directories, it defaults
    it inside `outputs/`. Custom explicit paths are kept intact.
    """
    path = Path(path_str)

    # If it is a bare file without parents (e.g., "output_3di.fasta")
    if len(path.parts) == 1:
        resolved_dir = resolve_user_path(default_dir)
        return resolved_dir / path

    return resolve_user_path(path_str)


def resolve_checkpoint_path(ckpt_str: str) -> str:
    """Resolves local or remote model checkpoint locations."""
    if "/" in ckpt_str and not Path(ckpt_str).exists():
        return ckpt_str

    path = Path(ckpt_str)
    if path.is_absolute():
        return str(path)

    corrected_cwd = Path.cwd()
    if corrected_cwd.name == "src":
        corrected_cwd = corrected_cwd.parent
    elif corrected_cwd.parent.name == "src":
        corrected_cwd = corrected_cwd.parent.parent

    cwd_path = (corrected_cwd / path).resolve()
    if cwd_path.is_dir():
        return str(cwd_path)

    current_dir = Path(__file__).resolve().parent
    repo_root = None
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break

    if repo_root:
        repo_path = (repo_root / path).resolve()
        if repo_path.is_dir():
            return str(repo_path)

    return str(cwd_path)


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Simple FASTA parser reading everything into memory."""
    records = []
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip().upper())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records


def write_fasta(records: List[Tuple[str, str]], output_path: str):
    """Write a list of (header, sequence) tuples to a FASTA file."""
    with open(output_path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def _open_if_is_name(filename_or_handle, mode="r"):
    out = filename_or_handle
    input_type = "handle"
    try:
        out = open(filename_or_handle, mode)
        input_type = "name"
    except TypeError:
        pass
    return (out, input_type)


class CleanSeq():
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
            self.remove_insertions = lambda x: x

    def __call__(self, seq):
        return self.remove_insertions(seq)


def iter_fasta(filename, clean=None, full_name=False):
    """Generator-based FASTA parser for memory efficiency."""
    prev_len = 0
    prev_name = None
    prev_seq = ""
    input_handle, input_type = _open_if_is_name(filename)
    seq_cleaner = CleanSeq(clean)

    for line in input_handle:
        line = line.strip()
        if not line:
            continue
        if line[0] == ">":
            name = line[1:] if full_name else line.split(None, 1)[0][1:]
            if prev_name is not None:
                yield prev_name, seq_cleaner(prev_seq)
            prev_name = name
            prev_seq = ""
        else:
            prev_seq += line
    if prev_name is not None:
        yield prev_name, seq_cleaner(prev_seq)

    if input_type == "name":
        input_handle.close()



def fasta2foldseek(
        aa_input: Union[str, List[Tuple[str, str]]],
        tdi_input: Union[str, List[Tuple[str, str]]],
        output_basename: str
):
    """Compiles foldseek binary database files.

    Accepts paths to physical FASTA files OR in-memory lists of (header, sequence) tuples.
    """
    # Create structural type layout headers
    with open(output_basename + ".dbtype", "wb") as f:
        f.write(b'\x00\x00\x00\x00')
    with open(output_basename + "_ss.dbtype", "wb") as f:
        f.write(b'\x00\x00\x00\x00')
    with open(output_basename + "_h.dbtype", "wb") as f:
        f.write(b'\x00\x0c\x00\x00')

    with open(f"{output_basename}", "wb") as aa_h, \
            open(f"{output_basename}_ss", "wb") as tdi_h, \
            open(f"{output_basename}_h", "wb") as header_h, \
            open(f"{output_basename}.index", "wb") as aa_index_h, \
            open(f"{output_basename}_ss.index", "wb") as tdi_index_h, \
            open(f"{output_basename}_h.index", "wb") as header_index_h, \
            open(f"{output_basename}.lookup", "wb") as lookup_h:

        # Programmatic Router: Check if we are handling files or direct lists
        if isinstance(aa_input, str):
            from .io import iter_fasta
            pep_iterator = iter_fasta(aa_input, full_name=True)
        else:
            pep_iterator = aa_input

        if isinstance(tdi_input, str):
            from .io import iter_fasta
            tdi_iterator = iter_fasta(tdi_input, full_name=True)
        else:
            tdi_iterator = tdi_input

        # Convert to an explicit iterator to support calling next() safely
        tdi_iter = iter(tdi_iterator)
        seq_index = -1

        for pep_header, pep_seq in pep_iterator:
            pep_name = pep_header.split(' ')[0]
            tdi_header, tdi_seq = next(tdi_iter)
            tdi_name = tdi_header.split(' ')[0]

            assert pep_header == tdi_header, f"Headers do not match: {pep_header} vs {tdi_header}"
            assert len(pep_seq) == len(tdi_seq), "Sequence lengths do not match"

            seq_index += 1

            # Binary Stream Write operations remain exactly identical...
            aa_start_pos = aa_h.tell()
            aa_index_h.write(f"{seq_index}\t{aa_start_pos}\t".encode())
            aa_h.write(pep_seq.encode() + b'\x0a\x00')
            aa_index_h.write(f"{aa_h.tell() - aa_start_pos}\n".encode())

            tdi_start_pos = tdi_h.tell()
            tdi_index_h.write(f"{seq_index}\t{tdi_start_pos}\t".encode())
            tdi_h.write(tdi_seq.encode() + b'\x0a\x00')
            tdi_index_h.write(f"{tdi_h.tell() - tdi_start_pos}\n".encode())

            header_start_pos = header_h.tell()
            header_index_h.write(f"{seq_index}\t{header_start_pos}\t".encode())
            header_h.write(pep_header.encode() + b'\x0a\x00')
            header_index_h.write(f"{header_h.tell() - header_start_pos}\n".encode())

            lookup_h.write(f"{seq_index}\t{pep_name}\t{seq_index}\n".encode())