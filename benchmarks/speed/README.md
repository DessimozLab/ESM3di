# Speed benchmark

Runtime of ESM3Di, ProstT5 (Foldseek's GGUF implementation) and ColabFold
(AlphaFold2 without MSA, followed by Foldseek 3Di extraction) on reviewed viral
Swiss-Prot proteins.

| File | Purpose |
|---|---|
| [prepare_data.py](prepare_data.py) | Downloads viral Swiss-Prot and writes nested random subsets (`data/subset_<n>.fasta`, 50–1000 aa, seed 42) or, with `--create-length-bins`, 32 sequences per target length (`data/length_bins/bin_L<len>.fasta`) |
| [run_benchmark.py](run_benchmark.py) | Times each model on each FASTA and appends one row per repetition to a results CSV |
| [models/](models) | One runner per method on a common base class (`base.py`), which times the call with CUDA synchronisation and records peak VRAM |
| `*_benchmark.sh` | SLURM job scripts used for the recorded runs |
| [plot_results.ipynb](plot_results.ipynb) | Reads the CSVs and draws the figures in [figures/](figures) |

The ESM3Di runner loads the model once and times `predict_fasta` (inference
only). ProstT5 and ColabFold are timed as full CLI calls, so they include
model loading. The ESM3Di end-to-end timings in `final2_nseq_esm3di_cli.csv`
come from the runner's `_run_inference_cli` method, which calls `esm3di
predict`; `run_benchmark.py` has no option for it, so it has to be swapped in
for `_run_inference` by hand.

## Results

| CSV | Models | Input | Used by |
|---|---|---|---|
| `final2_nseq.csv` | ESM3Di (inference only), ProstT5, ColabFold | 10–10,000 sequences (ColabFold up to 300; its rows are carried over from `final_nseq.csv`, run on the earlier subsets) | `plot_results.ipynb`, Figure 1d |
| `final2_nseq_esm3di_cli.csv` | ESM3Di end to end | 10–10,000 sequences | `plot_results.ipynb`, Figure 1d |
| `final_length_bins.csv` | all three | 32 sequences per length bin | `plot_results.ipynb` |
| `final2_length_bins_plms.csv` | ESM3Di (inference only), ProstT5 | length bins upsampled to 512 sequences | `plot_results.ipynb`, Supplementary Fig. 3 |
| `final_length_bins_esm3di_cli.csv` | ESM3Di end to end | length bins upsampled to 512 sequences | `plot_results.ipynb`, Supplementary Fig. 3 |
| `final_length_bins_plms.csv` | ESM3Di (inference only), ProstT5 | length bins upsampled to 512 sequences | not plotted; earlier run, replaced by `final2_length_bins_plms.csv` |
| `final_nseq.csv`, `final_nseq_esm3di_cli.csv`, `final_nseq_plms.csv`, `new_nseq_esm3di_cli.csv`, `test_nseq_esm3di_cli.csv` | ESM3Di (and ProstT5) | 10–10,000 sequences | not plotted; earlier or repeated runs |

The plots drop the first repetition of ESM3Di and ProstT5 as warm-up, except for ESM3Di end to end against the number of sequences (`final2_nseq_esm3di_cli.csv`), which averages all 5 repetitions.

## Running

```bash
python prepare_data.py                        # nested subsets
python prepare_data.py --create-length-bins   # length bins
python run_benchmark.py --models esm3di prostt5 --repeats 6 --results-file results/<name>.csv
python run_benchmark.py --models esm3di prostt5 --length-bins --upsample-nseqs-to 512 --repeats 6 \
    --results-file results/<name>.csv
```

Run from this directory. ProstT5 weights are downloaded to `weights/` on first
use; ColabFold needs `colabfold_batch` and `foldseek` on `PATH` (the job scripts
show the setup used on the cluster). The recorded runs used one GPU on a
shared cluster node; timings depend on the hardware and on other load on the node.
