# Rules for ESM3di database creation and all-vs-all Foldseek structural alignments

rule build_esm3di_foldseek_db:
    """
    Predicts 3Di sequences using ESM3di and directly compiles them 
    into a Foldseek-compatible database.
    """
    input:
        fasta=config["resolved_sequences"]
    output:
        # Foldseek DB outputs a series of database files with this prefix
        db=multiext("{out_dir}/{dataset}/db", "", "_h", "_ss")
    params:
        db_prefix="{out_dir}/{dataset}/db",
        batch_size=config.get("esm3di_batch_size", 4),
        model_ckpt=config.get("esm3di_model_ckpt", ""),
        revision=config.get("esm3di_revision", ""),
        extra_flags=lambda wildcards: " ".join(
            filter(None, [
                f"--model-ckpt {config.get('esm3di_model_ckpt')}" if config.get("esm3di_model_ckpt") else "",
                f"--revision {config.get('esm3di_revision')}" if config.get("esm3di_revision") else ""
            ])
        )
    log:
        "{out_dir}/{dataset}/logs/build_esm3di_foldseek_db.log"
    threads: 
        config.get("esm3di_threads", 4)
    resources: 
        gpu=int(config.get("esm3di_gpus") or 0)
    shell:
        """
        esm3di foldseek-db \
            --input-fasta "{input.fasta}" \
            --output-db "{params.db_prefix}" \
            --batch-size {params.batch_size} \
            --num-gpus {resources.gpu} \
            {params.extra_flags} > "{log}" 2>&1
        """


rule foldseek_allvall:
    """
    Performs all-vs-all structural alignment using Foldseek easy-search.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        db=multiext("{out_dir}/{dataset}/db", "", "_h", "_ss")
    output:
        aln="{out_dir}/{dataset}/allvall_1.csv"
    params:
        db_prefix="{out_dir}/{dataset}/db",
        tmp_dir="{out_dir}/{dataset}/tmp",
        foldseek=config.get("foldseek_path", "foldseek")
    log:
        "{out_dir}/{dataset}/logs/foldseek_allvall.log"
    threads: 
        config.get("foldseek_threads", 8)
    shell:
        """
        # Ensure cleanup of tmp directory even on unexpected failure
        trap 'rm -rf "{params.tmp_dir}"' EXIT
        
        mkdir -p "{params.tmp_dir}"
        
        "{params.foldseek}" easy-search \
            "{params.db_prefix}" \
            "{params.db_prefix}" \
            "{output.aln}" \
            "{params.tmp_dir}" \
            --threads {threads} \
            --format-output 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits' \
            --exhaustive-search \
            --alignment-type 2 \
            -e inf > "{log}" 2>&1
        """