# Rules for ESM3di database creation and all-vs-all Foldseek structural alignments

rule build_esm3di_foldseek_db:
    """
    Predicts 3Di sequences using ESM3di and directly compiles them 
    into a Foldseek-compatible database.
    """
    conda:
        "../envs/esm3di.yaml"
    input:
        fasta="results/{folder}/sequences_20.fasta"
    output:
        # Foldseek DB outputs a series of database files with this prefix
        db=multiext("results/{folder}/db", "", "_h", "_ss")
    params:
        db_prefix="results/{folder}/db",
        batch_size=config.get("esm3di_batch_size", 4),
        model_ckpt=config.get("esm3di_model_ckpt", ""),
        extra_flags=(
            f"--model-ckpt {config.get('esm3di_model_ckpt')}" 
            if config.get("esm3di_model_ckpt", "") else ""
        )
    log:
        "results/{folder}/logs/build_esm3di_foldseek_db.log"

    # threads: config.get("esm3di_threads", 4)
    # resources: gpu=config.get("esm3di_gpus", 1)

    shell:
        """
        esm3di foldseek-db \
            --input-fasta {input.fasta} \
            --output-db {params.db_prefix} \
            --batch-size {params.batch_size} \
            {params.extra_flags} > {log} 2>&1
        """

rule foldseek_allvall:
    """
    Performs all-vs-all structural alignment using Foldseek easy-search.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        db=multiext("results/{folder}/db", "", "_h", "_ss")
    output:
        aln="results/{folder}/allvall_1.csv"
    params:
        db_prefix="results/{folder}/db",
        tmp_dir="results/{folder}/tmp",
        foldseek=config.get("foldseek_path", "foldseek")
    log:
        "results/{folder}/logs/foldseek_allvall.log"
    #threads: config.get("foldseek_threads", 8)
    shell:
        """
        mkdir -p {params.tmp_dir}
        
        {params.foldseek} easy-search \
            {params.db_prefix} \
            {params.db_prefix} \
            {output.aln} \
            {params.tmp_dir} \
            --format-output 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits' \
            --exhaustive-search \
            --alignment-type 2 \
            -e inf > {log} 2>&1
            
        rm -rf {params.tmp_dir}
        """