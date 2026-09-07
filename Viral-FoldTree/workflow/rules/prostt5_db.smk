# Rules for ESM3di database creation and all-vs-all Foldseek structural alignments

rule build_prostt5_foldseek_db:
    """
    Runs Foldseek native ProstT5 createdb using the pre-existing working environment.
    """
    conda:
        "../envs/prostt5_env.yaml"
    input:
        fasta=f"data/{{dataset}}/{config.get('sequences', 'sequences.fasta')}"
    output:
        db=multiext("results/{dataset}/db", "", "_h", "_ss")
    params:
        db_prefix="results/{dataset}/db",
        prostt5_weights=config.get("prostt5_weights", "data/prostT5_weights"),
        gpu=config.get("foldseek_gpu", 1)
    log:
        "results/{dataset}/logs/build_prostt5_foldseek_db.log"
    threads: config.get("prostt5_threads", 4)
    shell:
        """
        foldseek createdb {input.fasta} {params.db_prefix} \
            --prostt5-model {params.prostt5_weights} \
            --gpu {params.gpu} \
            --threads {threads} > {log} 2>&1
        """

rule foldseek_allvall:
    """
    Performs all-vs-all structural alignment using Foldseek easy-search.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        db=multiext("results/{dataset}/db", "", "_h", "_ss")
    output:
        aln="results/{dataset}/allvall_1.csv"
    params:
        db_prefix="results/{dataset}/db",
        tmp_dir="results/{dataset}/tmp",
        foldseek=config.get("foldseek_path", "foldseek")
    log:
        "results/{dataset}/logs/foldseek_allvall.log"
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