rule download_prostt5_weights:
    output:
        weights_dir=directory(config.get("prostt5_model_path", "data/prostT5_weights"))
    conda:
        "../envs/prostt5_env.yaml"
    log:
        f"{config.get('output_dir')}/{config.get('dataset')}/logs/download_prostt5.log"
    shell:
        """
        foldseek databases ProstT5 {output.weights_dir} tmp > {log} 2>&1
        rm -rf tmp
        """

rule build_prostt5_foldseek_db:
    """
    Runs Foldseek native ProstT5 createdb using pre-existing or freshly downloaded weights.
    """
    conda:
        "../envs/prostt5_env.yaml"
    input:
        fasta=config["resolved_sequences"],
        prostt5_weights=config.get("prostt5_model_path", "data/prostT5_weights")
    output:
        db=multiext("{out_dir}/{dataset}/db", "", "_h", "_ss")
    params:
        db_prefix="{out_dir}/{dataset}/db",
        gpu=config.get("foldseek_gpu", 1)
    log:
        "{out_dir}/{dataset}/logs/build_prostt5_foldseek_db.log"
    threads: config.get("prostt5_threads", 4)
    shell:
        """
        foldseek createdb {input.fasta} {params.db_prefix} \
            --prostt5-model {input.prostt5_weights} \
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
        db=multiext("{out_dir}/{dataset}/db", "", "_h", "_ss")
    output:
        aln="{out_dir}/{dataset}/allvall_1.csv"
    params:
        db_prefix="{out_dir}/{dataset}/db",
        tmp_dir="{out_dir}/{dataset}/tmp",
        foldseek=config.get("foldseek_path", "foldseek")
    log:
        "{out_dir}/{dataset}/logs/foldseek_allvall.log"
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