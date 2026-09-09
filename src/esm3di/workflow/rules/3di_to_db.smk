# To validate downstream tree building start from 3di generated in paper

# Rules for converting 3Di FASTA files to Foldseek database

rule convert_3di_to_db:
    """
    Converts paired AA and 3Di FASTA files into a complete
    dual-alphabet (AA + 3Di) Foldseek binary database.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        aa_fast="data/misfud_fastas/" + config['sequences'] + "_aa.fas",
        tdi_fast="data/misfud_fastas/" + config['sequences'] + "_3di.fas"
    output:
        db="results/{dataset}/db",
        db_ss="results/{dataset}/db_ss",
        db_h="results/{dataset}/db_h",
        db_idx="results/{dataset}/db.index",
        db_ss_idx="results/{dataset}/db_ss.index",
        db_h_idx="results/{dataset}/db_h.index",
        db_lookup="results/{dataset}/db.lookup"
    params:
        db_prefix="results/{dataset}/db"
    log:
        "results/{dataset}/logs/build_fs_db.log"
    script:
        "../scripts/3di2foldseekdb.py"

rule foldseek_allvall:
    """
    Performs all-vs-all dual-alphabet (AA + 3Di) structural alignment 
    using Foldseek easy-search, matching FoldTree's search behavior.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        db="results/{dataset}/db",
        db_ss="results/{dataset}/db_ss",
        db_h="results/{dataset}/db_h"
    output:
        aln="results/{dataset}/allvall_1.csv"
    params:
        db_prefix="results/{dataset}/db",
        tmp_dir="results/{dataset}/tmp",
        foldseek=config.get("foldseek_path", "foldseek")
    log:
        "results/{dataset}/logs/foldseek_allvall.log"
    shell:
        """
        mkdir -p {params.tmp_dir}
        
        {params.foldseek} easy-search \
            {params.db_prefix} \
            {params.db_prefix} \
            {output.aln} \
            {params.tmp_dir} \
            --format-output 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,lddt,lddtfull,alntmscore' \
            --exhaustive-search \
            --alignment-type 2 \
            -e inf > {log} 2>&1
            
        rm -rf {params.tmp_dir}
        """