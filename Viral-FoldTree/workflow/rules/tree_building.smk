# Rules for distance matrix calculation, tree inference, and tree rooting


# TO DO: 
# - go through the scripts... exactly understand what they do 
# - add MAD


rule foldseek2distmat:
    """
    Converts Foldseek CSV alignment results into a distance matrix format for QuickTree.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{folder}/allvall_1.csv"
    output:
        "results/{folder}/foldtree_fastmemat.txt"
    params:
        fmt = None,
    log:
        "results/{folder}/logs/foldseek2distmat.log"
    script:
        "../scripts/foldseekres2distmat_simple.py"

rule quicktree:
    """
    Constructs an unrooted phylogenetic tree from the distance matrix using QuickTree.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{folder}/{mattype}_fastmemat.txt"
    output:
        "results/{folder}/{mattype}_struct_tree.nwk"
    log:
        "results/{folder}/logs/{mattype}_quicktree.log"
    shell:
        """
        quicktree -i m {input} > {output} 2> {log}
        """

rule postprocess_tree:
    """
    Post-processes the raw Newick tree topology.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{folder}/{mattype}_struct_tree.nwk"
    output:
        "results/{folder}/{mattype}_struct_tree.PP.nwk"
    log:
        "results/{folder}/logs/{mattype}_struct_postprocess.log"
    script:
        "../scripts/postprocess.py"

rule mad_root_struct:
    """
    Roots the phylogenetic tree using Minimal Ancestor Deviation (MAD).
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{folder}/{mattype}_struct_tree.PP.nwk"
    output:
        "results/{folder}/{mattype}_struct_tree.PP.nwk.rooted"
    log:
        "results/{folder}/logs/{mattype}_struct_madroot.log"
    params:
        mad=config.get("mad_path", "madroot/mad")
    shell:
        """
        {params.mad} {input} > {output} 2> {log}
        """

rule mad_root_post:
    """
    Cleans up and standardizes the final rooted tree output.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{folder}/{mattype}_struct_tree.PP.nwk.rooted"
    output:
        "results/{folder}/{mattype}_struct_tree.PP.nwk.rooted.final"
    log:
        "results/{folder}/logs/{mattype}_struct_madroot_post.log"
    script:
        "../scripts/process_madroot.py"


# maybe add smth like scores using ultrametricity and taxonomic congruence metrics (like in foldtree)