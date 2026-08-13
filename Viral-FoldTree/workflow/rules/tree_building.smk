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
        "results/{dataset}/allvall_1.csv"
    output:
        "results/{dataset}/foldtree_fastmemat.txt"
    params:
        fmt = None,
    log:
        "results/{dataset}/logs/foldseek2distmat.log"
    script:
        "../scripts/foldseekres2distmat_simple.py"

rule quicktree:
    """
    Constructs an unrooted phylogenetic tree from the distance matrix using QuickTree.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{dataset}/{mattype}_fastmemat.txt"
    output:
        "results/{dataset}/{mattype}_struct_tree.nwk"
    log:
        "results/{dataset}/logs/{mattype}_quicktree.log"
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
        "results/{dataset}/{mattype}_struct_tree.nwk"
    output:
        "results/{dataset}/{mattype}_struct_tree.PP.nwk"
    log:
        "results/{dataset}/logs/{mattype}_struct_postprocess.log"
    script:
        "../scripts/postprocess.py"

rule mad_root_struct:
    """
    Roots the phylogenetic tree using Minimal Ancestor Deviation (MAD).
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{dataset}/{mattype}_struct_tree.PP.nwk"
    output:
        "results/{dataset}/{mattype}_struct_tree.PP.nwk.rooted"
    log:
        "results/{dataset}/logs/{mattype}_struct_madroot.log"
    params:
        mad=config.get("mad_path", "madroot/mad")
    shell:
        """
        {params.mad} {input} -n >> {log} 2>&1
        """

rule mad_root_post:
    """
    Cleans up and standardizes the final rooted tree output.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "results/{dataset}/{mattype}_struct_tree.PP.nwk.rooted"
    output:
        "results/{dataset}/{mattype}_struct_tree.PP.nwk.rooted.final"
    log:
        "results/{dataset}/logs/{mattype}_struct_madroot_post.log"
    script:
        "../scripts/process_madroot.py"


# maybe add smth like scores using ultrametricity and taxonomic congruence metrics (like in foldtree)
