# Rules for distance matrix calculation, tree inference, and tree rooting

rule foldseek2distmat:
    """
    Converts Foldseek CSV alignment results into a distance matrix format for QuickTree.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "{out_dir}/{dataset}/allvall_1.csv"
    output:
        "{out_dir}/{dataset}/{model}_fastmemat.txt"
    params:
        fmt=None
    log:
        "{out_dir}/{dataset}/logs/{model}_foldseek2distmat.log"
    script:
        "../scripts/foldseekres2distmat_simple.py"

rule quicktree:
    """
    Constructs an unrooted phylogenetic tree from the distance matrix using QuickTree.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "{out_dir}/{dataset}/{model}_fastmemat.txt"
    output:
        "{out_dir}/{dataset}/{model}_struct_tree.nwk"
    log:
        "{out_dir}/{dataset}/logs/{model}_quicktree.log"
    shell:
        """
        quicktree -i m "{input}" > "{output}" 2> "{log}"
        """


rule postprocess_tree:
    """
    Post-processes the raw Newick tree topology.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "{out_dir}/{dataset}/{model}_struct_tree.nwk"
    output:
        "{out_dir}/{dataset}/{model}_struct_tree.PP.nwk"
    log:
        "{out_dir}/{dataset}/logs/{model}_posprocess_tree.log"
    script:
        "../scripts/postprocess.py"


rule mad_root_struct:
    """
    Roots the phylogenetic tree using Minimal Ancestor Deviation (MAD).
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "{out_dir}/{dataset}/{model}_struct_tree.PP.nwk"
    output:
        "{out_dir}/{dataset}/{model}_struct_tree.PP.nwk.rooted"
    log:
        "{out_dir}/{dataset}/logs/{model}_mad_root_struct.log"
    params:
        mad=config.get("mad_path", "madroot/mad")
    shell:
        """
        "{params.mad}" "{input}" -n >> "{log}" 2>&1
        """


rule mad_root_post:
    """
    Cleans up and standardizes the final rooted tree output.
    """
    conda:
        "../envs/foldtree.yaml"
    input:
        "{out_dir}/{dataset}/{model}_struct_tree.PP.nwk.rooted"
    output:
        "{out_dir}/{dataset}/{model}_struct_tree.PP.nwk.rooted.final"
    log:
        "{out_dir}/{dataset}/logs/{model}_mad_root_post.log"
    script:
        "../scripts/process_madroot.py"
