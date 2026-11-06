import sys

import foldseek2tree
import tipnames
import toytree

# Restore the original tip names that were replaced by placeholders for QuickTree
with open(snakemake.input.tree) as treein:
    newick = ''.join(line.strip() for line in treein)
newick = tipnames.restore_names(newick, tipnames.read_name_map(snakemake.input.names))
with open(snakemake.output[0], 'w') as treeout:
    treeout.write(newick + '\n')

# toytree recurses once per tree level, and trees of many near-identical
# sequences can be as deep as they have tips.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * newick.count(',') + 1000))

treefile = foldseek2tree.postprocess(snakemake.output[0] , snakemake.output[0] )
print(treefile , 'done')
