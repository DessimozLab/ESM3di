import sys

import toytree

# toytree recurses once per tree level, and trees of many near-identical
# sequences can be as deep as they have tips.
with open(snakemake.input[0]) as treein:
    ntips = treein.read().count(',') + 1
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * ntips + 1000))

#load the tree
tre = toytree.tree(snakemake.input[0])
#output the 1st tree

tre.write(snakemake.output[0], tree_format=0)

print( 'postprocessing MAD-root done')