"""Radial tree panels for the Filoviridae and Flaviviridae figures.

Sizes are physical (millimetres, in the fonts of ``benchmarks/pub_style.py``), so panels can be placed
in a layout program at 100%.

Tips are named ``Genus-species_accession`` (label_mode "scientific") or by a six-letter code such as
``E1HPHV_name`` (label_mode "six_letter"). A branch has the colour of its tips if they all share one
species or code, the colour of their genus if they share only that, and is gray otherwise.
"""

from collections import Counter
import colorsys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
import sys

from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import toytree

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
from pub_style import MM, MUTED_TEXT, use_pub_style  # noqa: E402

# One protein: each group code gets its own hue (Tableau 10 without gray, in this order)
GROUP_COLORS = ("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F")
UNCLASSIFIED_COLOR = "#79706E"
MIXED_COLOR = "#A0A0A0"
TEXT_COLOR = "#111111"

CLADE_SIZE_POWER = 0.2   # angular weight of a subclade of n tips: n**power (1: equal per tip)
EDGE_WIDTH = 0.4         # points
TIP_DIAMETER = 1.8       # points
LEGEND_FONT = 5          # points; the smallest size journals accept
LEGEND_MARKER = 2.4      # points
LEGEND_ROW_MM = 2.0
LEGEND_TEXT_MM = 1.8     # from the left of a legend entry to its text
BLOCK_GAP = 0.4          # legend rows between genera
TITLE_MM = 3.2           # above each tree
TITLE_BASELINE_MM = 2.5
ROW_GAP_MM = 1.5         # below each tree
SPECIES_FONT = 5         # points, species names of species_tree_grid
SUPPORT_FONT = 4.5       # points, node support of species_tree_grid
SUPPORT_OFFSET_MM = (0.4, 0.6)  # from the node: to the left, and up, clear of its branches
SUBTITLE_MM = 2.5        # for a line below the title of a tree in species_tree_grid
CORNER_FONT = 5.5        # points, scores at the top right of a tree: smaller than its title lines
CORNER_ROW_MM = 2.1      # between the score lines
COLUMN_HEADER_FONT = 6   # points, the line naming a column of trees
COLUMN_HEADER_BASELINE_MM = 2.0   # above the first title, which a column header pushes down by SUBTITLE_MM
COLUMN_RULE_MM = 0.6     # from the baseline of a column header to the rule below it
COLUMN_RULE_ABOVE_MM = 1.8   # to the rule above it, for a column header at the bottom of a figure
COLUMN_FOOTER_MM = 3.2   # band at the bottom of a figure that holds the column headers
SCORE_TABLE_ROW_MM = 1.85
SCORE_TABLE_GAP_MM = 1.5 # between the columns of a score table
CORNER_TOP_MM = TITLE_BASELINE_MM   # first score baseline, on the line of the title
INK_GRID_MM = 0.25       # resolution of the search for white space
INK_MARGIN_MM = 0.5      # distance a legend keeps from the tree
FIGURE_FORMATS = ("svg", "pdf", "png")
FIGURE2_PANEL_HEIGHT_MM = 116   # panels a and b have the same height
FIGURE2_SCORE_TABLE_TOP_MM = 0.8   # their score tables line up, above the trees
HEADING_MM = 5           # above a figure with a heading


# --- Labels and colours --------------------------------------------------------------------------

def parse_tip(name, mode):
    """(genus, species) or (protein, four-letter group split in two) of a tip name."""
    if mode == "scientific":
        genus, _, species = name.split("_", 1)[0].partition("-")
        if genus and species:
            return genus, species
        raise ValueError(f"Expected Genus-species_accession; got {name!r}.")
    match = re.match(r"^([A-Za-z0-9]{6})(?:_|$)", name)
    if not match:
        raise ValueError(f"Expected a six-letter prefix such as E1HPHV_; got {name!r}.")
    code = match.group(1)
    return code[:2], code[2:4], code[4:6]


def tip_keys(filenames, mode):
    keys = set()
    for filename in filenames:
        keys.update(parse_tip(name, mode) for name in toytree.tree(str(filename)).get_tip_labels())
    return keys


def species_colors(keys, genus_colors):
    """One hue per genus; its species are shades of it, closer together for fewer species."""
    colors = {}
    for genus in sorted({key[0] for key in keys}):
        hue, light, saturation = colorsys.rgb_to_hls(*to_rgb(genus_colors[genus]))
        members = sorted(key for key in keys if key[0] == genus)
        spread = min(.15, .05 * (len(members) - 1))
        center = min(max(light, .30 + spread), .68 - spread)   # the lightest shade stays legible as small text
        levels = [light] if len(members) == 1 else np.linspace(center - spread, center + spread, len(members))
        for key, level in zip(members, levels):
            colors[key] = colorsys.hls_to_rgb(hue, float(level), saturation)
    return colors


def group_colors(keys):
    """Per protein, one hue per group code in code order; unclassified codes (..UN) are dark gray."""
    colors = {}
    for protein in {key[0] for key in keys}:
        named = sorted(key for key in keys if key[0] == protein and key[2] != "UN")
        if len(named) > len(GROUP_COLORS):
            raise ValueError(f"{protein}: {len(named)} group codes; at most {len(GROUP_COLORS)} have distinct colours.")
        colors.update(zip(named, GROUP_COLORS))
    colors.update({key: UNCLASSIFIED_COLOR for key in keys if key[2] == "UN"})
    return colors


def species_legend(colors, keys=None):
    """Legend blocks (heading, entries, italic): each genus above its abbreviated species."""
    keys = sorted(colors if keys is None else keys)
    return [(genus, [(f"{genus[0]}. {key[1]}", colors[key]) for key in keys if key[0] == genus], True)
            for genus in sorted({key[0] for key in keys})]


def group_legend(colors, keys=None):
    """One block per group code, so the codes can spread over columns; unclassified codes share one."""
    keys = sorted(colors if keys is None else keys)
    blocks = [(None, [("".join(key[1:]), colors[key])], False) for key in keys if key[2] != "UN"]
    if any(key[2] == "UN" for key in keys):
        blocks.append((None, [("Unclassified", UNCLASSIFIED_COLOR)], False))
    return blocks


# --- Layout --------------------------------------------------------------------------------------

@dataclass
class TreeLayout:
    names: list          # tip names, in the order of the first rows of xy
    xy: np.ndarray       # node coordinates, in branch-length units
    edges: np.ndarray    # (parent, child) node indices
    edge_colors: list
    tip_colors: list


def balanced_coordinates(tree, keys, power=CLADE_SIZE_POWER):
    """Node coordinates with angular sectors by subclade size: a subclade of n tips of one key gets
    weight n**power, so repeated taxa do not use up the angle. Branch lengths are kept exactly."""
    totals = Counter(keys)
    weights = np.zeros(tree.nnodes)
    weights[:tree.ntips] = [totals[key] ** (power - 1) for key in keys]
    for node in tree:
        if not node.is_leaf():
            weights[node.idx] = sum(weights[child.idx] for child in node.children)
    coordinates, sectors = np.zeros((tree.nnodes, 2)), np.zeros((tree.nnodes, 2))
    sectors[tree.treenode.idx] = 0, 2 * np.pi
    for parent in tree.traverse("preorder"):
        start, end = sectors[parent.idx]
        cursor = start
        for child in parent.children:
            width = (end - start) * weights[child.idx] / weights[parent.idx]
            sectors[child.idx] = cursor, cursor + width
            midpoint = cursor + width / 2
            coordinates[child.idx] = coordinates[parent.idx] + child.dist * np.array([np.sin(midpoint), np.cos(midpoint)])
            cursor += width
    return coordinates


def fill_rotation(xy, aspect):
    """The rotation of the coordinates that can be drawn largest in a panel of the given aspect."""
    def rotated(angle):
        return xy @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    def scale(angle):
        extent = np.ptp(rotated(angle), axis=0)
        return max(extent[0] / aspect, extent[1])

    return rotated(min(np.linspace(0, np.pi, 180, endpoint=False), key=scale))


def rotate_together(layouts, size_mm):
    """Rotate trees by one angle, the one that lets them be drawn largest at one scale in panels of the given size. Unlike
    fill_rotation for each tree, this keeps them aligned with each other."""
    def rotation(angle):
        return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    def scale(angle):
        return max(np.max(np.ptp(layout.xy @ rotation(angle), axis=0) / (np.asarray(size_mm) - 1)) for layout in layouts)

    best = rotation(min(np.linspace(0, np.pi, 180, endpoint=False), key=scale))
    for layout in layouts:
        layout.xy = layout.xy @ best


def tree_layout(filename, mode, colors, genus_colors=None, aspect=1.0):
    """Coordinates and colours of a tree, rotated to fill a panel of the given aspect (width/height)."""
    tree = toytree.tree(str(filename))
    species = [None] * tree.nnodes     # key shared by all tips below a node, if any
    genera = [None] * tree.nnodes      # the same for the first part of the key (genus)
    for node in tree:                  # toytree iterates from the tips to the root
        i = node.idx
        if node.is_leaf():
            species[i] = parse_tip(node.name, mode)
            genera[i] = species[i][0]
        else:
            for shared in (species, genera):
                values = {shared[child.idx] for child in node.children}
                shared[i] = values.pop() if len(values) == 1 and None not in values else None
    keys = species[:tree.ntips]
    xy = fill_rotation(balanced_coordinates(tree, keys), aspect)
    edges, edge_colors = [], []
    for node in tree:
        if node.is_root():
            continue
        edges.append((node.up.idx, node.idx))
        if species[node.idx] is not None:
            edge_colors.append(colors[species[node.idx]])
        elif genus_colors and genera[node.idx] is not None:
            edge_colors.append(genus_colors[genera[node.idx]])
        else:
            edge_colors.append(MIXED_COLOR)
    return TreeLayout(list(tree.get_tip_labels()), xy, np.array(edges), edge_colors, [colors[key] for key in keys])


def align_layout(layout, reference):
    """Rotate (or mirror) ``layout`` onto ``reference``, using their shared tips, to ease comparison."""
    position = {name: i for i, name in enumerate(reference.names)}
    shared = [(i, position[name]) for i, name in enumerate(layout.names) if name in position]
    if len(shared) < 3:   # another protein: nothing to align on
        return layout
    moving, fixed = layout.xy[[i for i, _ in shared]], reference.xy[[j for _, j in shared]]
    u, _, vt = np.linalg.svd((moving - moving.mean(axis=0)).T @ (fixed - fixed.mean(axis=0)))
    layout.xy = (layout.xy - moving.mean(axis=0)) @ (u @ vt)
    return layout


def centered_extent(layout):
    low, high = layout.xy.min(axis=0), layout.xy.max(axis=0)
    return (low + high) / 2, high - low


# --- Drawing -------------------------------------------------------------------------------------

def draw_tree(ax, layout, units_per_mm, size_mm, edge_width=EDGE_WIDTH, tip_diameter=TIP_DIAMETER):
    """Draw a tree centred in ``ax`` at a fixed scale."""
    center, _ = centered_extent(layout)
    half = np.array(size_mm) * units_per_mm / 2
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_axis_off()
    segments = layout.xy[layout.edges]
    order = sorted(range(len(segments)), key=lambda i: layout.edge_colors[i] != MIXED_COLOR)   # mixed branches below
    ax.add_collection(LineCollection(segments[order], colors=[layout.edge_colors[i] for i in order],
                                     linewidths=edge_width, capstyle="round", zorder=1, gid="branches"))
    tips = ax.scatter(*layout.xy[:len(layout.names)].T, s=tip_diameter ** 2, c=layout.tip_colors, linewidths=0,
                      zorder=2, gid="tips")
    tips.set_clip_on(False)


def text_width_mm(text, fontsize=LEGEND_FONT, **style):
    fig = plt.figure()
    width = fig.text(0, 0, text, fontsize=fontsize, **style).get_window_extent(fig.canvas.get_renderer()).width
    width = width / fig.dpi / MM
    plt.close(fig)
    return width


def block_rows(block):
    return len(block[1]) + (block[0] is not None) * (1 + BLOCK_GAP)


def legend_columns(blocks, columns):
    """Split the blocks, in order, into columns that are about equally long."""
    columns = min(columns, len(blocks))
    best = None
    for breaks in combinations(range(1, len(blocks)), columns - 1):
        parts = [blocks[i:j] for i, j in zip((0, *breaks), (*breaks, len(blocks)))]
        longest = max(sum(block_rows(block) for block in part) for part in parts)
        if best is None or longest < best[0]:
            best = longest, parts
    return best[1]


def legend_geometry(blocks, columns=1, title=(), row_mm=LEGEND_ROW_MM):
    """The columns of a legend with their widths, and its total width and height (mm)."""
    parts = legend_columns(blocks, columns)
    widths = []
    for part in parts:
        widest = 0.0
        for name, entries, italic in part:
            if name is not None:
                widest = max(widest, text_width_mm(name, fontweight="bold", fontstyle="italic"))
            for label, _ in entries:
                widest = max(widest, LEGEND_TEXT_MM + text_width_mm(label, fontstyle="italic" if italic else "normal"))
        widths.append(widest + 0.8)
    width = max([sum(widths)] + [text_width_mm(line, fontweight="bold") + 0.5 for line in title])
    rows = max(sum(block_rows(block) for block in part) for part in parts)
    height = (rows + len(title) + 0.3 * bool(title)) * row_mm
    return parts, widths, width, height


def draw_legend(ax, blocks, x, y, columns=1, title=(), row_mm=LEGEND_ROW_MM):
    """A legend with its top left corner at (x, y), in an axes whose data units are mm from the top left."""
    parts, widths, _, _ = legend_geometry(blocks, columns, title, row_mm)
    for line in title:
        ax.text(x, y + row_mm / 2, line, fontsize=LEGEND_FONT, fontweight="bold", color=TEXT_COLOR, va="center")
        y += row_mm
    y += 0.3 * row_mm * bool(title)
    for part, width in zip(parts, widths):
        row_y = y
        for name, entries, italic in part:
            if name is not None:
                ax.text(x, row_y + row_mm / 2, name, fontsize=LEGEND_FONT, fontweight="bold", fontstyle="italic",
                        color=TEXT_COLOR, va="center")
                row_y += row_mm
            for label, color in entries:
                ax.scatter(x + 0.5, row_y + row_mm / 2, s=LEGEND_MARKER ** 2, color=color, linewidths=0, gid="legend_dot")
                ax.text(x + LEGEND_TEXT_MM, row_y + row_mm / 2, label, fontsize=LEGEND_FONT, color=TEXT_COLOR, va="center",
                        fontstyle="italic" if italic else "normal")
                row_y += row_mm
            if name is not None:
                row_y += BLOCK_GAP * row_mm
        x += width


def column_header(ax, x, y, text, width_mm, rule_above=False):
    """A line naming a column of trees, with its baseline at y, centred on a rule that spans the column: below the
    text, or above it (rule_above) for a header under the trees it names."""
    ax.text(x + width_mm / 2, y, text, fontsize=COLUMN_HEADER_FONT, color=TEXT_COLOR, va="baseline", ha="center")
    rule_y = y - COLUMN_RULE_ABOVE_MM if rule_above else y + COLUMN_RULE_MM
    ax.add_line(Line2D((x, x + width_mm), (rule_y, rule_y), color=MUTED_TEXT, linewidth=0.4, solid_capstyle="butt"))


def score_table(columns, rows):
    """A table for draw_score_table. columns: (name of the score, "higher" or "lower" as the better direction, number
    format); rows: (name of the tree, score, ...), with numeric scores. The best score of each column is set in bold."""
    best = [(max if direction == "higher" else min)(row[i + 1] for row in rows) for i, (_, direction, _) in enumerate(columns)]
    return dict(columns=tuple(name for name, _, _ in columns),
                rows=[(row[0], *(format(value, number) for value, (_, _, number) in zip(row[1:], columns))) for row in rows],
                bold=[tuple(value == top for value, top in zip(row[1:], best)) for row in rows])


def score_table_geometry(table):
    """The column widths of a score table, and its total width and height (mm)."""
    names = max(text_width_mm(row[0]) for row in table["rows"])
    widths = [max(text_width_mm(table["columns"][i]),
                  *(text_width_mm(row[i + 1], fontweight="bold") for row in table["rows"]))
              for i in range(len(table["columns"]))]
    width = names + sum(widths) + SCORE_TABLE_GAP_MM * len(widths)
    return names, widths, width, (1 + len(table["rows"])) * SCORE_TABLE_ROW_MM


def draw_score_table(ax, table, x, y):
    """A table from score_table with its top left corner at (x, y), in an axes with data units in mm from the top left:
    the names of the scores above the scores, which are right-aligned."""
    names, widths, _, _ = score_table_geometry(table)
    for r, (row, bold) in enumerate(zip((("", *table["columns"]), *table["rows"]), ([False] * len(widths), *table["bold"]))):
        row_y = y + (r + 0.5) * SCORE_TABLE_ROW_MM
        ax.text(x, row_y, row[0], fontsize=LEGEND_FONT, color=MUTED_TEXT, va="center")
        right = x + names
        for value, width, is_bold in zip(row[1:], widths, bold):
            right += SCORE_TABLE_GAP_MM + width
            ax.text(right, row_y, value, fontsize=LEGEND_FONT, color=MUTED_TEXT, va="center", ha="right",
                    fontweight="bold" if is_bold else "normal")


def rich_text(ax, x, y, parts, **common):
    """Text of several styles on one line, from (x, y): ``parts`` is a list of (text, style)."""
    for text, style in parts:
        ax.text(x, y, text, **common, **style)
        x += text_width_mm(text, **{"fontsize": common.get("fontsize", LEGEND_FONT), **style})


def mm_canvas(fig, width_mm, height_mm):
    """A transparent axes over the whole figure with data units in mm from its top left corner."""
    ax = fig.add_axes((0, 0, 1, 1), zorder=3)
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)
    ax.set_axis_off()
    return ax


def layout_ink(grid, layout, units_per_mm, left, top, size_mm):
    """Mark the cells of ``grid`` (cells of INK_GRID_MM, from the top left) that a drawn tree covers."""
    center, _ = centered_extent(layout)

    def cells(xy):
        columns = (left + size_mm[0] / 2 + (xy[:, 0] - center[0]) / units_per_mm) / INK_GRID_MM
        rows = (top + size_mm[1] / 2 - (xy[:, 1] - center[1]) / units_per_mm) / INK_GRID_MM
        keep = (rows >= 0) & (rows < grid.shape[0]) & (columns >= 0) & (columns < grid.shape[1])
        grid[rows[keep].astype(int), columns[keep].astype(int)] = True

    cells(layout.xy[:len(layout.names)])
    for start, end in layout.xy[layout.edges]:
        steps = int(np.linalg.norm(end - start) / units_per_mm / (INK_GRID_MM / 2)) + 2
        cells(np.linspace(start, end, steps))


def legend_ink(grid, width_mm, height_mm):
    """Ink under a legend at each top left corner, in grid cells, keeping INK_MARGIN_MM from the ink."""
    margin = int(np.ceil(INK_MARGIN_MM / INK_GRID_MM))
    grown = grid.copy()
    for shift in range(1, margin + 1):
        grown[shift:] |= grid[:-shift]
        grown[:-shift] |= grid[shift:]
    wide = grown.copy()
    for shift in range(1, margin + 1):
        wide[:, shift:] |= grown[:, :-shift]
        wide[:, :-shift] |= grown[:, shift:]
    h, w = int(np.ceil(height_mm / INK_GRID_MM)), int(np.ceil(width_mm / INK_GRID_MM))
    if h > wide.shape[0] or w > wide.shape[1]:
        return None
    total = np.zeros((wide.shape[0] + 1, wide.shape[1] + 1))
    total[1:, 1:] = wide.cumsum(axis=0).cumsum(axis=1)
    return total[h:, w:] - total[:-h, w:] - total[h:, :-w] + total[:-h, :-w]


def free_spot(grid, width_mm, height_mm):
    """Top left corner (mm) of the legend position that covers the least ink, and that ink in mm2;
    among equally good positions, the one nearest to the edge of the free area's corner."""
    ink = legend_ink(grid, width_mm, height_mm)
    if ink is None:
        return None, np.inf
    rows, columns = np.mgrid[:ink.shape[0], :ink.shape[1]]
    to_edge = np.minimum(rows, ink.shape[0] - 1 - rows) + np.minimum(columns, ink.shape[1] - 1 - columns)
    row, column = np.unravel_index(np.argmin(ink * 1e4 + to_edge), ink.shape)
    return (column * INK_GRID_MM, row * INK_GRID_MM), ink[row, column] * INK_GRID_MM ** 2


def closest_spot(grid, width_mm, height_mm):
    """Top left corner (mm) of the legend position nearest to the tree at the left, and the ink it covers (mm2)."""
    ink = legend_ink(grid, width_mm, height_mm)
    if ink is None:
        return None, np.inf
    rows, columns = np.nonzero(ink == ink.min())
    first = np.lexsort((rows, columns))[0]   # the leftmost, then the highest
    return (columns[first] * INK_GRID_MM, rows[first] * INK_GRID_MM), ink.min() * INK_GRID_MM ** 2


def cropped_tree_image(path):
    """A rendered tree on white, without the margins of the file, as a square, so that it is drawn as large as possible."""
    image = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    white.alpha_composite(image)
    image = white.convert("RGB")
    rows, columns = np.nonzero(np.asarray(image).min(axis=2) < 250)
    side = max(np.ptp(rows), np.ptp(columns)) + 1
    x_start, y_start = (columns.min() + columns.max() - side) // 2, (rows.min() + rows.max() - side) // 2
    return image.crop((x_start, y_start, x_start + side, y_start + side))


def middle_spot(grid, width_mm, height_mm, middle_mm):
    """Like free_spot, but among equally good positions the one vertically nearest to ``middle_mm``, then
    the rightmost."""
    ink = legend_ink(grid, width_mm, height_mm)
    if ink is None:
        return None, np.inf
    rows, columns = np.mgrid[:ink.shape[0], :ink.shape[1]]
    offset = np.abs((rows + height_mm / INK_GRID_MM / 2) - middle_mm / INK_GRID_MM)
    row, column = np.unravel_index(np.argmin(ink * 1e6 + offset * 1e3 - columns), ink.shape)
    return (column * INK_GRID_MM, row * INK_GRID_MM), ink[row, column] * INK_GRID_MM ** 2


def save_figure(fig, path, close=True):
    """Save at the exact figure size (no tight cropping), so all panels keep the same scale."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):   # the paper style crops otherwise
        for extension in FIGURE_FORMATS:
            fig.savefig(path.with_suffix(f".{extension}"), dpi=600)
    if close:
        plt.close(fig)


def tree_grid(trees, output, *, mode, colors, genus_colors=None, legend_blocks, legend_per_row=False, legend_inside=False,
              legend_middle=False, tree_width_mm=50, height_mm=None, tree_height_mm=50, scale_by="figure", parts_dir=None,
              heading=None, edge_width=EDGE_WIDTH, tip_diameter=TIP_DIAMETER, fill_together=False, note_offset_mm=0,
              score_table=None, score_table_top_mm=0, transparent=False, header_bottom=False):
    """Grid of trees with a legend; saves ``output`` as SVG, PDF and PNG.

    ``trees`` is a list of rows; each tree is a dict with file, title and, optionally, note (shown
    next to the title, e.g. the score) and part (file name of the tree alone). Trees are rotated or mirrored
    onto the first tree of their row (the first row: of the figure). legend_inside puts the legend
    in the white space between the trees, in as few columns as fit, with
    legend_middle as near as possible to the vertical middle of the figure, at the right; otherwise it is at their right,
    centred, and with legend_per_row ``legend_blocks`` holds one legend per row. scale_by "figure" draws
    all trees at one scale, "row" one per row. height_mm fixes the height of the figure, sharing it
    among the rows; otherwise each tree gets tree_height_mm. parts_dir also saves each tree (without
    title) and the legend as separate files at the same scale. In a notebook, the figure also appears in
    the output cell. heading (parts of rich_text) is a line above the trees that names what they show. legend_blocks
    may be None, for trees whose legend is elsewhere; a title may be empty, leaving only the note. subtitle is a
    second line below the title; corner is a list of lines at the top right of the tree, in a smaller font. fill_together rotates
    all trees by one angle so that they fill their panels, for panels much taller or wider than the trees.
    note_offset_mm moves the notes down, to line them up with the subtitles of species_tree_grid. A tree may have a
    header (a line above its title that names its column, see column_header) and title_offset_mm (moves its title down,
    to line it up with the titles below a header). score_table (see draw_score_table) goes at the top right of the
    figure, score_table_top_mm from its top, centred. header_bottom puts the column headers in a band under the trees
    instead of above them. transparent leaves out the background, for a panel that reaches over its neighbour in a composite.
    """
    use_pub_style()
    footer_mm = COLUMN_FOOTER_MM if header_bottom else 0
    if height_mm is None:
        height_mm = len(trees) * (tree_height_mm + TITLE_MM + ROW_GAP_MM) + footer_mm
    else:
        tree_height_mm = (height_mm - footer_mm) / len(trees) - TITLE_MM - ROW_GAP_MM
    row_mm = tree_height_mm + TITLE_MM + ROW_GAP_MM
    heading_mm = HEADING_MM if heading else 0
    height_mm += heading_mm
    legends = legend_blocks if legend_per_row else [legend_blocks]
    legend_width = 0 if legend_inside or not legend_blocks else max(legend_geometry(blocks)[2] for blocks in legends) + 1
    width_mm = len(trees[0]) * tree_width_mm + legend_width
    tree_size = np.array((tree_width_mm - 2, tree_height_mm))

    layouts = [[tree_layout(tree["file"], mode, colors, genus_colors, aspect=tree_size[0] / tree_size[1])
                for tree in row] for row in trees]
    for r, row in enumerate(layouts):
        for c, layout in enumerate(row):
            if (r, c) != (0, 0):
                align_layout(layout, layouts[0][0] if c == 0 else row[0])
    if fill_together:
        rotate_together([layout for row in layouts for layout in row], tree_size)

    def units_per_mm(group):
        return float(np.max(np.max([centered_extent(layout)[1] for layout in group], axis=0) / (tree_size - 1)))

    scales = ([units_per_mm(row) for row in layouts] if scale_by == "row"
              else [units_per_mm([layout for row in layouts for layout in row])] * len(layouts))

    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    grid = np.zeros((int(height_mm / INK_GRID_MM), int(width_mm / INK_GRID_MM)), bool)

    def axes_mm(x, y, w, h):
        return fig.add_axes((x / width_mm, 1 - (y + h) / height_mm, w / width_mm, h / height_mm))

    tree_ink = np.zeros_like(grid)
    if transparent:
        fig.patch.set_visible(False)
    for r, row in enumerate(trees):
        for c, tree in enumerate(row):
            left, top = c * tree_width_mm, heading_mm + r * row_mm
            draw_tree(axes_mm(left + 1, top + TITLE_MM, *tree_size), layouts[r][c], scales[r], tree_size, edge_width, tip_diameter)
            layout_ink(tree_ink, layouts[r][c], scales[r], left + 1, top + TITLE_MM, tree_size)
            grid |= tree_ink
            title_y = top + TITLE_BASELINE_MM + tree.get("title_offset_mm", 0)
            header_mm = TITLE_MM + tree.get("title_offset_mm", 0) + (SUBTITLE_MM if tree.get("subtitle") else 0)
            grid[int(top / INK_GRID_MM):int((top + header_mm) / INK_GRID_MM)] = True
            title = fig.text((left + 1) / width_mm, 1 - title_y / height_mm, tree["title"], fontsize=7,
                             fontweight="bold", color=TEXT_COLOR, va="baseline")
            if tree.get("note"):
                title_end = title.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
                fig.text(title_end + (1.5 if tree["title"] else 0) / width_mm, 1 - (top + TITLE_BASELINE_MM + note_offset_mm) / height_mm, tree["note"], fontsize=6,
                         color=TEXT_COLOR, va="baseline")
            if tree.get("subtitle"):   # second line, as in species_tree_grid
                fig.text((left + 1) / width_mm, 1 - (top + TITLE_BASELINE_MM + SUBTITLE_MM) / height_mm, tree["subtitle"],
                         fontsize=6, color=TEXT_COLOR, va="baseline")
            right = left + tree_width_mm - 1
            for line_number, line in enumerate(tree.get("corner", ())):   # scores at the top right of the tree
                baseline = top + CORNER_TOP_MM + line_number * CORNER_ROW_MM
                fig.text(right / width_mm, 1 - baseline / height_mm, line, fontsize=CORNER_FONT,
                         color=TEXT_COLOR, va="baseline", ha="right")
                first = right - text_width_mm(line, fontsize=CORNER_FONT)
                grid[int((baseline - CORNER_ROW_MM) / INK_GRID_MM):int(baseline / INK_GRID_MM),
                     int(first / INK_GRID_MM):int(right / INK_GRID_MM)] = True
    canvas = mm_canvas(fig, width_mm, height_mm)
    if score_table:
        table_width, table_height = score_table_geometry(score_table)[2:]
        table_x, table_y = (width_mm - table_width) / 2, heading_mm + score_table_top_mm
        ink = legend_ink(tree_ink, table_width, table_height)[int(table_y / INK_GRID_MM), int(table_x / INK_GRID_MM)] * INK_GRID_MM ** 2
        print(f"{Path(output).stem}: score table covering {round(ink, 1)} mm2 of tree, with its margin")
        draw_score_table(canvas, score_table, table_x, table_y)
        grid[int(table_y / INK_GRID_MM):int((table_y + table_height) / INK_GRID_MM), int(table_x / INK_GRID_MM):] = True
    for r, row in enumerate(trees):
        for c, tree in enumerate(row):
            if tree.get("header"):
                header_y = (height_mm - COLUMN_FOOTER_MM if header_bottom else heading_mm + r * row_mm) + COLUMN_HEADER_BASELINE_MM
                column_header(canvas, c * tree_width_mm + 1, header_y, tree["header"], tree_width_mm - 2,
                              rule_above=header_bottom)
    if footer_mm:
        grid[int((height_mm - footer_mm) / INK_GRID_MM):] = True
    if heading:
        grid[:int(heading_mm / INK_GRID_MM)] = True
        rich_text(canvas, 1, TITLE_BASELINE_MM, heading, fontsize=7, color=TEXT_COLOR, va="baseline")
    columns = 1
    if legend_blocks and legend_inside:
        edge = int(1 / INK_GRID_MM)   # in line with the titles
        grid[:, :edge] = grid[:, -edge:] = grid[-edge:] = True
        options = []
        for columns in range(1, 4):
            _, _, width, height = legend_geometry(legend_blocks, columns)
            spot, ink = middle_spot(grid, width, height, height_mm / 2) if legend_middle else free_spot(grid, width, height)
            if spot is not None:
                options.append((round(ink, 1), columns, spot))
        ink, columns, spot = min(options, key=lambda option: option[:2])
        print(f"{Path(output).stem}: legend in {columns} column(s), covering {ink} mm2 of tree")
        draw_legend(canvas, legend_blocks, *spot, columns=columns)
    elif legend_blocks and legend_per_row:
        for r, blocks in enumerate(legends):
            draw_legend(canvas, blocks, width_mm - legend_width + 0.5,
                        heading_mm + r * row_mm + (row_mm - legend_geometry(blocks)[3]) / 2)
    elif legend_blocks:
        draw_legend(canvas, legend_blocks, width_mm - legend_width + 0.5, (height_mm - legend_geometry(legend_blocks)[3]) / 2)
    save_figure(fig, output, close=False)

    if parts_dir:
        stem = Path(output).stem
        for r, row in enumerate(trees):
            for c, tree in enumerate(row):
                part = plt.figure(figsize=(tree_size[0] * MM, tree_size[1] * MM))
                draw_tree(part.add_axes((0, 0, 1, 1)), layouts[r][c], scales[r], tree_size)
                save_figure(part, Path(parts_dir) / f"{stem}_{tree['part']}")
        for blocks in legends:
            _, _, width, height = legend_geometry(blocks, columns)
            part = plt.figure(figsize=(width * MM, height * MM))
            draw_legend(mm_canvas(part, width, height), blocks, 0, 0, columns=columns)
            save_figure(part, Path(parts_dir) / f"{stem}_legend")
    plt.show()   # in a notebook: the figure in the output cell


def genus_legend(genus_colors, genera):
    """Legend blocks with one entry per genus, in alphabetical order."""
    return [(None, [(genus, genus_colors[genus])], True) for genus in sorted(genera)]


def species_tree_grid(rows, output, *, colors, legend_blocks, width_mm, row_mm, tip_mm=None, legend_columns=2,
                      abbreviate=False, linewidth=0.6, legend_left_mm=None, legend_top_mm=None, header_bottom=False):
    """Grid of rectangular trees of one sequence per species, with the species names in their colours and a
    legend, if any, in the white space near the vertical middle; saves ``output`` as SVG, PDF and PNG.

    ``rows`` is a list of rows; each tree is a dict with tree (a rooted toytree), title and, optionally, note
    (shown next to the title), header (a line above the title that names the column, see column_header) with header_rule_mm
    (the length of its rule, by default the width of the tree), subtitle (a line below it), scale (trees with the same scale are drawn at one scale)
    and support (text to write at the node of a clade, keyed by the clade's tips as a frozenset; a clade the tree does not
    hold has no node and so no text). width_mm is the
    width of a tree, row_mm the height of a row, or one height per row. tip_mm is the distance between tips, the
    same in all trees; by default each tree fills its row. abbreviate writes the genus as its initial (its colour still
    tells the genus) and draws the branches as long as the names of the tips then allow, for small trees. legend_left_mm
    fixes the distance of the legend from the left edge; only its height is searched then. legend_top_mm fixes that too. header_bottom puts the column headers in a band under the trees instead of above them.
    """
    use_pub_style()
    heights = [row_mm] * len(rows) if np.isscalar(row_mm) else list(row_mm)
    trees = [tree for row in rows for tree in row]
    labels = {name: parse_tip(name, "scientific") for tree in trees for name in tree["tree"].get_tip_labels()}
    def label_text(key):
        return f"{key[0][0]}. {key[1]}" if abbreviate else " ".join(key)

    label_widths = {key: text_width_mm(label_text(key), fontsize=SPECIES_FONT, fontstyle="italic") for key in set(labels.values())}
    label_mm = max(label_widths.values())
    depths, longest, tight = {}, {}, {}
    for tree in trees:
        depth = np.zeros(tree["tree"].nnodes)
        for node in tree["tree"].traverse("preorder"):
            for child in node.children:
                depth[child.idx] = depth[node.idx] + child.dist
        depths[id(tree)] = depth
        longest[tree.get("scale")] = max(longest.get(tree.get("scale"), 0), depth.max())
    mm_per_unit = {scale: (width_mm - label_mm - 3) / length for scale, length in longest.items()}
    if abbreviate:   # the tip that reaches furthest with its name sets the scale
        for tree in trees:
            for node in tree["tree"]:
                if node.is_leaf():
                    room = (width_mm - 2.5 - label_widths[labels[node.name]]) / depths[id(tree)][node.idx]
                    tight[tree.get("scale")] = min(tight.get(tree.get("scale"), room), room)
        mm_per_unit = tight

    total_mm, height_mm = width_mm * max(len(row) for row in rows), sum(heights)
    fig = plt.figure(figsize=(total_mm * MM, height_mm * MM))
    canvas = mm_canvas(fig, total_mm, height_mm)
    grid = np.zeros((int(height_mm / INK_GRID_MM), int(total_mm / INK_GRID_MM)), bool)
    renderer = fig.canvas.get_renderer()
    for r, row in enumerate(rows):
        top = sum(heights[:r])
        for c, tree in enumerate(row):
            left = c * width_mm + 1
            depth = depths[id(tree)]
            scale = mm_per_unit[tree.get("scale")]
            header_mm = 0 if header_bottom else (SUBTITLE_MM if tree.get("header") else 0)
            if tree.get("header"):
                header_y = (height_mm - COLUMN_FOOTER_MM if header_bottom else top) + COLUMN_HEADER_BASELINE_MM
                column_header(canvas, left, header_y, tree["header"], tree.get("header_rule_mm", width_mm - 2),
                              rule_above=header_bottom)
            title = canvas.text(left, top + TITLE_BASELINE_MM + header_mm, tree["title"], fontsize=7, fontweight="bold",
                                color=TEXT_COLOR, va="baseline")
            if tree.get("note"):
                title_end = title.get_window_extent(renderer).x1 / fig.dpi / MM
                canvas.text(title_end + 1.5, top + TITLE_BASELINE_MM + header_mm, tree["note"], fontsize=6, color=TEXT_COLOR, va="baseline")
            subtitle_mm = 0
            if tree.get("subtitle"):
                subtitle_mm = SUBTITLE_MM
                canvas.text(left, top + TITLE_BASELINE_MM + header_mm + subtitle_mm, tree["subtitle"], fontsize=6, color=TEXT_COLOR, va="baseline")
            tips = tree["tree"].ntips
            support, clades = tree.get("support") or {}, {}
            footer_mm = COLUMN_FOOTER_MM if header_bottom and r == len(rows) - 1 else 0
            step = tip_mm or (heights[r] - TITLE_MM - header_mm - subtitle_mm - ROW_GAP_MM - footer_mm) / tips
            y = np.zeros(tree["tree"].nnodes)
            segments = []
            for node in tree["tree"]:                  # from the tips to the root
                if node.is_leaf():
                    y[node.idx] = top + TITLE_MM + header_mm + subtitle_mm + (tips - node.idx - 0.5) * step
                    clades[node.idx] = frozenset((node.name,))
                    key = labels[node.name]
                    canvas.text(left + depth[node.idx] * scale + 1, y[node.idx], label_text(key), fontsize=SPECIES_FONT,
                                fontstyle="italic", color=colors[key], va="center")
                else:
                    y[node.idx] = np.mean([y[child.idx] for child in node.children])
                    x = left + depth[node.idx] * scale
                    clades[node.idx] = frozenset().union(*(clades[child.idx] for child in node.children))
                    if clades[node.idx] in support:
                        # above the branch that leads into the node, against the node; a white outline
                        # keeps it readable where a short branch leaves it over the tree's own lines
                        canvas.text(x - SUPPORT_OFFSET_MM[0], y[node.idx] - SUPPORT_OFFSET_MM[1],
                                    support[clades[node.idx]], fontsize=SUPPORT_FONT, color=MUTED_TEXT,
                                    ha="right", va="center", zorder=4,
                                    path_effects=[patheffects.withStroke(linewidth=1.3, foreground="white")])
                    segments.append(((x, min(y[n.idx] for n in node.children)), (x, max(y[n.idx] for n in node.children))))
                    segments.extend(((x, y[n.idx]), (left + depth[n.idx] * scale, y[n.idx])) for n in node.children)
            canvas.add_collection(LineCollection(segments, colors=TEXT_COLOR, linewidths=linewidth, capstyle="round"))
            for start, end in segments:
                cells = (np.linspace(start, end, 200) / INK_GRID_MM).astype(int)
                grid[cells[:, 1].clip(0, grid.shape[0] - 1), cells[:, 0].clip(0, grid.shape[1] - 1)] = True
    for text in canvas.texts:
        box = text.get_window_extent(renderer)
        first, last = int(box.x0 / fig.dpi / MM / INK_GRID_MM), int(np.ceil(box.x1 / fig.dpi / MM / INK_GRID_MM))
        upper, lower = (int((height_mm - y / fig.dpi / MM) / INK_GRID_MM) for y in (box.y1, box.y0))
        grid[upper:lower + 1, first:last] = True
    edge = int(1 / INK_GRID_MM)
    grid[:, :edge] = grid[:, -edge:] = grid[-edge:] = True
    if header_bottom:
        grid[int((height_mm - COLUMN_FOOTER_MM) / INK_GRID_MM):] = True
    if legend_blocks:
        spot, ink = middle_spot(grid, *legend_geometry(legend_blocks, legend_columns)[2:], height_mm / 2)
        if legend_left_mm is not None:
            spot = legend_left_mm, spot[1] if legend_top_mm is None else legend_top_mm
        print(f"{Path(output).stem}: legend covering {round(ink, 1)} mm2 of tree")
        draw_legend(canvas, legend_blocks, *spot, columns=legend_columns)
    save_figure(fig, output, close=False)
    plt.show()   # in a notebook: the figure in the output cell
