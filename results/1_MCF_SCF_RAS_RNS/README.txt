Name structure:
{Instance Group}-{Instance}_{Connectivity Mode}_{Variant}_sol.json

Instance Groups:
  i - medium
  l - large
  s - sparse

Connectivity Modes:
  node_sep - Rooted Node Separators
  arc_sep  - Rooted Arc Separators
  scf      - Single-Commodity-Flow
  mfc	   - Multi-Commodity-Flow (is not listed since no instance could be solve within 1h)

Variants (all variants use the Bicolor Radius Algorithm):
  1 - plain
  2 - with z variables