Name structure:
{Instance Group}-{Instance}_{Connectivity Mode}_{Variant}_sol.json

Instance Groups:
  i - medium
  l - large
  s - sparse

Connectivity Modes:
  arc_sep  - Rooted Arc Separators
  scf      - Single-Commodity-Flow

Variants (all variants include z variables, Extended Indegree Cuts and Root-Ring-Cuts, and use Bicolor Radius Algorithm):
  2 - Fixing Nodes (FN)
  3 - FN + Conditional Fixes (CF)
  4 - FN + CF + Conflict Pairs (CP)
  5 - FN + CF + CP + Scoring