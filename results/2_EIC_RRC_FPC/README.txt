Name structure:
{Instance Group}-{Instance}_{Connectivity Mode}_{Variant}_sol.json

Instance Groups:
  i - medium
  l - large
  s - sparse

Connectivity Modes:
  arc_sep  - Rooted Arc Separators
  scf      - Single-Commodity-Flow

Variants (all variants include z variables and use Bicolor Radius Algorithm):
  3 - Extended Indegree Cuts (EIC)
  4 - Root-Ring-Cuts (RRC)
  5 - EIC + RRC
  6 - Fine Path Cuts (FPC)
  7 - EIC + RRC + FPC