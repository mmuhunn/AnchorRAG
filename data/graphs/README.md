# Derived Topic Graphs

This directory contains the seven scored topic graphs used by the main 210-query
live rerun. Each graph is a derived JSON artifact with nodes, edges, stance
labels, source names, derived trust/polarity scores, and community ids.

## What is included

For every node we keep the derived/structural fields needed to reproduce the
paper: `title`, `heading`, `source`, `tags`, `stance_label`, `bias_rating`,
`political_score_avg`, `community`, `community_purity`, `S_trust`, `P_polar`,
and the node `id`. Edges (`links`) and graph-level metadata are kept as-is.

## What is removed

The raw article body (`text`) on each node has been blanked out in the public
release because the underlying corpus (AllSides/Qbias news articles from
outlets such as Politico, Fox News, etc.) is third-party content that this
repository cannot redistribute in full.

Cached reproduction (`make reproduce-cached`) and the artifact audit
(`make audit`) do not need the raw body and continue to pass on the stripped
graphs. Only the live generation path needs the body.

## Re-hydrating the graphs for live reproduction

Obtain the AllSides/Qbias topic subset CSVs (each file follows the schema
`title, tags, heading, source, text, bias_rating`) and pass the directory to
the hydrate script:

```bash
python3.8 scripts/hydrate_graphs.py --subset-dir /path/to/AllSides_Qbias/data/subsets
```

The script matches each node by `title` against the subset CSV of the same
topic and writes the `text` field back into the graph file in place. Use
`--check` to dry-run and report match coverage without modifying anything.

Once hydrated, `make live-dry-run` reports `graph_text_hydration: 279/279`
style counts per topic, and `make reproduce-main-live` can proceed.
