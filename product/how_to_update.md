# How to update the influence map

This guide is for someone at Animal Ask who wants to regenerate the policy influence
map with improved or updated citation classifications. No changes to the core code
are required — the pipeline is designed so that improving the classifier automatically
flows through to the finished graphic.

---

## What the diagram shows

The influence map visualises how often farmed animal welfare legislation in Denmark (DK),
Germany (DE), and Austria (AT) explicitly cites another country's law or policy as a
reference point. Each arrow represents a body of citations identified in parliamentary
legislation; the arrow's width scales with the count (N), and the colour indicates
whether the citation is *informational* (learning from another country's experience)
or *cautionary* (referencing a country as a negative example or laggard).

The study was limited by the quality of the automated citation classifier. A better
classifier — one that more accurately distinguishes real policy citations from noise,
and that more reliably assigns the correct mechanism type — will produce more accurate
N values and arrow colours. This guide shows you how to feed that improvement through
to the map.

---

## Prerequisites

You will need:

- **Python 3.10+** and **Jupyter** (Jupyter Lab or Jupyter Notebook)
- The Python packages listed in `product/product_requirements.txt`
  (`matplotlib`, `numpy`, `cartopy`). Install them with:
  ```
  pip install -r product/product_requirements.txt
  ```
- An **Anthropic API key** in the environment variable `ANTHROPIC_API_KEY` — only
  needed if you are rerunning the LLM classification step (Option A below).
  If you are supplying your own classifications (Option B), no API key is needed.
- The full repository, including the `data/` folder with the corpus RDS files.

The `product/` notebook (`influence_map.ipynb`) has its own `%pip install` cell at
the top that installs all its dependencies when you run it.

---

## File map

```
parlawspeechCC_v2.ipynb        ← study pipeline; produces the edge JSON
data/
  citation_classifications.csv ← classification cache (empty in current repo)
  influence_edges.json         ← auto-generated edge counts; read by product notebook
product/
  influence_map.ipynb          ← generates the map; reads influence_edges.json
  product_requirements.txt     ← Python dependencies for the product notebook
  output/
    influence_map_base.svg     ← primary Illustrator import (text editable)
    influence_map_base.pdf     ← print-ready backup
```

The pipeline runs left-to-right: **study notebook → `influence_edges.json` →
product notebook → SVG/PDF**.

---

## Step 1 — Improve the classification, then generate `influence_edges.json`

Choose whichever option matches your situation.

### Option A: Rerun with a better LLM classifier

The study uses Claude Haiku to classify each citation context. To upgrade:

1. Open `parlawspeechCC_v2.ipynb` in Jupyter.
2. Go to **Cell 32** (labelled `CLASSIFICATION_MODEL`). You will see:
   ```python
   CLASSIFICATION_MODEL = 'claude-haiku-4-5-20251001'
   SYSTEM_PROMPT = """..."""
   ```
3. Change `CLASSIFICATION_MODEL` to a more capable model, for example:
   ```python
   CLASSIFICATION_MODEL = 'claude-sonnet-4-6'
   ```
   You can also edit `SYSTEM_PROMPT` in the same cell to refine the classification
   instructions.
4. **Delete `data/citation_classifications.csv`** (or clear its contents). The
   notebook skips classification if the cache file exists — deleting it forces a
   fresh run.
5. Run the notebook from **Cell 32 through to the end**. The classification step
   (Cell 35) will call the API for each citation context and write results to
   `data/citation_classifications.csv`. Depending on corpus size, this may take
   several minutes and will incur API costs.
6. Once complete, the **export cell** (labelled `export-influence-edges`, just after
   the within-trio matrix heatmap) writes `data/influence_edges.json` automatically.

### Option B: Supply your own classifications

If you have a human-annotated or externally-classified dataset, you can bypass the
LLM step entirely by providing the classifications as a CSV.

The file must be saved to `data/citation_classifications.csv` and contain at minimum
these columns:

| Column | Description | Example values |
|--------|-------------|----------------|
| `citing_country` | ISO code of the document's country | `DE`, `AT`, `DK` |
| `cited_country` | ISO code of the country being cited | `SE`, `CH`, `NL` |
| `cit_types` | Pipe-separated mechanism/exclusion types | `informational` · `cautionary` · `informational\|legitimating` · `false_positive` · `out_of_scope` · `eu_implementation` |
| `cit_confidence` | Your confidence in the classification | `high`, `medium`, `low` |
| `doc_type` | Type of source document | `bill`, `law` |

Valid values for `cit_types` are: `informational`, `cautionary`, `legitimating`,
`false_positive`, `out_of_scope`, `eu_implementation`, `unclear`, `parse_error`.
Multiple mechanism types can be combined with `|` (e.g. `informational|cautionary`).
Exclusion types (`false_positive`, `out_of_scope`, `eu_implementation`) should
appear alone.

Once the CSV is in place:

1. Open `parlawspeechCC_v2.ipynb` in Jupyter.
2. Run **from Cell 35 to the end**. Cell 35 will detect the CSV and load it
   without calling the API. The export cell will then write `data/influence_edges.json`.

---

## Step 2 — Regenerate the map

1. Open `product/influence_map.ipynb` in Jupyter.
2. Run **all cells** (Kernel → Restart and Run All).
   - The first cell installs Python dependencies.
   - The data cell checks for `../data/influence_edges.json`. If it exists, the map
     uses those counts. If it is missing or empty, the notebook falls back to the
     original hardcoded values and prints a warning in orange at the bottom of the map.
   - The drawing cell generates the map and saves it to `product/output/`.
3. Check the note at the bottom-left of the map: it will say either
   **"Pipeline data · N analytical citations"** (grey, meaning your updated counts
   are in use) or **"Hardcoded fallback"** (orange, meaning the JSON was not found).

The map is saved as:
- `product/output/influence_map_base.svg` — import this into Illustrator
- `product/output/influence_map_base.pdf` — for direct printing or sharing

---

## Step 3 — Illustrator finishing

The SVG preserves text as editable text objects (not outlines). Recommended
annotation layers to add on top of the base layer in Illustrator:

- **"Not yet studied" arrows** for routes that appear likely from the literature
  but were outside this study's corpus (e.g. SE → DK, SE → DE). Use dashed strokes.
- **Country labels with full names** beneath the ISO codes if the audience is not
  familiar with abbreviations.
- **A title and subtitle** appropriate to the Animal Ask audience.
- **Source and date line** at the bottom (e.g. *Source: ParlLawSpeech V2; Bridgewater
  (2026); [your update date]*).

Do not edit the SVG file directly — it will be overwritten the next time the
notebook runs.

---

## Troubleshooting

**The map shows "Hardcoded fallback" even though I ran the study notebook.**
→ Check that `data/influence_edges.json` exists and is not empty. The export cell
should print "Saved: data/influence_edges.json" — scroll back through the Cell 42
output in the study notebook to confirm it ran.

**Classification is very slow or times out.**
→ The corpus contains hundreds of citation contexts. On a slow connection or with
a larger model, this can take 10–30 minutes. Let it complete. Partial results are
not saved, so do not interrupt mid-run.

**A country I expect to see is missing from the map.**
→ It either had zero analytical citations in your classifier's output, or it is
not in the `LON_LAT` dictionary in the product notebook's data cell. Open
`product/influence_map.ipynb`, find the `LON_LAT` dict, and add the country with
its approximate geographic centroid `(longitude, latitude)`.

**Cartopy downloads shapefiles on the first run.**
→ This is expected. Natural Earth map files (~10 MB total) are downloaded once and
cached in `~/.local/share/cartopy/`. Subsequent runs use the cache and are fast.
An internet connection is required only for this first run.

**`ANTHROPIC_API_KEY` not found.**
→ Set the environment variable before launching Jupyter:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
jupyter lab
```

---

## Summary of the update workflow

```
1. Improve classifier (Option A or B)
         ↓
2. Run parlawspeechCC_v2.ipynb → data/influence_edges.json
         ↓
3. Run product/influence_map.ipynb → product/output/influence_map_base.svg
         ↓
4. Import SVG into Illustrator and add annotation layers
```

No manual editing of N values or arrow colours is ever needed — all of that is
derived automatically from your classifications.

---

## Phrase / shibboleth search

`product/phrase_search.py` lets you search for a specific legislative term across
the corpus and see which countries use it and in what cross-citation context.
This is useful for tracking whether a specific phrase or provision seeded in one
country subsequently appears in another country's legislation.

**Prerequisites:** Python 3.10+ and the packages in `requirements.txt`.
Classification data (`data/citation_classifications.csv`) must exist — run the
main study notebook first to generate it.

### Basic usage

```bash
# Search citation contexts for a German welfare term:
python product/phrase_search.py --term "schwanzkupieren"

# Also search the full corpus (bills, laws, speeches):
python product/phrase_search.py --term "kastenstand" --corpus

# Print the matched text passages:
python product/phrase_search.py --term "kastenstand" --context

# Start from an English term — auto-translates to German and Danish
# (requires ANTHROPIC_API_KEY):
python product/phrase_search.py --term "tail docking" --translate
```

### Updating the influence map for a specific term

The `--export-edges` flag saves a filtered version of the edge weights —
only counting citations that appear near your search term:

```bash
python product/phrase_search.py --term "schwanzkupieren" --export-edges
```

Then to see the filtered map:

1. Rename `data/influence_edges.json` → `data/influence_edges_backup.json`
2. Rename `data/phrase_search_edges.json` → `data/influence_edges.json`
3. Run `product/influence_map.ipynb` (Kernel → Restart and Run All)
4. The map now shows only citations where "schwanzkupieren" appears in context
5. Restore: rename the backup back to `data/influence_edges.json`

### How "similar terms" works

By default the tool searches for:
1. The exact phrase you typed
2. Each individual word in a multi-word phrase
3. Morphological stems (common German/Danish suffixes stripped)

For example, `--term "schwanzkupieren"` also searches for `"kupieren"` and
the stem `"schwanzkupier"`, catching inflected forms like `"schwanzkupiert"`,
`"schwanzkupierten"`, etc.

Use `--exact` to disable this and match the literal string only.
