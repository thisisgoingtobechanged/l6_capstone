#!/usr/bin/env python3
"""
phrase_search.py  —  Shibboleth phrase tracker for ParlLawSpeech V2
--------------------------------------------------------------------
Given a search term, this tool:
  1. Searches the citation-context records (citation_classifications.csv) to
     find every cross-country citation that occurred near that term, then
     redraws the influence map restricted to those citations.
  2. Optionally searches the full corpus (bills, laws, speeches RDS files)
     to see every document that uses the term, regardless of whether it
     contains a cross-country citation.
  3. Finds *similar* terms automatically — morphological variants and related
     roots — so a search for "schwanzkupieren" also surfaces "kupieren",
     "schwanzkürzung", and inflected forms.

Usage examples
--------------
  # Search citation contexts only (fast; no RDS files needed):
  python product/phrase_search.py --term "schwanzkupieren"

  # Also search full corpus (requires RDS files in data/):
  python product/phrase_search.py --term "schwanzkupieren" --corpus

  # Print the text passages that matched:
  python product/phrase_search.py --term "kastenstand" --context

  # Auto-translate an English term to German/Danish before searching
  # (requires ANTHROPIC_API_KEY in the environment):
  python product/phrase_search.py --term "tail docking" --translate

  # Save the filtered influence-map edge weights (same format as
  # influence_edges.json) so you can regenerate the diagram:
  python product/phrase_search.py --term "kastenstand" --export-edges

Outputs
-------
  product/output/phrase_search_<term>.png   — bar chart + timeline
  data/phrase_search_edges.json             — filtered edge weights for the map
      (only written with --export-edges; load this in influence_map.ipynb by
       temporarily renaming it to data/influence_edges.json)
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent            # repo root
DATA_DIR    = ROOT / 'data'
OUTPUT_DIR  = Path(__file__).parent / 'output'
CLASSIF_CSV = DATA_DIR / 'citation_classifications.csv'
EDGES_JSON  = DATA_DIR / 'phrase_search_edges.json'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CORPUS CONSTANTS (must match parlawspeechCC_v2.ipynb) ─────────────────────

COUNTRIES   = {'austria': 'AT', 'germany': 'DE', 'denmark': 'DK'}
DOC_TYPES   = ['laws', 'bills', 'speeches']
TEXT_COLS   = {'laws': 'law_text', 'bills': 'bill_text', 'speeches': 'text'}
TITLE_COLS  = {'laws': 'title_law', 'bills': 'title_bill', 'speeches': 'title'}
DATE_COL    = 'date'

COUNTRY_COLORS = {'AT': '#2196F3', 'DE': '#F44336', 'DK': '#4CAF50'}
EXCL_TYPES     = {'false_positive', 'out_of_scope', 'eu_implementation', 'parse_error'}

# ── TEXT NORMALISATION ────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    return unicodedata.normalize('NFC', str(text)).lower()


# ── MORPHOLOGICAL RELAXATION ──────────────────────────────────────────────────
#
# Simple suffix-stripping for German and Danish.  Not a proper stemmer —
# just enough to surface the most common inflected forms without false positives.
# Handles: verb infinitives (-ieren, -en), nouns (-ung, -keit, -heit, -schaft,
# -tion, -tät), Danish nouns (-hed, -ning, -else, -sel), and plurals (-e, -en,
# -er, -s, -erne, -ene).

_DE_SUFFIXES = [
    'ierung', 'ieren', 'ierung', 'ungen', 'ung', 'keiten', 'keit',
    'heiten', 'heit', 'schaften', 'schaft', 'tionen', 'tion',
    'täten', 'tät', 'lichen', 'liche', 'licher', 'lich',
    'enden', 'ende', 'ener', 'ene', 'enes', 'enen',
    'ters', 'tern', 'ter', 'ten', 'tem', 'tes', 'te',
    'ers', 'ern', 'er', 'es', 'en', 'em', 's', 'e',
]
_DK_SUFFIXES = [
    'ninger', 'ning', 'heder', 'hed', 'elser', 'else',
    'sler', 'sel', 'erne', 'ene', 'ernes', 'enes',
    'ernes', 'ers', 'ens', 'er', 'es', 'en', 'et', 'e', 's',
]
_ALL_SUFFIXES = sorted(set(_DE_SUFFIXES + _DK_SUFFIXES), key=lambda s: -len(s))

MIN_STEM = 4  # never strip below this character length

def stem(word: str) -> str:
    """Return a simple morphological root for word (lowercase)."""
    w = normalise(word)
    for suffix in _ALL_SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= MIN_STEM:
            return w[: len(w) - len(suffix)]
    return w


def build_search_variants(query: str) -> list[str]:
    """
    Return a list of search strings to try, ordered by specificity:
      1. The exact normalised query (most specific)
      2. Individual words in a multi-word query
      3. Stems of individual words (broadest)
    Duplicates are removed while preserving order.
    """
    q = normalise(query)
    words = q.split()
    stems = [stem(w) for w in words]

    seen, variants = set(), []
    for v in [q] + words + stems:
        if v not in seen and len(v) >= MIN_STEM:
            seen.add(v)
            variants.append(v)
    return variants


# ── CITATION CSV SEARCH ───────────────────────────────────────────────────────

def _load_classifications() -> pd.DataFrame | None:
    """Load citation_classifications.csv; return None if absent or empty."""
    if not CLASSIF_CSV.exists():
        return None
    try:
        df = pd.read_csv(CLASSIF_CSV)
    except Exception as e:
        print(f'  Warning: could not read {CLASSIF_CSV}: {e}')
        return None
    if df.empty or 'context' not in df.columns:
        return None
    return df


def _parse_types(s):
    if not isinstance(s, str):
        return []
    return [t.strip() for t in s.split('|') if t.strip()]


def search_citation_contexts(
    query: str,
    similar: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Search the citation-context records for query (and similar variants).

    Returns
    -------
    matches : DataFrame
        Subset of classified_df where context contains a variant.
    variants_used : list[str]
        Which search strings actually produced hits.
    """
    df = _load_classifications()
    if df is None:
        print('  No citation_classifications.csv found (or file is empty).')
        print('  Run parlawspeechCC_v2.ipynb to generate it.')
        return pd.DataFrame(), []

    # Exclude noise rows
    excl_mask = df['cit_types'].apply(
        lambda s: any(t in EXCL_TYPES for t in _parse_types(s))
    )
    analytical = df[~excl_mask].copy()

    variants = build_search_variants(query) if similar else [normalise(query)]

    hit_mask = pd.Series(False, index=analytical.index)
    variants_used = []
    for v in variants:
        m = analytical['context'].str.contains(re.escape(v), na=False, regex=True)
        if m.any():
            hit_mask |= m
            variants_used.append(v)

    return analytical[hit_mask].copy(), variants_used


# ── CORPUS-WIDE SEARCH (optional) ────────────────────────────────────────────

def _try_load_rds(path: Path) -> pd.DataFrame | None:
    try:
        import pyreadr
        return pyreadr.read_r(str(path))[None]
    except ImportError:
        print('  pyreadr not installed — skipping corpus search.')
        print('  Run: pip install pyreadr')
        return None
    except Exception as e:
        print(f'  Could not load {path}: {e}')
        return None


def search_full_corpus(query: str, similar: bool = True) -> pd.DataFrame:
    """
    Search the full RDS corpus (bills, laws, speeches) for the query.
    Requires RDS files in data/Corpora_PLS_<country>/.
    Returns a DataFrame with columns: country, doc_type, title, date, match_count.
    """
    variants = build_search_variants(query) if similar else [normalise(query)]
    rows = []

    for country_name, code in COUNTRIES.items():
        for doc_type in DOC_TYPES:
            rds_path = DATA_DIR / f'Corpora_PLS_{country_name}' \
                                 / f'Corpus_{doc_type}_{country_name}.RDS'
            df = _try_load_rds(rds_path)
            if df is None:
                continue

            text_col  = TEXT_COLS.get(doc_type)
            title_col = TITLE_COLS.get(doc_type, 'title')
            if text_col not in df.columns:
                continue

            for _, row in df.iterrows():
                raw = normalise(str(row.get(text_col, '')))
                if not raw:
                    continue

                match_count = 0
                for v in variants:
                    match_count += len(re.findall(re.escape(v), raw))
                if match_count == 0:
                    continue

                rows.append({
                    'country':     code,
                    'doc_type':    doc_type,
                    'title':       str(row.get(title_col, ''))[:120],
                    'date':        row.get(DATE_COL, ''),
                    'match_count': match_count,
                })

    return pd.DataFrame(rows)


# ── EDGE EXPORT ───────────────────────────────────────────────────────────────

def export_edges(matches: pd.DataFrame, label: str = '') -> None:
    """
    Write a filtered influence_edges.json from the matched citation rows.
    Same format as the main pipeline export, so influence_map.ipynb can read it.
    """
    if matches.empty:
        print('  No matches to export.')
        return

    PEER = {'informational', 'cautionary', 'legitimating'}
    edges_out = []

    for (src, tgt), grp in matches.groupby(['citing_country', 'cited_country']):
        if src == tgt:
            continue
        mc = Counter(
            t for row in grp['cit_types']
            for t in _parse_types(row)
            if t in PEER
        )
        dominant = max(mc, key=mc.get) if mc else 'informational'
        edges_out.append({
            'src': str(src),
            'tgt': str(tgt),
            'n':   int(len(grp)),
            'mechanism': dominant,
            'mechanism_counts': dict(mc),
            'phrase_filter': label,
        })

    EDGES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EDGES_JSON, 'w') as f:
        json.dump(edges_out, f, indent=2)
    print(f'\n  Filtered edge weights saved to: {EDGES_JSON}')
    print('  To update the influence map with these counts:')
    print(f'    1. Rename data/influence_edges.json → data/influence_edges_backup.json')
    print(f'    2. Rename data/phrase_search_edges.json → data/influence_edges.json')
    print(f'    3. Run product/influence_map.ipynb (Kernel → Restart and Run All)')
    print(f'    4. Restore: rename the backup back to data/influence_edges.json')


# ── VISUALISATION ─────────────────────────────────────────────────────────────

def plot_results(
    ctx_matches: pd.DataFrame,
    corpus_hits: pd.DataFrame,
    query: str,
    variants_used: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print('  matplotlib not installed — skipping chart.')
        print('  Run: pip install matplotlib')
        return

    n_panels = 1 + (not corpus_hits.empty)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f'Phrase search: "{query}"'
        + (f'\n(variants searched: {", ".join(variants_used[:6])})' if len(variants_used) > 1 else ''),
        fontsize=12, fontweight='bold', y=1.02,
    )

    # ── Panel 1: citation-context hits by country pair ─────────────────────
    ax = axes[0]
    if ctx_matches.empty:
        ax.text(0.5, 0.5, 'No citation-context matches', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='grey')
        ax.set_title('Citation contexts containing term')
    else:
        pair_counts = (
            ctx_matches
            .groupby(['citing_country', 'cited_country'])
            .size()
            .reset_index(name='n')
        )
        pair_counts['pair'] = pair_counts['citing_country'] + ' → ' + pair_counts['cited_country']
        pair_counts = pair_counts.sort_values('n', ascending=False)
        colors = [COUNTRY_COLORS.get(r['citing_country'], '#888') for _, r in pair_counts.iterrows()]
        bars = ax.barh(pair_counts['pair'], pair_counts['n'], color=colors, alpha=0.85)
        ax.bar_label(bars, padding=3)
        ax.set_xlabel('Citation contexts containing term')
        ax.set_title('Cross-country citations near term\n(citing country → cited country)')
        ax.invert_yaxis()

    # ── Panel 2 (optional): corpus-wide hits by country + doc type ─────────
    if not corpus_hits.empty:
        ax2 = axes[1]
        pivot = corpus_hits.groupby(['country', 'doc_type'])['match_count'].sum().unstack(fill_value=0)
        pivot.plot(kind='bar', ax=ax2, color=['#4D748B', '#B93223', '#5a9e5a'], alpha=0.85)
        ax2.set_title('Term frequency in full corpus\n(by country and document type)')
        ax2.set_xlabel('Country')
        ax2.set_ylabel('Total occurrences')
        ax2.tick_params(axis='x', rotation=0)
        ax2.legend(title='Doc type')

    plt.tight_layout()

    slug = re.sub(r'[^\w]', '_', query.lower())[:30]
    out_path = OUTPUT_DIR / f'phrase_search_{slug}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n  Chart saved: {out_path}')


# ── TRANSLATION (optional) ────────────────────────────────────────────────────

def translate_query(query: str) -> list[str]:
    """
    Use the Anthropic API to translate query into German and Danish.
    Returns a list of translated terms (may be empty on failure).
    """
    try:
        import anthropic
    except ImportError:
        print('  anthropic package not installed (pip install anthropic).')
        return []

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('  ANTHROPIC_API_KEY not set — skipping translation.')
        return []

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f'Translate this animal welfare legislative term into German and Danish. '
        f'Return a JSON object with keys "de" and "dk" and exactly one translation '
        f'each (the most likely legal/parliamentary phrasing). '
        f'Term: "{query}"'
    )
    try:
        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=120,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = msg.content[0].text.strip()
        # Extract JSON from the response
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            d = json.loads(m.group())
            translations = [v for v in [d.get('de'), d.get('dk')] if v]
            de_t = d.get('de', '?')
            dk_t = d.get('dk', '?')
            print(f'  Translations: DE="{de_t}"  DK="{dk_t}"')
            return translations
    except Exception as e:
        print(f'  Translation failed: {e}')
    return []


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Search the pig-welfare corpus for a phrase and see cross-country influence.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--term', required=True,
        help='Search term or phrase, e.g. "schwanzkupieren" or "kastenstand".'
             ' Use the legislative language (German for DE/AT, Danish for DK).'
             ' For English terms, add --translate.',
    )
    parser.add_argument(
        '--similar', dest='similar', action='store_true', default=True,
        help='Also search morphological variants and component words (default: on).',
    )
    parser.add_argument(
        '--exact', dest='similar', action='store_false',
        help='Exact match only — disables morphological relaxation.',
    )
    parser.add_argument(
        '--corpus', action='store_true', default=False,
        help='Also search the full corpus RDS files (bills, laws, speeches).'
             ' Requires RDS files in data/Corpora_PLS_*/.',
    )
    parser.add_argument(
        '--context', action='store_true', default=False,
        help='Print the matched text passages to the terminal.',
    )
    parser.add_argument(
        '--translate', action='store_true', default=False,
        help='Auto-translate the term to German and Danish before searching'
             ' (requires ANTHROPIC_API_KEY).',
    )
    parser.add_argument(
        '--export-edges', action='store_true', default=False,
        help='Save filtered edge weights to data/phrase_search_edges.json'
             ' (same format as influence_edges.json; use with influence_map.ipynb).',
    )
    args = parser.parse_args()

    queries = [args.term]
    if args.translate:
        print('\nTranslating term...')
        translations = translate_query(args.term)
        queries.extend(translations)

    # ── Search citation contexts ───────────────────────────────────────────
    print(f'\nSearching citation contexts for: {queries}')
    all_ctx_matches = []
    all_variants    = []
    for q in queries:
        m, v = search_citation_contexts(q, similar=args.similar)
        all_ctx_matches.append(m)
        all_variants.extend(v)

    ctx_matches = (
        pd.concat(all_ctx_matches, ignore_index=True).drop_duplicates()
        if all_ctx_matches else pd.DataFrame()
    )
    variants_used = list(dict.fromkeys(all_variants))  # deduplicated, order-preserving

    # ── Summarise citation results ─────────────────────────────────────────
    if ctx_matches.empty:
        print('\n  No matches in citation contexts.')
    else:
        print(f'\n  Found {len(ctx_matches)} citation-context match(es).')
        if variants_used:
            print(f'  Variants that matched: {", ".join(variants_used)}')
        print()
        pair_summary = (
            ctx_matches
            .groupby(['citing_country', 'cited_country', 'doc_type'])
            .size()
            .rename('contexts')
            .reset_index()
        )
        print(pair_summary.to_string(index=False))

    if args.context and not ctx_matches.empty:
        print('\n── Sample passages ──────────────────────────────────────────────')
        for _, row in ctx_matches.head(8).iterrows():
            print(f'\n  [{row["citing_country"]} → {row["cited_country"]}]'
                  f'  {row.get("doc_type","")}  {row.get("date","")}')
            print(f'  {row.get("title","")[:80]}')
            print(f'  …{str(row.get("context",""))[:300]}…')

    # ── Corpus-wide search (optional) ─────────────────────────────────────
    corpus_hits = pd.DataFrame()
    if args.corpus:
        print('\nSearching full corpus (this may take a few minutes)...')
        corpus_pieces = []
        for q in queries:
            corpus_pieces.append(search_full_corpus(q, similar=args.similar))
        corpus_hits = (
            pd.concat(corpus_pieces, ignore_index=True).drop_duplicates()
            if corpus_pieces else pd.DataFrame()
        )
        if corpus_hits.empty:
            print('  No corpus matches found.')
        else:
            print(f'\n  Corpus hits: {len(corpus_hits)} documents across all countries.')
            summary = (
                corpus_hits
                .groupby(['country', 'doc_type'])
                .agg(documents=('match_count', 'count'),
                     total_mentions=('match_count', 'sum'))
            )
            print(summary.to_string())

    # ── Export filtered edges ──────────────────────────────────────────────
    if args.export_edges:
        print('\nExporting filtered edge weights...')
        export_edges(ctx_matches, label=args.term)

    # ── Chart ─────────────────────────────────────────────────────────────
    print('\nGenerating chart...')
    plot_results(ctx_matches, corpus_hits, args.term, variants_used)

    print('\nDone.')


if __name__ == '__main__':
    main()
