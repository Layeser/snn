#!/usr/bin/env python3
"""Generate use_figma placement scripts for LaTeX formula SVGs.

Run after: bash assets/figma_formulas/generate_formulas.sh

Page layout (4 pages):
  Fig 0 — Overview      backbone pipeline + notation
  Fig 1 — SPS
  Fig 2 — Attention
  Fig 3 — MLP

Requires Figma **Full** seat for MCP write (use_figma / upload_assets).
Dev seat on Pro: read + rate limits only, no MCP edits.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMULA_DIR = ROOT / "assets" / "figma_formulas"
OUT_DIR = FORMULA_DIR / "placement_scripts"
FILE_KEY = "OmmGSZinWakbBSAn8nvKiS"

PAGE_OVERVIEW = "Fig 0 — Overview"
PAGE_SPS = "Fig 1 — SPS"
PAGE_ATTN = "Fig 2 — Attention"
PAGE_MLP = "Fig 3 — MLP"


def js_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def make_import_script(
    name: str,
    svg_path: Path,
    page_name: str,
    parent_query: str | None,
    remove_text_ids: list[str],
    remove_text_contains: str | None = None,
) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    svg_js = js_escape(svg)
    remove_ids_js = json.dumps(remove_text_ids)
    remove_contains_js = json.dumps(remove_text_contains)

    parent_block = ""
    if parent_query:
        parent_block = f"""
const parent = figma.currentPage.query('{parent_query}').first();
if (!parent) return {{ error: 'parent not found: {parent_query}' }};
"""

    return f"""// Auto-generated: import {name}.svg onto {page_name}
const createdNodeIds = [];
const mutatedNodeIds = [];
const removedNodeIds = [];

const page = figma.root.children.find(p => p.name === {json.dumps(page_name)});
if (!page) return {{ error: 'page not found: {page_name}' }};
await figma.setCurrentPageAsync(page);

for (const id of {remove_ids_js}) {{
  const old = await figma.getNodeByIdAsync(id);
  if (old) {{ old.remove(); removedNodeIds.push(id); }}
}}

const needle = {remove_contains_js};
if (needle) {{
  for (const t of figma.currentPage.query('TEXT')) {{
    if (t.characters.includes(needle)) {{ t.remove(); removedNodeIds.push(t.id); }}
  }}
}}

{parent_block}
const node = await figma.createNodeFromSvgAsync(`{svg_js}`);
node.name = 'LaTeX: {name}';
createdNodeIds.push(node.id);

if (typeof parent !== 'undefined' && parent) {{
  parent.appendChild(node);
  node.layoutSizingHorizontal = 'FILL';
  node.layoutSizingVertical = 'HUG';
}} else {{
  node.x = 80;
  node.y = 80;
  figma.currentPage.appendChild(node);
}}

return {{ createdNodeIds, mutatedNodeIds, removedNodeIds, name: {json.dumps(name)} }};
"""


PLACEMENTS = [
    {
        "name": "factorized_overview",
        "page": PAGE_OVERVIEW,
        "parent_query": 'FRAME[name="Transformer Block (x L)"]',
        "remove": ["5:49"],
    },
    {
        "name": "sdt_overview",
        "page": PAGE_OVERVIEW,
        "parent_query": 'FRAME[name="Transformer Block (x L)"]',
        "remove": ["5:50"],
    },
    {
        "name": "notation",
        "page": PAGE_OVERVIEW,
        "parent_query": 'FRAME[name="Notation"]',
        "remove": ["9:135", "9:136", "9:137"],
    },
    {
        "name": "fact_block",
        "page": PAGE_ATTN,
        "parent_query": 'FRAME[name="(b) Factorized mode — matrix multiply (STAtten)"]',
        "remove": ["9:44", "9:45", "9:46"],
    },
    {
        "name": "sdt_block",
        "page": PAGE_ATTN,
        "parent_query": 'FRAME[name="(c) SDT mode — element-wise Hadamard (SDSA)"]',
        "remove": ["9:54", "9:55", "9:56", "9:57", "9:58"],
    },
    {
        "name": "q_shape",
        "page": PAGE_ATTN,
        "parent_query": None,
        "remove": [],
        "remove_text_contains": "Q \\in",
    },
    {
        "name": "k_shape",
        "page": PAGE_ATTN,
        "parent_query": None,
        "remove": [],
        "remove_text_contains": "K \\in",
    },
    {
        "name": "v_shape",
        "page": PAGE_ATTN,
        "parent_query": None,
        "remove": [],
        "remove_text_contains": "V \\in",
    },
]

REORGANIZE_PAGES = f"""
// 4-page layout: Overview, SPS, Attention, MLP
const createdNodeIds = [];
const mutatedNodeIds = [];

const legacyNames = ['Overview + Fig 3 MLP', 'Page 1', '{PAGE_OVERVIEW}'];
let overview = null;
for (const n of legacyNames) {{
  overview = figma.root.children.find(p => p.name === n);
  if (overview) break;
}}
if (!overview) return {{ error: 'overview page not found' }};
overview.name = '{PAGE_OVERVIEW}';
mutatedNodeIds.push(overview.id);

let mlpPage = figma.root.children.find(p => p.name === '{PAGE_MLP}');
if (!mlpPage) {{
  mlpPage = figma.createPage();
  mlpPage.name = '{PAGE_MLP}';
  createdNodeIds.push(mlpPage.id);
}}

await figma.setCurrentPageAsync(overview);
const mlpFrame = overview.children.find(n => n.name === 'MLP Detail');
if (mlpFrame) {{
  mlpPage.appendChild(mlpFrame);
  mlpFrame.x = 60;
  mlpFrame.y = 60;
  mutatedNodeIds.push(mlpFrame.id);
}}

for (const ch of overview.children) {{
  if (ch.y < 0 && ch.visible !== false) {{
    ch.visible = false;
    mutatedNodeIds.push(ch.id);
  }}
}}

const sps = figma.root.children.find(p => p.name === '{PAGE_SPS}');
const attn = figma.root.children.find(p => p.name === '{PAGE_ATTN}');
const order = [overview, sps, attn, mlpPage].filter(Boolean);
for (let i = 0; i < order.length; i++) {{
  figma.root.insertChild(i, order[i]);
}}

return {{
  createdNodeIds,
  mutatedNodeIds,
  pages: figma.root.children.map(p => p.name),
  mlpMoved: !!mlpFrame,
}};
"""

FIX_LAYOUT = f"""
const mutatedNodeIds = [];
const page = figma.root.children.find(p => p.name === '{PAGE_OVERVIEW}');
if (!page) return {{ error: 'overview not found' }};
await figma.setCurrentPageAsync(page);

for (const ch of page.children) {{
  if (ch.y < 0 && ch.visible !== false) {{
    ch.visible = false;
    mutatedNodeIds.push(ch.id);
  }}
}}

return {{ mutatedNodeIds, page: page.name }};
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for p in PLACEMENTS:
        svg = FORMULA_DIR / f"{p['name']}.svg"
        if not svg.exists():
            raise SystemExit(f"Missing {svg}. Run generate_formulas.sh first.")
        script = make_import_script(
            p["name"],
            svg,
            p["page"],
            p.get("parent_query"),
            p["remove"],
            p.get("remove_text_contains"),
        )
        out = OUT_DIR / f"import_{p['name']}.js"
        out.write_text(script, encoding="utf-8")
        manifest.append({"name": p["name"], "script": str(out.relative_to(ROOT)), "page": p["page"]})

    (OUT_DIR / "reorganize_pages.js").write_text(REORGANIZE_PAGES.strip(), encoding="utf-8")
    (OUT_DIR / "fix_layout.js").write_text(FIX_LAYOUT.strip(), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(manifest)} import scripts + reorganize_pages.js + fix_layout.js")
    print(f"File key: {FILE_KEY}")
    print("Run reorganize_pages.js first, then fix_layout.js, then import_*.js via use_figma (Full seat).")


if __name__ == "__main__":
    main()
