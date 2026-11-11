"""Turn MinerU's middle JSON into reading-order blocks, attach figure descriptions, and render paper text.

Copied from SPIQA_Deepseek (https://github.com/taylorsm9/SPIQA_Deepseek) intake/mineru.py (parse_middle, attach_descriptions) and intake/prompts.py
(label_of, the figure/table text format of ocr_parts), minus image crops.

MinerU splits multi-panel figures into one chart/image block per panel, with the "Figure N:" caption
attached to the last panel. We group consecutive visual blocks on a page up to that caption and crop the
union of their bboxes, so each figure is one image (as in SPIQA). Tables keep MinerU's HTML.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


FIG_CAP = re.compile(r"^\s*(fig\.?|figure)\s*\d+", re.I)
TAB_CAP = re.compile(r"^\s*table\s*[\dIVX]+", re.I)
VISUAL = {"chart", "image"}
DROP = {"header", "footer", "page_number", "aside_text"}  # same as MinerU's own markdown.md
CROP_DPI = 200
PAD = 0.004  # normalized padding around crops


@dataclass
class Block:
    kind: str                  # text | figure | table
    text: str = ""
    caption: str = ""
    html: str = ""
    crop: str = ""             # path relative to ROOT
    page: int = -1
    bbox: list | None = None   # normalized union bbox
    desc: str = ""             # MinerU image analysis of the figure's panels (advanced tier only; arm E)


def spans(content) -> str:
    """Flatten MinerU span lists (text, inline equations, nested bodies) to markdown text."""
    if isinstance(content, str):
        return content
    out = []
    for c in content or []:
        t = c.get("type", "")
        if t == "equation_inline":
            out.append(f"${c['content']}$")
        elif isinstance(c.get("content"), list):
            out.append(spans(c["content"]))
        else:
            out.append(str(c.get("content", "")))
    return "".join(out)


def _children(b, suffix):
    return [c for c in b.get("content", []) if isinstance(c, dict) and c.get("type", "").endswith(suffix)]


def _analysis(b, parts) -> str:
    """Image-analysis text MinerU wrote into a chart/image body (empty unless run with image analysis)."""
    body = " ".join(c["content"] for c in parts
                    if c["type"].endswith("_body") and isinstance(c.get("content"), str) and c["content"].strip())
    return f"[{b.get('sub_type') or b.get('type')}]\n{body.strip()}" if body.strip() else ""


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def parse_middle(middle: dict) -> list[Block]:
    blocks: list[Block] = []
    group: list[tuple[list, str, str]] = []   # (bbox, subcaption, image analysis) of open figure group
    group_page = -1

    def close_group(caption=""):
        nonlocal group
        if group:
            subs = [s for _, s, _ in group if s]
            descs = [d for _, _, d in group if d]
            cap = caption or " ".join(subs)
            if caption and subs:
                cap = caption + " " + " ".join(subs)
            blocks.append(Block("figure", caption=cap.strip(), page=group_page,
                                bbox=_union([bb for bb, _, _ in group]), desc="\n\n".join(descs)))
        group = []

    for page in middle["pages"]:
        for b in page["blocks"]:
            t = b.get("type")
            if t in VISUAL:
                if group and group_page != page["page_idx"]:
                    close_group()
                group_page = page["page_idx"]
                # A "Figure N:" caption listed before this block's body belongs to the previous group
                # (MinerU attaches it to the nearest panel, which can be the next figure's first panel)
                parts = [c for c in b.get("content", []) if isinstance(c, dict)]
                body_at = next((i for i, c in enumerate(parts) if c["type"].endswith("_body")), 0)
                before, after = [], []
                for i, c in enumerate(parts):
                    if c["type"].endswith("_caption"):
                        (before if i < body_at else after).append(spans(c["content"]))
                lead = [c for c in before if FIG_CAP.match(c)]
                if lead and group:
                    close_group(" ".join(lead))
                    before = [c for c in before if not FIG_CAP.match(c)]
                caps = before + after
                main = [c for c in caps if FIG_CAP.match(c)]
                sub = " ".join(c for c in caps if not FIG_CAP.match(c))
                group.append((b["bbox"], sub, _analysis(b, parts)))
                if main:
                    close_group(" ".join(main))
                continue
            # A free-standing "Figure N:" text block right after an open group is its caption
            if group and t == "text" and FIG_CAP.match(spans(b.get("content"))):
                close_group(spans(b["content"]))
                continue
            close_group()
            if t in DROP:
                continue
            if t == "table":
                cap = " ".join(spans(c["content"]) for c in _children(b, "table_caption"))
                foot = " ".join(spans(c["content"]) for c in _children(b, "table_footnote"))
                body = _children(b, "table_body")
                html = body[0]["content"] if body else ""
                if not cap and blocks and blocks[-1].kind == "text" and TAB_CAP.match(blocks[-1].text):
                    cap = blocks.pop().text
                # Sub-tables (a), (b), ... inherit the parent "Table N" caption of the previous sub-table
                if not TAB_CAP.match(cap) and blocks and blocks[-1].kind == "table":
                    parent = TAB_CAP.match(blocks[-1].caption) and blocks[-1].caption.split(" (", 1)[0]
                    if parent:
                        cap = f"{parent} {cap}".strip()
                blocks.append(Block("table", caption=cap, html=(html + ("\n" + foot if foot else "")).strip(),
                                    page=page["page_idx"], bbox=b["bbox"]))
            elif t == "equation":
                blocks.append(Block("text", text=f"$${spans(b.get('content'))}$$"))
            elif t in ("paragraph_title", "doc_title"):
                blocks.append(Block("text", text="#" * int(b.get("level") or 1) + " " + spans(b.get("content"))))
            else:
                blocks.append(Block("text", text=spans(b.get("content"))))
    close_group()
    return [b for b in blocks if b.kind != "text" or b.text.strip()]


def _overlap(a, b) -> float:
    """Intersection over the smaller box's area."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 0.0


def attach_descriptions(blocks: list[Block], adv: list[Block], label_of) -> int:
    """Copy the advanced run's figure analyses onto the standard run's figure blocks, in place.

    A figure matches advanced-run visuals on the same page whose boxes overlap it by at least half; image
    analysis can turn a figure that is really a table into a table block, so those count too (their HTML is
    the description). Unmatched figures fall back to the same "Figure N" label. Returns figures matched.
    """
    cands = [a for a in adv if a.kind == "figure" or (a.kind == "table" and not TAB_CAP.match(a.caption))]
    found = 0
    for b in blocks:
        if b.kind != "figure":
            continue
        hits = [a for a in cands if a.page == b.page and b.bbox and a.bbox and _overlap(a.bbox, b.bbox) >= 0.5]
        if not hits and label_of(b.caption):
            hits = [a for a in cands if a.kind == "figure" and label_of(a.caption) == label_of(b.caption)]
        parts = [a.desc if a.kind == "figure" else f"[table]\n{a.html}" for a in hits]
        b.desc = "\n\n".join(p for p in parts if p)
        found += bool(b.desc)
    return found


LABEL = re.compile(r"(fig(?:ure)?|table)\.?\s*(\d+)", re.I)


def label_of(caption: str) -> str | None:
    """'Figure 3: ...' -> 'figure3'; 'Table 2 (b)' -> 'table2'."""
    m = LABEL.match(caption.strip()) if caption else None
    return f"{'table' if m.group(1).lower() == 'table' else 'figure'}{m.group(2)}" if m else None


def load_blocks(zip_path: Path) -> list[Block]:
    import zipfile
    return parse_middle(json.loads(zipfile.ZipFile(zip_path).read("middle_json.json")))


def render_text(blocks: list[Block]) -> str:
    """Blocks in reading order; tables as caption + HTML, figures as caption + description (ocr_parts, no crops)."""
    buf = []
    for b in blocks:
        if b.kind == "text":
            buf.append(b.text)
        elif b.kind == "table":
            buf.append(f"[Table] {b.caption or '(no caption)'}\n{b.html or '(table content not recognized)'}")
        else:
            desc = f"\nFigure content, read from the image by the OCR model:\n{b.desc}" if b.desc else ""
            buf.append(f"[Figure: image not shown] {b.caption or '(no caption)'}{desc}")
    return "\n\n".join(buf)


def paper_text(std_zip: Path, adv_zip: Path) -> tuple[str, dict]:
    """Standard-tier text with the advanced tier's figure descriptions under each figure's caption."""
    blocks = load_blocks(std_zip)
    described = attach_descriptions(blocks, load_blocks(adv_zip), label_of)
    figures = sum(b.kind == "figure" for b in blocks)
    return render_text(blocks), {"figures": figures, "figures_described": described}
