"""Small helpers to build the self-contained HTML report (figures embedded as base64 PNG)."""
from __future__ import annotations

import base64
from html import escape as esc
from pathlib import Path


def img(path, alt, caption=None):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="plate"><img src="data:image/png;base64,{b64}" alt="{esc(alt)}">{cap}</figure>'


def table(header, rows, num_cols=(), highlight_rows=(), caption=None):
    cls = lambda i: ' class="num"' if i in num_cols else ""
    th = "".join(f"<th{cls(i)}>{h}</th>" for i, h in enumerate(header))
    body = "".join(f'<tr{" class=hl" if r in highlight_rows else ""}>' + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(row))
                   + "</tr>" for r, row in enumerate(rows))
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tablewrap"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def signed(x, nd=2, unit="%"):
    """+1.23% with a typographic minus."""
    return f"{x:+.{nd}f}{unit}".replace("-", "&minus;")


def num(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "&minus;")


def tree_html(node, action_names, features, formats, unit="training days"):
    """Nested rendering of a policy tree (dict from PolicyTree.root); formats[j] formats feature j's threshold."""
    def rec(n):
        if n["leaf"]:
            return (f'<div class="leaf"><span class="act">{esc(action_names[n["action"]])}</span>'
                    f'<span class="n">{n["n"]} {unit}</span></div>')
        t = formats[n["feature"]].format(n["threshold"])
        return (f'<div class="split"><div class="q">{esc(features[n["feature"]])}</div><div class="branches">'
                f'<div class="br"><div class="cond">&le; {t}</div>{rec(n["left"])}</div>'
                f'<div class="br"><div class="cond">&gt; {t}</div>{rec(n["right"])}</div></div></div>')
    return f'<div class="tree">{rec(node)}</div>'
