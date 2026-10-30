"""Convert the first page of a PDF to a standalone SVG using MuPDF.

Text is converted to outlines, so the SVG carries no font dependency and renders
identically wherever it is opened: a browser, a README, a LaTeX include.

Usage: python pdf_to_svg.py <input.pdf> <output.svg>
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import pymupdf

_UNUSED_NS = ' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"'
_DRAWABLE = {"path", "use", "image", "rect", "text", "polygon", "polyline", "circle", "ellipse"}


def convert(pdf_path: str, svg_path: str) -> tuple[float, float]:
    with pymupdf.open(pdf_path) as doc:
        if doc.page_count < 1:
            raise SystemExit(f"{pdf_path}: no pages")
        page = doc[0]
        svg = page.get_svg_image(text_as_path=True)
        width, height = page.rect.width, page.rect.height

    # MuPDF declares an Inkscape namespace it never uses; dropping it keeps the header honest
    # for anyone who opens the SVG.
    if "inkscape:" not in svg:
        svg = svg.replace(_UNUSED_NS, "", 1)
    if not svg.startswith("<?xml"):
        svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg

    with open(svg_path, "w", encoding="utf-8") as handle:
        handle.write(svg)

    # A blank page converts to well-formed XML, so a broken build would otherwise pass silently
    # and only show up as an empty box in the README.
    root = ET.fromstring(svg)
    if not root.get("width") or not root.get("height"):
        raise SystemExit(f"{svg_path}: no width/height on the root element")
    drawable = sum(1 for el in root.iter() if el.tag.split("}")[-1] in _DRAWABLE)
    if drawable == 0:
        raise SystemExit(f"{svg_path}: parsed as XML but contains no drawable elements")

    print(f"svg   : {svg_path}")
    print(f"size  : {width:.3f} x {height:.3f} pt")
    print(f"shapes: {drawable} drawable elements")
    return width, height


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    convert(sys.argv[1], sys.argv[2])
