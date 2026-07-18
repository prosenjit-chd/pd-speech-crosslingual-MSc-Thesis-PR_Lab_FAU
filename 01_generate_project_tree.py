"""
01_generate_project_tree.py

Place this file in the root of your project and run:

    python 01_generate_project_tree.py

It generates:
    project_tree.txt
    project_tree.pdf   (only if ReportLab is installed)

To enable PDF output:
    pip install reportlab
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

ROOT = Path(__file__).resolve().parent
EXCLUDE_NAMES: set[str] = set()
OUTPUT_TXT = ROOT / "project_tree.txt"
OUTPUT_PDF = ROOT / "project_tree.pdf"
GENERATED_OUTPUTS = {OUTPUT_TXT.name, OUTPUT_PDF.name}


def natural_sort_key(path: Path) -> tuple[int, str]:
    return (0 if path.is_dir() else 1, path.name.lower())


def should_skip(path: Path) -> bool:
    return path.name in EXCLUDE_NAMES or path.name in GENERATED_OUTPUTS


def safe_children(folder: Path) -> list[Path]:
    try:
        return sorted(
            [item for item in folder.iterdir() if not should_skip(item)],
            key=natural_sort_key,
        )
    except (PermissionError, OSError):
        return []


def build_tree(folder: Path, prefix: str = "") -> list[str]:
    lines: list[str] = []
    children = safe_children(folder)

    for index, child in enumerate(children):
        is_last = index == len(children) - 1
        connector = "└── " if is_last else "├── "
        display_name = child.name + ("/" if child.is_dir() else "")
        lines.append(f"{prefix}{connector}{display_name}")

        if child.is_dir():
            extension = "    " if is_last else "│   "
            nested = build_tree(child, prefix + extension)
            if nested:
                lines.extend(nested)
            else:
                try:
                    has_items = any(child.iterdir())
                except (PermissionError, OSError):
                    lines.append(prefix + extension + "[Unable to read folder]")

    return lines


def write_text_report(lines: Iterable[str]) -> None:
    header = [
        "PROJECT TREE STRUCTURE",
        f"Root: {ROOT}",
        "=" * 100,
        ROOT.name + "/",
    ]
    content = "\n".join(header + list(lines)) + "\n"
    OUTPUT_TXT.write_text(content, encoding="utf-8")


def write_pdf_report(lines: list[str]) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
    except ImportError:
        return False

    candidate_fonts = [
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
    ]

    font_name = "Courier"
    for font_path in candidate_fonts:
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont("TreeMono", str(font_path)))
                font_name = "TreeMono"
                break
            except Exception:
                pass

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28,
        title="Project Tree Structure",
        author="Project Inventory Script",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.fontName = font_name
    title_style.fontSize = 16

    mono_style = ParagraphStyle(
        "TreeStyle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=7.5,
        leading=9,
        leftIndent=0,
        rightIndent=0,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=0,
    )

    story = [
        Paragraph("PROJECT TREE STRUCTURE", title_style),
        Spacer(1, 8),
        Paragraph(f"Root: {ROOT}", mono_style),
        Spacer(1, 8),
        Paragraph(ROOT.name + "/", mono_style),
    ]

    for raw_line in lines:
        safe_line = (
            raw_line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", "&nbsp;")
        )
        story.append(Paragraph(safe_line, mono_style))

    doc.build(story)
    return True


def main() -> None:
    print(f"Scanning project root:\n{ROOT}\n")
    tree_lines = build_tree(ROOT)
    write_text_report(tree_lines)
    pdf_created = write_pdf_report(tree_lines)

    print(f"Text tree created:\n{OUTPUT_TXT}")
    if pdf_created:
        print(f"PDF tree created:\n{OUTPUT_PDF}")
    else:
        print(
            "\nPDF was not created because ReportLab is not installed.\n"
            "Install it with:\n"
            "    pip install reportlab\n"
            "Then run this script again."
        )

    print(f"\nTotal displayed entries: {len(tree_lines)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        sys.exit(1)
