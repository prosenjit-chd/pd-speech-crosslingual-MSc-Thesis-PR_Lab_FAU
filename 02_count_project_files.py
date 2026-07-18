"""
02_count_project_files.py

Place this file in the root of your project and run:

    python 02_count_project_files.py

It generates:
    project_counter_report.txt
    project_extension_counts.csv

The report includes:
    - total folders
    - total files
    - total Python files
    - total CSV files
    - total audio files
    - serial-wise count by file extension
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parent
OUTPUT_TXT = ROOT / "project_counter_report.txt"
OUTPUT_CSV = ROOT / "project_extension_counts.csv"
EXCLUDE_NAMES: set[str] = set()
GENERATED_OUTPUTS = {OUTPUT_TXT.name, OUTPUT_CSV.name}

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg",
    ".wma", ".aiff", ".aif", ".opus",
}


def should_skip(path: Path) -> bool:
    return path.name in EXCLUDE_NAMES or path.name in GENERATED_OUTPUTS


def scan_project() -> tuple[list[Path], list[Path]]:
    folders: list[Path] = []
    files: list[Path] = []

    for item in ROOT.rglob("*"):
        if should_skip(item):
            continue

        try:
            relative_parts = item.relative_to(ROOT).parts
        except ValueError:
            continue

        if any(part in EXCLUDE_NAMES for part in relative_parts):
            continue

        try:
            if item.is_dir():
                folders.append(item)
            elif item.is_file():
                files.append(item)
        except (PermissionError, OSError):
            continue

    return folders, files


def normalized_extension(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    return suffix if suffix else "[no extension]"


def format_number(value: int) -> str:
    return f"{value:,}"


def main() -> None:
    print(f"Scanning project root:\n{ROOT}\n")

    folders, files = scan_project()
    python_files = [f for f in files if f.suffix.lower() == ".py"]
    csv_files = [f for f in files if f.suffix.lower() == ".csv"]
    audio_files = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]
    extension_counter = Counter(normalized_extension(f) for f in files)

    sorted_extensions = sorted(
        extension_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )

    summary_rows = [
        ("Total folders", len(folders)),
        ("Total files", len(files)),
        ("Total Python files (.py)", len(python_files)),
        ("Total CSV files (.csv)", len(csv_files)),
        ("Total audio files", len(audio_files)),
        ("Total unique file extensions", len(extension_counter)),
        ("Total folders + files", len(folders) + len(files)),
    ]

    report_lines: list[str] = [
        "PROJECT FILE AND FOLDER COUNTER",
        f"Root: {ROOT}",
        "=" * 100,
        "",
        "MAIN SUMMARY",
        "-" * 100,
    ]

    for serial, (label, count) in enumerate(summary_rows, start=1):
        report_lines.append(f"{serial:02d}. {label:<40} {format_number(count):>12}")

    report_lines.extend([
        "",
        "SERIAL-WISE FILE COUNT BY EXTENSION",
        "-" * 100,
        f"{'No.':<6}{'Extension':<25}{'Count':>12}",
        "-" * 100,
    ])

    for serial, (extension, count) in enumerate(sorted_extensions, start=1):
        report_lines.append(f"{serial:<6}{extension:<25}{format_number(count):>12}")

    report_lines.extend([
        "",
        "AUDIO EXTENSIONS INCLUDED",
        "-" * 100,
        ", ".join(sorted(AUDIO_EXTENSIONS)),
        "",
        "PYTHON FILE LOCATIONS",
        "-" * 100,
    ])

    for serial, path in enumerate(sorted(python_files), start=1):
        report_lines.append(f"{serial:04d}. {path.relative_to(ROOT)}")

    report_lines.extend(["", "CSV FILE LOCATIONS", "-" * 100])
    for serial, path in enumerate(sorted(csv_files), start=1):
        report_lines.append(f"{serial:04d}. {path.relative_to(ROOT)}")

    report_lines.extend(["", "AUDIO FILE LOCATIONS", "-" * 100])
    for serial, path in enumerate(sorted(audio_files), start=1):
        report_lines.append(f"{serial:04d}. {path.relative_to(ROOT)}")

    OUTPUT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["serial", "extension", "count"])
        for serial, (extension, count) in enumerate(sorted_extensions, start=1):
            writer.writerow([serial, extension, count])

    print("MAIN SUMMARY")
    print("-" * 60)
    for label, count in summary_rows:
        print(f"{label:<40} {format_number(count):>12}")

    print(f"\nText report created:\n{OUTPUT_TXT}")
    print(f"\nExtension counter CSV created:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        sys.exit(1)
