#!/usr/bin/env python3
"""
Convert every .docx in raw_transcripts/docx/ into a plain .txt file,
saved directly in raw_transcripts/ (one level up), same base filename.

Usage:
    python docx_to_txt.py
"""
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

SRC_DIR = Path("raw_transcripts/docx")
DEST_DIR = Path("raw_transcripts")


def docx_to_text(path: Path) -> str:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8", "ignore")
    lines = []
    for p in ET.fromstring(xml).iter(NS + "p"):
        para_text = []
        for r in p.iter(NS + "r"):
            # check if this run has <w:caps/> or <w:caps w:val="true"/> applied
            rpr = r.find(NS + "rPr")
            caps = rpr is not None and rpr.find(NS + "caps") is not None
            for t in r.iter(NS + "t"):
                if t.text:
                    para_text.append(t.text.upper() if caps else t.text)
        lines.append("".join(para_text))
    return "\n".join(lines)


def main():
    docx_files = sorted(SRC_DIR.glob("*.docx"))
    if not docx_files:
        print(f"No .docx files found in {SRC_DIR.resolve()}")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for path in docx_files:
        out_path = DEST_DIR / (path.stem + ".txt")
        text = docx_to_text(path)
        out_path.write_text(text, encoding="utf-8")
        print(f"{path.name} -> {out_path}")


if __name__ == "__main__":
    main()
