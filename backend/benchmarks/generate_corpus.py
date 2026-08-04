import random
from pathlib import Path

from faker import Faker
from fpdf import FPDF

CORPUS_DIR = Path(__file__).parent / "corpus"
fake = Faker()


def generate_txt(path: Path, num_paragraphs: int):
    with path.open("w", encoding="utf-8") as f:
        for _ in range(num_paragraphs):
            f.write(fake.paragraph(nb_sentences=10))
            f.write("\n\n")


def generate_md(path: Path, num_sections: int):
    with path.open("w", encoding="utf-8") as f:
        f.write("# Synthetic Markdown Document\n\n")
        for i in range(num_sections):
            f.write(f"## Section {i + 1}: {fake.sentence()}\n\n")
            f.write(fake.paragraph(nb_sentences=5) + "\n\n")
            if random.random() > 0.5:
                f.write(f"- {fake.sentence()}\n")
                f.write(f"- {fake.sentence()}\n")
                f.write(f"- {fake.sentence()}\n\n")
            if random.random() > 0.8:
                f.write(f"```python\ndef test_{i}():\n    pass\n```\n\n")


def generate_pdf(path: Path, num_pages: int):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for i in range(num_pages):
        if i > 0:
            pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 10, f"Page {i + 1} Title", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=12)
        pdf.ln(5)
        for _ in range(5):
            # Encode correctly to avoid fpdf latin-1 issues with some faker chars
            text = (
                fake.paragraph(nb_sentences=10)
                .encode("latin-1", "replace")
                .decode("latin-1")
            )
            pdf.multi_cell(0, 10, text)
            pdf.ln(5)
    pdf.output(str(path))


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating small.txt (10 paragraphs)...")
    generate_txt(CORPUS_DIR / "small.txt", 10)

    print("Generating medium.md (20 sections)...")
    generate_md(CORPUS_DIR / "medium.md", 20)

    print("Generating large.txt (200 paragraphs)...")
    generate_txt(CORPUS_DIR / "large.txt", 200)

    print("Generating small.pdf (2 pages)...")
    generate_pdf(CORPUS_DIR / "small.pdf", 2)

    print("Generating large.pdf (20 pages)...")
    generate_pdf(CORPUS_DIR / "large.pdf", 20)

    print("Synthetic corpus generation complete!")


if __name__ == "__main__":
    main()
