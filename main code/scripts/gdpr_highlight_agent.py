import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, DictionaryObject, NumberObject, ArrayObject, FloatObject

from preprocess_pdfs import normalize_text, take_first_n_per_folder


# Simple regex candidates (fallback or pre-screener)
REGEX_PATTERNS: Dict[str, re.Pattern[str]] = {
	"EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
	"PHONE": re.compile(r"(?:(?:\+\d{1,3}[\s-])?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]?\d{3,4}(?:[\s-]?\d{3,4})?)"),
	"SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
	"PASSPORT": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
	"IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
	"CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
	"NAME": re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b"),
	"AMOUNT": re.compile(r"\b(?:USD|INR|EUR|GBP|\$|₹|€|£)?\s?\d{1,3}(?:[, ]\d{3})*(?:\.\d{2})?\b"),
}
MASK_CATEGORIES = {"NAME", "AMOUNT"}


OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
RULES_PATH = Path(__file__).parent / "gdpr_rules.json"


def load_rules() -> Dict:
	if RULES_PATH.exists():
		return json.loads(RULES_PATH.read_text(encoding="utf-8"))
	return {"version": "none", "categories": []}


def call_openai_detect(text: str) -> List[Dict[str, str]]:
	"""Use OpenAI to identify sensitive substrings.
	Returns a list of {"category": str, "substring": str} dicts.
	"""
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		return []

	from openai import OpenAI  # lazy import
	client = OpenAI(api_key=api_key)

	rules = load_rules()
	prompt = (
		"You are a GDPR detection engine. Using the provided JSON rules, "
		"extract a list of detections from the TEXT as JSON. Each item must be an object "
		"with keys: category (rule id), substring (verbatim text). Only include short exact substrings. "
		"Max 100 items. Output JSON array only."
	)

	text_sample = text[:8000]  # cap prompt size
	resp = client.chat.completions.create(
		model=OPENAI_MODEL,
		messages=[
			{"role": "system", "content": "You return only JSON arrays with detections."},
			{"role": "user", "content": f"RULES:\n{json.dumps(rules)}\n\n{prompt}\n\nTEXT:\n{text_sample}"},
		],
		temperature=0.0,
	)
	content = resp.choices[0].message.content.strip()
	try:
		items = json.loads(content)
		out: List[Dict[str, str]] = []
		for it in items:
			if isinstance(it, dict) and isinstance(it.get("substring"), str):
				cat = str(it.get("category")) if it.get("category") is not None else "unknown"
				sub = it.get("substring").strip()
				if sub:
					out.append({"category": cat, "substring": sub})
		return out
	except Exception:
		# If not valid JSON, try to extract JSON array substring
		m = re.search(r"\[(?:.|\n)*\]", content)
		if m:
			try:
				items2 = json.loads(m.group(0))
				out2: List[Dict[str, str]] = []
				for it in items2:
					if isinstance(it, dict) and isinstance(it.get("substring"), str):
						cat = str(it.get("category")) if it.get("category") is not None else "unknown"
						sub = it.get("substring").strip()
						if sub:
							out2.append({"category": cat, "substring": sub})
				return out2
			except Exception:
				return []
		return []


def regex_candidates(text: str) -> List[Dict[str, str]]:
	out: List[Dict[str, str]] = []
	for name, pattern in REGEX_PATTERNS.items():
		for m in pattern.finditer(text):
			frag = m.group(0)
			if frag and not any(d.get("substring") == frag for d in out):
				out.append({"category": name, "substring": frag})
	return out[:100]


def _add_highlight(writer: PdfWriter, page_index: int, rect: Tuple[float, float, float, float]):
    page = writer.pages[page_index]
    llx, lly, urx, ury = rect
    highlight = DictionaryObject()
    highlight.update({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Highlight"),
        NameObject("/Rect"): ArrayObject([FloatObject(llx), FloatObject(lly), FloatObject(urx), FloatObject(ury)]),
        NameObject("/C"): ArrayObject([FloatObject(1), FloatObject(1), FloatObject(0)]),  # yellow
        NameObject("/QuadPoints"): ArrayObject([
            FloatObject(llx), FloatObject(ury), FloatObject(urx), FloatObject(ury),
            FloatObject(llx), FloatObject(lly), FloatObject(urx), FloatObject(lly)
        ]),
        NameObject("/F"): NumberObject(4),
    })
    if "/Annots" in page:
        page[NameObject("/Annots")].append(highlight)
    else:
        page[NameObject("/Annots")] = ArrayObject([highlight])


def _add_black_box(writer: PdfWriter, page_index: int, rect: Tuple[float, float, float, float]):
    page = writer.pages[page_index]
    llx, lly, urx, ury = rect
    square = DictionaryObject()
    square.update({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Square"),
        NameObject("/Rect"): ArrayObject([FloatObject(llx), FloatObject(lly), FloatObject(urx), FloatObject(ury)]),
        NameObject("/IC"): ArrayObject([FloatObject(0), FloatObject(0), FloatObject(0)]),  # interior color black
        NameObject("/C"): ArrayObject([FloatObject(0), FloatObject(0), FloatObject(0)]),
        NameObject("/F"): NumberObject(4),
        NameObject("/CA"): FloatObject(1),
        NameObject("/Border"): ArrayObject([FloatObject(0), FloatObject(0), FloatObject(0)]),
    })
    if "/Annots" in page:
        page[NameObject("/Annots")].append(square)
    else:
        page[NameObject("/Annots")] = ArrayObject([square])


def highlight_terms_in_pdf(src_pdf: Path, dst_pdf: Path, detections: List[Dict[str, str]]) -> None:
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Always write a copy; maybe without highlights if none
    reader = PdfReader(str(src_pdf))
    writer = PdfWriter()
    for p in reader.pages:
        writer.add_page(p)

    cleaned_terms = []
    mask_set = set()
    for d in detections:
        sub = (d.get("substring") or "").strip()
        cat = str(d.get("category") or "").upper()
        if not sub:
            continue
        cleaned_terms.append(sub)
        if cat in MASK_CATEGORIES:
            mask_set.add(sub.lower())
    if not cleaned_terms:
        with open(dst_pdf, "wb") as f:
            writer.write(f)
        return

    # Use pdfplumber to locate word boxes, then add annotations via pypdf
    with pdfplumber.open(str(src_pdf)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            if not words:
                continue
            # Build a list of word texts for naive matching
            word_texts = [w.get("text", "") for w in words]
            lowered_words = [t.lower() for t in word_texts]

            for term in cleaned_terms:
                if len(term) < 3:
                    continue
                term_l = term.lower()
                # Try to match term across consecutive words by sliding window
                max_window = min(len(words), max(1, len(term.split()) + 6))
                found_any = False
                for window in range(1, max_window + 1):
                    for start in range(0, len(words) - window + 1):
                        snippet = " ".join(lowered_words[start:start + window])
                        if term_l in snippet:
                            # Build a union bbox covering from first to last word in window
                            x0 = min(words[start]["x0"], words[start + window - 1]["x0"]) if window > 1 else words[start]["x0"]
                            top = min(words[start]["top"], words[start + window - 1]["top"]) if window > 1 else words[start]["top"]
                            x1 = max(words[start]["x1"], words[start + window - 1]["x1"]) if window > 1 else words[start]["x1"]
                            bottom = max(words[start]["bottom"], words[start + window - 1]["bottom"]) if window > 1 else words[start]["bottom"]

                            # pdfplumber coordinate origin is top-left; pypdf expects bottom-left
                            page_height = page.height
                            llx = float(x0)
                            urx = float(x1)
                            lly = float(page_height - bottom)
                            ury = float(page_height - top)
                            if term_l in mask_set:
                                _add_black_box(writer, page_index, (llx, lly, urx, ury))
                            else:
                                _add_highlight(writer, page_index, (llx, lly, urx, ury))
                            found_any = True
                    if found_any:
                        # Avoid adding multiple overlapping windows for same term
                        break

    with open(dst_pdf, "wb") as f:
        writer.write(f)


def extract_doc_text(src_pdf: Path) -> str:
    texts: List[str] = []
    ocr_used = False
    with pdfplumber.open(str(src_pdf)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if not txt.strip():
                # OCR fallback per page
                try:
                    import pytesseract  # lazy import
                    # allow overriding tesseract binary path and language via env
                    tess_cmd = os.environ.get("TESSERACT_CMD")
                    if tess_cmd:
                        pytesseract.pytesseract.tesseract_cmd = tess_cmd
                    ocr_lang = os.environ.get("TESSERACT_LANG", "eng")
                    ocr_dpi = int(os.environ.get("TESSERACT_DPI", "300"))

                    pil_img = page.to_image(resolution=ocr_dpi).original
                    ocr_text = pytesseract.image_to_string(pil_img, lang=ocr_lang) or ""
                    if ocr_text.strip():
                        txt = ocr_text
                        ocr_used = True
                except Exception:
                    pass
            texts.append(txt)
    return normalize_text("\n".join(texts))


def process_pdf(src_pdf: Path, out_root: Path) -> Tuple[bool, Optional[str], List[Dict[str, str]]]:
	try:
		text = extract_doc_text(src_pdf)
		cand_regex = regex_candidates(text)
		cand_ai = call_openai_detect(text)
		# dedupe by substring
		seen = set()
		detections: List[Dict[str, str]] = []
		for d in cand_regex + cand_ai:
			key = d.get("substring", "").strip()
			if key and key not in seen:
				seen.add(key)
				detections.append(d)
		dst = out_root / src_pdf.parent.name / (src_pdf.stem + ".highlighted.pdf")
		highlight_terms_in_pdf(src_pdf, dst, detections)
		return True, None, detections
	except Exception as e:
		return False, str(e), []


def main() -> int:
	# Usage: python scripts/gdpr_highlight_agent.py [roots...] [--out=path]
	args = sys.argv[1:]
	default_roots = [Path("output") / "images_as_pdfs", Path("archive")]
	out_dir = Path("output") / "gdpr_highlighted"

	# parse --out
	for i, a in list(enumerate(args)):
		if a.startswith("--out="):
			out_dir = Path(a.split("=", 1)[1]).expanduser()
			args.pop(i)
			break

	# parse --log
	log_path: Optional[Path] = None
	for i, a in list(enumerate(args)):
		if a.startswith("--log="):
			log_path = Path(a.split("=", 1)[1]).expanduser()
			args.pop(i)
			break

	roots = [Path(a) for a in args] if args else default_roots
	pdfs = take_first_n_per_folder(roots, n=5)
	if not pdfs:
		print("No PDFs found in provided roots.")
		return 0

	success = 0
	fail = 0
	rows: List[List[str]] = [["pdf_path", "category", "substring"]]
	for pdf in pdfs:
		ok, err, detections = process_pdf(pdf, out_dir)
		if ok:
			print(f"OK  - {pdf} -> {out_dir}")
			success += 1
			for d in detections:
				rows.append([str(pdf), d.get("category", "unknown"), d.get("substring", "")])
		else:
			print(f"ERR - {pdf}: {err}")
			fail += 1

	# Write log
	if log_path is None:
		log_path = out_dir / "detections.csv"
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with open(log_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerows(rows)

	print(f"Done. Processed {success} file(s); {fail} failed.")
	return 0 if fail == 0 else 2


if __name__ == "__main__":
	sys.exit(main())


