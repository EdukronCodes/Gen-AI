import re
from pathlib import Path
from typing import Iterable, List, Set


PDF_EXTS = {".pdf"}


def list_pdf_files_by_folder(roots: List[Path]) -> List[Path]:
	files: List[Path] = []
	for root in roots:
		if not root.exists() or not root.is_dir():
			continue
		for p in sorted(root.rglob("*.pdf")):
			files.append(p)
	return files


def take_first_n_per_folder(roots: List[Path], n: int = 5) -> List[Path]:
	"""Recursively scan all subfolders under each root and take first n PDFs per folder."""
	selected: List[Path] = []
	seen_dirs: Set[Path] = set()
	for root in roots:
		if not root.exists() or not root.is_dir():
			continue
		# include the root itself and all subdirectories
		dirs: List[Path] = [root] + [p for p in root.rglob('*') if p.is_dir()]
		for d in sorted(dirs):
			if d in seen_dirs:
				continue
			seen_dirs.add(d)
			pdfs = sorted(d.glob('*.pdf'))
			if pdfs:
				selected.extend(pdfs[:n])
	return selected


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
	text = text.replace("\x00", " ")
	text = WHITESPACE_RE.sub(" ", text)
	return text.strip()


