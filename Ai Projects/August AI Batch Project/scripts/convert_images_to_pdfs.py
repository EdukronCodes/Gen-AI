import sys
from pathlib import Path
from typing import Iterable, List, Tuple

try:
	from PIL import Image
except ImportError as exc:
	print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
	raise


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_recursive(root: Path) -> List[Path]:
	files: List[Path] = []
	for ext in SUPPORTED_EXTS:
		files.extend(sorted(root.rglob(f"*{ext}")))
	return files


def convert_image_to_pdf(img_path: Path, output_path: Path) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with Image.open(img_path) as im:
		if im.mode in ("RGBA", "P"):
			im = im.convert("RGB")
		elif im.mode == "LA":
			im = im.convert("RGB")
		elif im.mode == "L":
			im = im.convert("RGB")

		im.save(output_path, "PDF", resolution=300.0)

	return output_path


def main() -> int:
	# Usage:
	#   python scripts/convert_images_to_pdfs.py [input_root1] [input_root2] ... [output_dir]
	# If only one extra arg is passed, treat it as input_root; output defaults to ./output/images_as_pdfs
	# If more than one arg is passed, last arg may be --out=PATH to set output directory.

	args = sys.argv[1:]
	default_input = Path("archive")
	default_output = Path("output") / "images_as_pdfs"

	# Parse optional --out= flag
	out_flag_indices: List[int] = [i for i, a in enumerate(args) if a.startswith("--out=")]
	output_dir = default_output
	if out_flag_indices:
		idx = out_flag_indices[-1]
		output_dir = Path(args[idx].split("=", 1)[1]).expanduser()
		args.pop(idx)

	input_roots: List[Path]
	if not args:
		input_roots = [default_input]
	else:
		input_roots = [Path(a) for a in args]

	# Validate roots
	valid_roots: List[Path] = []
	for root in input_roots:
		if root.exists() and root.is_dir():
			valid_roots.append(root)
		else:
			print(f"Skip: not a directory -> {root}", file=sys.stderr)

	if not valid_roots:
		print("No valid input directories provided.", file=sys.stderr)
		return 1

	success = 0
	fail = 0
	for root in valid_roots:
		images = list_images_recursive(root)
		if not images:
			print(f"No images found in {root}. Supported: {sorted(SUPPORTED_EXTS)}")
			continue

		for img in images:
			try:
				# Mirror directory structure under output/{root.name}
				rel = img.relative_to(root)
				pdf_path = output_dir / root.name / rel.with_suffix(".pdf")
				result = convert_image_to_pdf(img, pdf_path)
				print(f"OK  - {img} -> {result}")
			except Exception as e:  # noqa: BLE001
				fail += 1
				print(f"ERR - {img}: {e}", file=sys.stderr)
			else:
				success += 1

	print(f"Done. Converted {success} file(s); {fail} failed.")
	return 0 if fail == 0 else 2


if __name__ == "__main__":
	sys.exit(main())


