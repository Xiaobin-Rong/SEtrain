#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan directories and write CSV lists for dataset generation.")
    parser.add_argument("--clean-root", type=Path, nargs="+", required=True, help="One or more folders with clean speech wav files.")
    parser.add_argument("--noise-root", type=Path, nargs="+", required=True, help="One or more folders with noise wav files.")
    parser.add_argument("--rir-root", type=Path, nargs="+", required=True, help="One or more folders with RIR wav files.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent, help="Where to place the generated CSV files.")
    parser.add_argument("--split", default="train", help="Prefix for the CSV filenames, e.g. train or dev.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of entries per CSV.")
    parser.add_argument("--relative", action="store_true", help="Store paths relative to output directory when possible.")
    parser.add_argument("--ext", nargs="*", default=[".wav"], help="File extensions to include (default: .wav).")
    return parser.parse_args()


def collect_files(roots: Iterable[Path], exts: Iterable[str]) -> List[Path]:
    exts = {ext.lower() for ext in exts}
    files: List[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        joined = ", ".join(str(p.expanduser().resolve()) for p in roots)
        raise RuntimeError(f"No matching files found under: {joined}")
    files.sort()
    return files


def maybe_limit(items: List[Path], limit: int | None) -> List[Path]:
    if limit is None:
        return items
    return items[:limit]


def to_output_path(path: Path, output_dir: Path, make_relative: bool) -> str:
    resolved = path.resolve()
    if make_relative:
        try:
            return str(resolved.relative_to(output_dir.resolve()))
        except ValueError:
            pass
    return str(resolved)


def write_csv(entries: List[Path], csv_path: Path, output_dir: Path, make_relative: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_dir"])
        for entry in entries:
            writer.writerow([to_output_path(entry, output_dir, make_relative)])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    clean_files = maybe_limit(collect_files(args.clean_root, args.ext), args.limit)
    noise_files = maybe_limit(collect_files(args.noise_root, args.ext), args.limit)
    rir_files = maybe_limit(collect_files(args.rir_root, args.ext), args.limit)

    write_csv(clean_files, output_dir / f"{args.split}_clean_dir.csv", output_dir, args.relative)
    write_csv(noise_files, output_dir / f"{args.split}_noise_dir.csv", output_dir, args.relative)
    write_csv(rir_files, output_dir / f"{args.split}_rir_dir.csv", output_dir, args.relative)

    print(f"Wrote {len(clean_files)} clean, {len(noise_files)} noise, {len(rir_files)} rir entries to {output_dir}")


if __name__ == "__main__":
    main()
