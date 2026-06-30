#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Try importing fitz and PIL for PDF rendering
try:
    import fitz
    from PIL import Image
except ImportError:
    print("Error: PyMuPDF (fitz) or Pillow (PIL) is not installed in the python environment.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"
PDF = Path("/Users/will/github/mineru-model/testdata/nasa_systems_engineering_handbook_rev2.pdf")

def render_page(page_number: int, out_path: Path) -> None:
    doc = fitz.open(PDF)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(out_path)

def verify_output(json_path: Path, expected_blocks: int, expected_types: list[str]) -> bool:
    if not json_path.exists():
        print(f"Error: Output file {json_path} was not created.")
        return False
    try:
        blocks = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error: Failed to parse output JSON file {json_path}: {e}")
        return False
        
    block_count = len(blocks)
    types = [b.get("type") for b in blocks]
    
    if block_count != expected_blocks:
        print(f"Error: Mismatch in block count for {json_path.name}!")
        print(f"  Expected: {expected_blocks}, Found: {block_count}")
        return False
        
    if types != expected_types:
        print(f"Error: Mismatch in block types for {json_path.name}!")
        print(f"  Expected: {expected_types}, Found: {types}")
        return False
        
    print(f"Success: Correctness verified for {json_path.name} (blocks={block_count}, types={types}).")
    return True

def run_mu(image_path: Path, output_dir: Path, threads: int) -> bool:
    cmd = [
        str(MU),
        "--backend", "metal",
        "--no-cpu-fallback",
        "--json",
        "--kv-cache-bf16",
        "--use-icb",
        "--threads", str(threads),
        "--image", str(image_path),
        "--output-dir", str(output_dir)
    ]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error: mu process exited with non-zero code {res.returncode}")
        print("--- STDERR ---")
        print(res.stderr)
        print("--------------")
        return False
    return True

def main():
    if not MU.exists():
        print(f"Error: mu executable not found at {MU}. Please build it first.")
        sys.exit(1)
    if not PDF.exists():
        print(f"Error: Reference PDF handbook not found at {PDF}.")
        sys.exit(1)
        
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        img_path = tmp_dir / "page_224.png"
        
        print("Rendering reference page 224...")
        render_page(224, img_path)
        
        # 1. Run with 1 thread
        print("\n--- Test 1: Single Thread ---\n")
        out_dir_t1 = tmp_dir / "out_t1"
        out_dir_t1.mkdir()
        if not run_mu(img_path, out_dir_t1, 1):
            sys.exit(1)
        if not verify_output(out_dir_t1 / "page_224.json", 3, ["table", "footer", "page_number"]):
            sys.exit(1)
            
        # 2. Run with 4 concurrent threads (verify thread isolation)
        print("\n--- Test 2: Concurrent Threads (threads=4) ---\n")
        out_dir_t4 = tmp_dir / "out_t4"
        out_dir_t4.mkdir()
        # We pass page_224.png twice to verify concurrent multi-image safety
        img_path2 = tmp_dir / "page_224_copy.png"
        render_page(224, img_path2)
        
        cmd = [
            str(MU),
            "--backend", "metal",
            "--no-cpu-fallback",
            "--json",
            "--kv-cache-bf16",
            "--use-icb",
            "--threads", "4",
            "--image", str(img_path),
            "--image", str(img_path2),
            "--output-dir", str(out_dir_t4)
        ]
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"Error: concurrent mu process failed with code {res.returncode}")
            print(res.stderr)
            sys.exit(1)
            
        if not verify_output(out_dir_t4 / "page_224.json", 3, ["table", "footer", "page_number"]) or \
           not verify_output(out_dir_t4 / "page_224_copy.json", 3, ["table", "footer", "page_number"]):
            sys.exit(1)
            
    print("\nALL REGRESSION CHECKS PASSED SUCCESSFULLY!")
    sys.exit(0)

if __name__ == "__main__":
    main()
