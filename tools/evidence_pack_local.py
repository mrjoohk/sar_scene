from __future__ import annotations
import json, hashlib, zipfile
from pathlib import Path
import argparse

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-id", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--ci-run-url", required=True)
    ap.add_argument("--trace-matrix", required=True)
    ap.add_argument("--junit", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--out-dir", default="evidence/manifests")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline_id": args.baseline_id,
        "repo": args.repo,
        "commit": args.commit,
        "ci_run_url": args.ci_run_url,
        "artifacts": {
            "trace_matrix": args.trace_matrix,
            "junit.xml": args.junit,
            "coverage.xml": args.coverage,
        },
    }

    manifest_path = out_dir / f"manifest_{args.baseline_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = out_dir / f"evidence_{args.baseline_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(manifest_path, arcname=manifest_path.name)
        for p in [Path(args.trace_matrix), Path(args.junit), Path(args.coverage)]:
            if p.exists():
                z.write(p, arcname=str(p))

    checksum = sha256_file(zip_path)
    checksum_path = out_dir / f"evidence_{args.baseline_id}.sha256"
    checksum_path.write_text(f"{checksum}  {zip_path.name}\n", encoding="utf-8")

    print(str(zip_path))
    print(str(manifest_path))
    print(str(checksum_path))

if __name__ == "__main__":
    main()
