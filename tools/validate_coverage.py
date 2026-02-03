from __future__ import annotations
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

def read_policy_coverage_min(repo_root: Path, policy_pack: str) -> float:
    p = repo_root / "policy" / f"{policy_pack}.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return float(data["testing"]["coverage_min"])

def parse_coverage_xml_rate(path: Path) -> float:
    tree = ET.parse(path)
    root = tree.getroot()
    # coverage.py XML은 보통 <coverage line-rate="0.85" ...>
    lr = root.attrib.get("line-rate")
    if lr is None:
        raise RuntimeError("coverage.xml missing 'line-rate' attribute")
    return float(lr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-xml", required=True)
    ap.add_argument("--policy-pack", default="defense-lite")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cov_path = Path(args.coverage_xml).resolve()

    cov_min = read_policy_coverage_min(repo_root, args.policy_pack)
    cov_rate = parse_coverage_xml_rate(cov_path)

    print(f"[coverage] rate={cov_rate:.4f} min={cov_min:.4f}")
    if cov_rate + 1e-12 < cov_min:
        raise SystemExit(f"Coverage gate failed: {cov_rate:.4f} < {cov_min:.4f}")

if __name__ == "__main__":
    main()
