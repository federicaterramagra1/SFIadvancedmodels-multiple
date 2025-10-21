
"""
Hoeffding post-hoc checks + Anytime CI (robust parser).
Gestisce:
  - STAT: FRcrit_avg= / avg FRcrit= / avgFRcrit= / FR_avg(critical)= / FR_avg= / FR_avg: ...
  - n: injections_used= / n= / "on <n> injections"
  - ESAUSTIVA: "Failure Rate (critical): ..." OPPURE "Average FR: ... on ... injections"
  - Wilson: rilegge dal file o ricalcola se mancante.
"""

import re, math, argparse, csv
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Optional, List, Dict


# ---------- Wilson (fallback) ----------
def wilson_ci(p_hat: float, n: int, conf: float = 0.95):
    if n <= 0:
        return 0.0, 1.0, 0.5
    z = {0.90:1.645,0.95:1.96,0.99:2.576}.get(conf,1.96)
    p = min(max(p_hat, 1e-12), 1-1e-12)
    denom = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z * math.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))) / denom
    return max(0.0, center-half), min(1.0, center+half), half

# ---------- Anytime-valid Hoeffding ----------
def alpha_spend(n:int, alpha_total:float=0.05)->float:
    return alpha_total * 6.0/(math.pi**2) / (n**2)

def hoeffding_anytime_ci(p_hat:float, n:int, alpha_total:float=0.05):
    if n<=0: return 0.0,1.0,0.5
    h = math.sqrt(math.log(2.0/alpha_spend(n,alpha_total)) / (2.0*n))
    return max(0.0,p_hat-h), min(1.0,p_hat+h), h

# ---------- Regex ----------
NUM = r'[0-9.eE+-]+'
RE_N_FILE  = re.compile(r'_STAT_N(\d+)_batch', re.IGNORECASE)
RE_N_LINE  = re.compile(r'\bN\s*=\s*(\d+)\b')

# p-hat (STAT)
RE_PHAT = [
    re.compile(r'\bFRcrit_avg\s*[:=]\s*(' + NUM + r')', re.IGNORECASE),
    re.compile(r'\bavg\s*FRcrit\s*[:=]\s*(' + NUM + r')', re.IGNORECASE),
    re.compile(r'\bavgFRcrit\s*[:=]\s*(' + NUM + r')', re.IGNORECASE),
    re.compile(r'\bFR_avg(?:\s*\(critical\))?\s*[:=]\s*(' + NUM + r')', re.IGNORECASE),
]

# n (STAT)
RE_INJ = [
    re.compile(r'\binjections_used\s*=\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'\bn\s*=\s*(\d+)\b'),
    re.compile(r'\bon\s*(\d+)\s*injections\b', re.IGNORECASE),
]

# Wilson
RE_WILSON = [
    re.compile(r'WilsonCI\(\s*\d+%\s*\):\s*\[(' + NUM + r')\s*,\s*(' + NUM + r')\]\s*half\s*=\s*(' + NUM + r')', re.IGNORECASE),
    re.compile(r'\bCI\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]', re.IGNORECASE),
]

RE_BASE     = re.compile(r'baseline_pred_dist\s*=\s*\[([^\]]+)\]', re.IGNORECASE)
RE_TOP_FR   = re.compile(r'\bFRcrit\s*=\s*(' + NUM + r')\b')

# ESAUSTIVA: FR
RE_EX_FR = [
    re.compile(r'failure\s*rate(?:\s*\(critical\))?\s*[:=]\s*(' + NUM + r')', re.IGNORECASE),
    re.compile(r'average\s*FR(?:\s*\(critical\))?\s*[:=]\s*(' + NUM + r')', re.IGNORECASE),
]

def parse_floats_list(s:str)->List[float]:
    return [float(x) for x in re.findall(NUM, s)]

# ---------- Parsers ----------
def parse_stat_file(path: Path) -> Dict:
    txt = path.read_text(encoding='utf-8', errors='ignore')

    mN = RE_N_LINE.search(txt) or RE_N_FILE.search(path.name)
    N = int(mN.group(1)) if mN else None

    p_hat = None
    for R in RE_PHAT:
        m = R.search(txt)
        if m: 
            p_hat = float(m.group(1))
            break

    n_inj = None
    for R in RE_INJ:
        m = R.search(txt)
        if m: 
            n_inj = int(m.group(1))
            break

    w_low = w_high = w_half = None
    for R in RE_WILSON:
        m = R.search(txt)
        if m:
            vals = list(map(float, m.groups()))
            if len(vals)==3:
                w_low, w_high, w_half = vals
            else:
                w_low, w_high = vals[0], vals[1]
                w_half = (w_high - w_low)/2.0
            break

    mB = RE_BASE.search(txt)
    baseline = parse_floats_list(mB.group(1)) if mB else None
    top_fr = [float(x) for x in RE_TOP_FR.findall(txt)]

    # dataset/net/batch
    parts = path.parts
    dataset = net = batch = None
    for i,p in enumerate(parts):
        if p.startswith('batch_'):
            batch = p.split('_',1)[1]; net = parts[i-1]; dataset = parts[i-2]; break

    return dict(file=str(path), dataset=dataset, net=net, batch=batch,
                N=N, p_hat=p_hat, injections_used=n_inj,
                wilson_low=w_low, wilson_high=w_high, wilson_half=w_half,
                baseline=baseline, top_fr=top_fr)


def find_stat_files(root: Path) -> List[Path]:
    # Considera tutti i file dentro minimal_stat che contengono _STAT_N nel nome
    return [p for p in root.rglob("*.txt") if "_STAT_N" in p.name]

def guess_exhaustive_file(stat_path: Path, N: int) -> Optional[Path]:
    try:
        parts = stat_path.parts
        for i, p in enumerate(parts):
            if p.startswith("batch_"):
                batch = p.split("_", 1)[1]
                net = parts[i-1]
                dataset = parts[i-2]
                break
        minimal_dir = stat_path.parent.parent / "minimal"
        return minimal_dir / f"{dataset}_{net}_minimal_N{N}_batch{batch}.txt"
    except Exception:
        return None

def parse_exhaustive_fr(path: Optional[Path]) -> Optional[float]:
    if not path or not path.exists():
        return None
    txt = path.read_text(encoding='utf-8', errors='ignore')
    for R in RE_EX_FR:
        m = R.search(txt)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    # fallback generico
    for line in txt.splitlines():
        if any(k in line.lower() for k in ("failure", "average fr")):
            nums = re.findall(NUM, line)
            if nums:
                try:
                    return float(nums[0])
                except ValueError:
                    pass
    return None

# ---------- Source design checks ----------
def scan_source_design_ok(src_root: Path) -> Tuple[bool, bool]:
    py_files = list(src_root.rglob("*.py"))
    text = "\n".join(p.read_text(encoding='utf-8', errors='ignore') for p in py_files)
    has_restore = ("restore_golden" in text) or ("restore_golden()" in text)
    has_srs = (("def srs_combinations" in text) and ("rnd.sample" in text)) or ("random.sample(" in text)
    return has_restore, has_srs

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results_minimal", help="Root dei risultati (contiene */minimal_stat/*_STAT_N*.txt)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha globale per Anytime (default 0.05)")
    ap.add_argument("--csv", type=str, default="hoeffding_checks.csv", help="Output CSV")
    ap.add_argument("--report", type=str, default="hoeffding_checks.txt", help="Output report testuale")
    ap.add_argument("--src-root", type=str, default=".", help="Root sorgenti per flag design (restore_golden, srs)")
    args = ap.parse_args()

    root = Path(args.root)
    has_restore, has_srs = scan_source_design_ok(Path(args.src_root))

    paths = find_stat_files(root)
    rows, groups_baseline = [], defaultdict(list)

    for path in sorted(paths):
        info = parse_stat_file(path)
        N, p_hat, n = info["N"], info["p_hat"], info["injections_used"]
        if N is None or p_hat is None or n is None:
            # file non parsato (manca N o p_hat o n)
            continue

        # Anytime (Hoeffding, alpha-spending)
        a_low, a_high, a_half = hoeffding_anytime_ci(p_hat, n, args.alpha)

        # Wilson: prendi dal file se presente, altrimenti ricalcola
        w_low, w_high, w_half = info["wilson_low"], info["wilson_high"], info["wilson_half"]
        if w_low is None or w_high is None or w_half is None:
            w_low, w_high, w_half = wilson_ci(p_hat, n)

        # Esaustiva (se esiste il gemello in minimal/)
        ex_path = guess_exhaustive_file(Path(info["file"]), N)
        fr_exh = parse_exhaustive_fr(ex_path)
        exh_in_wilson = (fr_exh is not None and w_low <= fr_exh <= w_high)
        exh_in_any = (fr_exh is not None and a_low <= fr_exh <= a_high)

        # bounding & baseline
        bounded_ok = (0.0 <= p_hat <= 1.0) and all(0.0 <= v <= 1.0 for v in (info["top_fr"] or [p_hat]))
        key = (info["dataset"], info["net"], info["batch"])
        if info["baseline"]:
            groups_baseline[key].append(info["baseline"])

        rows.append(dict(
            dataset=info["dataset"], net=info["net"], batch=info["batch"], N=N,
            p_hat_stat=p_hat, injections_used=n,
            anytime_low=a_low, anytime_high=a_high, anytime_half=a_half,
            wilson_low=w_low, wilson_high=w_high, wilson_half=w_half,
            FR_exhaustive=fr_exh,
            exh_in_wilson=exh_in_wilson if fr_exh is not None else None,
            exh_in_anytime=exh_in_any if fr_exh is not None else None,
            bounded_ok=bounded_ok,
            design_has_restore=has_restore, design_has_srs=has_srs,
            stat_file=str(path), exhaustive_file=str(ex_path) if ex_path else None
        ))

    # baseline const check per (dataset, net, batch)
    baseline_ok_map: Dict[Tuple[str, str, str], bool] = {}
    for key, lists in groups_baseline.items():
        ok = True
        if len(lists) >= 2:
            ref = lists[0]
            for v in lists[1:]:
                if len(ref) != len(v) or sum(abs(a - b) for a, b in zip(ref, v)) > 1e-9:
                    ok = False
                    break
        baseline_ok_map[key] = ok

    # CSV
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset","net","batch","N","p_hat_stat","injections_used",
              "anytime_low","anytime_high","anytime_half",
              "wilson_low","wilson_high","wilson_half",
              "FR_exhaustive","exh_in_wilson","exh_in_anytime",
              "bounded_ok","baseline_const_ok","design_has_restore","design_has_srs",
              "stat_file","exhaustive_file"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r["baseline_const_ok"] = baseline_ok_map.get((r["dataset"], r["net"], r["batch"]), True)
            r["design_has_restore"] = bool(r["design_has_restore"])
            r["design_has_srs"] = bool(r["design_has_srs"])
            w.writerow(r)

    # Report
    out_report = Path(args.report)
    with out_report.open("w", encoding="utf-8") as f:
        f.write("# Hoeffding checks (post-hoc)\n\n")
        f.write(f"Source design: restore_golden={has_restore}, srs_sampling={has_srs}\n\n")
        grp = defaultdict(list)
        for r in rows:
            grp[(r["dataset"], r["net"])].append(r)
        for (ds, net), items in grp.items():
            f.write(f"## {ds} / {net}\n")
            items = sorted(items, key=lambda x: x["N"])
            # prendi una chiave per baseline const
            bkey = (items[0]["dataset"], items[0]["net"], items[0]["batch"])
            f.write(f"- Baseline const across runs: {baseline_ok_map.get(bkey, True)}\n")
            for r in items:
                line = (f"N={r['N']:>4d}  p̂={r['p_hat_stat']:.6f}  n={r['injections_used']:<6d}  "
                        f"Anytime[{r['anytime_low']:.6f},{r['anytime_high']:.6f}]  "
                        f"Wilson[{r['wilson_low']:.6f},{r['wilson_high']:.6f}]  "
                        f"bounded_ok={r['bounded_ok']}")
                if r["FR_exhaustive"] is not None:
                    line += (f"  | EXH={r['FR_exhaustive']:.6f} "
                             f"in_Wilson={r['exh_in_wilson']} in_Anytime={r['exh_in_anytime']}")
                f.write(line + "\n")
            f.write("\n")

    print(f"Wrote CSV → {out_csv}")
    print(f"Wrote report → {out_report}")

if __name__ == "__main__":
    main()
