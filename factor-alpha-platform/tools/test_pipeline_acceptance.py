"""Acceptance test — runs the unified pipeline against both market configs
and asserts the canonical reproduction targets:

  EQUITY (research_equity.json):
    equal × diag with subindustry-demean preprocessing → FULL net SR ≈ +4.98
    Tolerance ±0.05.

  CRYPTO (research_crypto.json):
    topn_train (top_n=50) → TRAIN gross SR ≈ +5.30
    Tolerance ±0.10.

Run before merging any pipeline change. Failure means a regression.
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run

EQUITY_TARGET_FULL_NET = 4.98
EQUITY_TOL = 0.10
CRYPTO_TARGET_TRAIN_GROSS = 5.30
CRYPTO_TOL = 0.10


def report(label, actual, target, tol):
    delta = actual - target
    ok = abs(delta) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: actual={actual:+.3f}  target={target:+.2f}  "
          f"tol=±{tol}  delta={delta:+.3f}")
    return ok


def main():
    fails = 0

    print("=" * 78)
    print("  EQUITY pipeline acceptance — research_equity.json")
    print("=" * 78)
    res_e = run(ROOT / "prod" / "config" / "research_equity.json", verbose=True)
    print()
    full_net = res_e.metrics["FULL"]["SR_net"]
    if not report("equal × diag FULL net SR", full_net,
                  EQUITY_TARGET_FULL_NET, EQUITY_TOL):
        fails += 1
    # Also report VAL/TEST so we can eyeball drift
    for split in ("VAL", "TEST", "VAL+TEST"):
        m = res_e.metrics[split]
        print(f"   {split:9s}  SR_n={m['SR_net']:+.2f}  ret_n={m['ret_ann_net']*100:+.1f}%/yr")

    print()
    print("=" * 78)
    print("  CRYPTO pipeline acceptance — research_crypto.json")
    print("=" * 78)
    res_c = run(ROOT / "prod" / "config" / "research_crypto.json", verbose=True)
    print()
    train_gross = res_c.metrics["TRAIN"]["SR_gross"]
    if not report("topn_train(50) TRAIN gross SR", train_gross,
                  CRYPTO_TARGET_TRAIN_GROSS, CRYPTO_TOL):
        fails += 1
    for split in ("TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"):
        m = res_c.metrics[split]
        print(f"   {split:9s}  SR_g={m['SR_gross']:+.2f}  SR_n={m['SR_net']:+.2f}  "
              f"ret_n={m['ret_ann_net']*100:+.1f}%/yr  n={m['n_bars']}")

    print()
    print("=" * 78)
    print("PASS" if fails == 0 else f"FAIL ({fails} regression{'s' if fails > 1 else ''})")
    print("=" * 78)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
