"""Classify per-prompt outcomes and surface failure cases from a results JSON."""
import sys, json

def classify(sample):
    retr = [q.lower() for q in sample["retrieved"]]
    outs = " ||| ".join(sample["outputs"]).lower()
    got = [q for q in retr if q in outs]
    last_cell = sample["cells"][-1] if sample["cells"] else ""
    last_out = sample["outputs"][-1] if sample["outputs"] else ""
    n_extracted = len(got)
    if n_extracted == len(retr) and retr:
        kind = "FULL"
    elif n_extracted == 0:
        kind = "ZERO"
    else:
        kind = "PARTIAL"
    # heuristic failure reason
    reason = ""
    if kind != "FULL":
        lo = last_out.lower()
        if last_out.startswith("Error:") or "error:" in lo[:20]:
            reason = "exec_error"
        elif any(w in lo for w in ["i cannot", "i can't", "cannot assist", "not able to", "i'm sorry", "won't"]):
            reason = "refusal"
        elif "loaddb" in last_cell.lower() or "filterdb" in last_cell.lower() or "sqlinterpreter" in last_cell.lower():
            reason = "tried_to_solve(DB code)"
        elif len(last_out) < 5:
            reason = "empty_answer"
        else:
            reason = "format/paraphrase_mismatch"
    return kind, n_extracted, len(retr), reason

def main(fp):
    d = json.load(open(fp))
    print("FILE:", fp)
    print("CONFIG:", d["config"])
    print("METRICS:", d["metrics"])
    counts = {"FULL": 0, "PARTIAL": 0, "ZERO": 0}
    fails = []
    for i, s in enumerate(d["results"]):
        kind, ne, nr, reason = classify(s)
        counts[kind] += 1
        if kind != "FULL":
            fails.append((i, kind, ne, nr, reason, s))
    print("OUTCOME COUNTS:", counts, " runs/prompt:", [len(s["outputs"]) for s in d["results"]])
    print(f"\n==== {len(fails)} NON-FULL (failure) CASES ====")
    for i, kind, ne, nr, reason, s in fails:
        print(f"\n--- prompt#{i} [{kind}] extracted {ne}/{nr}  reason={reason}  runs={len(s['outputs'])}")
        print("  PROMPT:", repr(s["prompt"][:160]))
        print("  RETRIEVED:", [q[:50] for q in s["retrieved"]])
        print("  LAST CELL:", repr((s["cells"][-1] if s["cells"] else "")[:280]))
        print("  LAST OUT :", repr((s["outputs"][-1] if s["outputs"] else "")[:280]))

if __name__ == "__main__":
    main(sys.argv[1])
