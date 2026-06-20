"""Batch driver for the EHRAgent MEXTRA reproduction. Runs a named suite."""
import sys, os, json
sys.path.insert(0, "/home/user/MEXTRA2")  # make `repro` importable when run as a script
import repro.realrun as R   # realrun only parses argv under __main__, safe to import

Q = "EHRAgent/running/queries"
def P(name): return os.path.join(R.REPO, Q, name)

def nsweep(out, k, label):
    """Print EN/RN at n prefixes 10..50 from a single n=50 run."""
    rows = []
    for n in [10, 20, 30, 40, 50]:
        if n <= len(out["results"]):
            m = R.compute_metrics(out["results"][:n], k)
            rows.append((n, m["EN"], m["RN"], m["EE"], m["CER"], m["AER"]))
    print(f"\n[n-sweep {label}]  n: EN RN EE CER AER")
    for r in rows:
        print(f"   n={r[0]:>2}: EN={r[1]} RN={r[2]} EE={r[3]} CER={r[4]} AER={r[5]}")
    return rows

def main(suite):
    if suite == "nsweep_edit":
        b = R.run_experiment(P("general/general_50.json"), 50, 200, 4, "edit_distance", 12, "ehr_edit_basic_n50_m200_k4")
        nsweep(b, 4, "EHRAgent edit BASIC (Ibasic)")
        a = R.run_experiment(P("edit_specific/edit_specific_50.json"), 50, 200, 4, "edit_distance", 12, "ehr_edit_advan_n50_m200_k4")
        nsweep(a, 4, "EHRAgent edit ADVANCED (Iadvan)")

    elif suite == "msweep_edit":
        for m in [50, 100, 300, 400, 500]:
            R.run_experiment(P("general/general_30.json"), 30, m, 4, "edit_distance", 12, f"ehr_edit_basic_n30_m{m}_k4")

    elif suite == "ksweep_edit":
        for k in [1, 2, 3, 5]:
            R.run_experiment(P("general/general_30.json"), 30, 200, k, "edit_distance", 12, f"ehr_edit_basic_n30_m200_k{k}")

    elif suite == "cosine":
        R.run_experiment(P("cosine_specific/cosine_specific_30.json"), 30, 200, 4, "cosine", 8, "ehr_cos_advan_n30_m200_k4")
        R.run_experiment(P("general/general_30.json"), 30, 200, 4, "cosine", 8, "ehr_cos_basic_n30_m200_k4")

    elif suite == "msweep_cos":
        for m in [50, 100, 300, 400, 500]:
            R.run_experiment(P("cosine_specific/cosine_specific_30.json"), 30, m, 4, "cosine", 8, f"ehr_cos_advan_n30_m{m}_k4")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
