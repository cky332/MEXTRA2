"""Aggregate all repro results and compare against the paper's reported numbers."""
import sys, os, json, glob
sys.path.insert(0, "/home/user/MEXTRA2")
from repro.analyze import classify

RES = "/home/user/MEXTRA2/repro/results"

# ---- paper reference numbers ----
PAPER = {
    "Table1 EHRAgent MEXTRA (edit,n30,m200,k4)": dict(EN=50, RN=55, EE=0.42, CER=0.83, AER=0.83),
    "Table1 RAP MEXTRA (cos,n30,m200,k3)":       dict(EN=26, RN=27, EE=0.29, CER=0.87, AER=0.90),
}
PAPER_EHR_EDIT_M = {50:31, 100:43, 200:50, 300:51, 400:58, 500:59}   # Table 2, EN
PAPER_EHR_COS_MINILM_M = {50:14, 100:20, 200:20, 300:23, 400:27, 500:24}
PAPER_EHR_K = {1:(8,21), 2:(27,34), 3:(39,45), 4:(50,55), 5:(59,65)}  # (EN,RN) Fig3a

def load(tag):
    fp = os.path.join(RES, tag + ".json")
    return json.load(open(fp)) if os.path.exists(fp) else None

def fmt(m): return f"EN={m['EN']} RN={m['RN']} EE={m['EE']} CER={m['CER']} AER={m['AER']}"

def failbreak(d):
    c = {"FULL":0,"PARTIAL":0,"ZERO":0}; reasons={}
    for s in d["results"]:
        kind,_,_,reason = classify(s)
        c[kind]+=1
        if kind!="FULL": reasons[reason]=reasons.get(reason,0)+1
    return c, reasons

def section(title): print("\n"+"="*78+"\n"+title+"\n"+"="*78)

def main():
    section("HEADLINE: EHRAgent edit, n=30, m=200, k=4  (Table 1)")
    # prefer the n50 run's n=30 prefix if present, else standalone
    import repro.realrun as R
    d = load("ehr_edit_basic_n50_m200_k4") or load("ehr_edit_basic_n30_m200_k4")
    if d:
        res = d["results"][:30]
        m = R.compute_metrics(res, 4)
        p = PAPER["Table1 EHRAgent MEXTRA (edit,n30,m200,k4)"]
        print("  PAPER (GPT-4o): ", p)
        print("  REPRO (DeepSeek):", fmt(m))
        c, reasons = failbreak({"results":res})
        print("  outcomes:", c, " failure reasons:", reasons)

    section("n-SWEEP  EHRAgent edit (Fig 4a) — basic vs advanced")
    for tag,label in [("ehr_edit_basic_n50_m200_k4","BASIC Ibasic"),
                      ("ehr_edit_advan_n50_m200_k4","ADVANCED Iadvan")]:
        d = load(tag)
        if not d: continue
        print(f"  [{label}]")
        for n in [10,20,30,40,50]:
            m = R.compute_metrics(d["results"][:n], 4)
            print(f"    n={n:>2}: {fmt(m)}")

    section("m-SWEEP  EHRAgent edit (Table 2 / Fig 2a), n=30 k=4")
    print(f"  {'m':>4} | {'REPRO EN':>8} {'RN':>4} {'EE':>5} | {'PAPER EN':>8}")
    for m_ in [50,100,200,300,400,500]:
        d = load(f"ehr_edit_basic_n30_m{m_}_k4")
        if m_==200 and not d:
            dd = load("ehr_edit_basic_n50_m200_k4")
            d = {"results": dd["results"][:30]} if dd else None
        if not d: continue
        mm = R.compute_metrics(d["results"], 4)
        print(f"  {m_:>4} | {mm['EN']:>8} {mm['RN']:>4} {mm['EE']:>5} | {PAPER_EHR_EDIT_M[m_]:>8}")

    section("k-SWEEP  EHRAgent edit (Fig 3a), n=30 m=200")
    print(f"  {'k':>2} | {'REPRO EN':>8} {'RN':>4} | PAPER (EN,RN)")
    for k_ in [1,2,3,4,5]:
        d = load(f"ehr_edit_basic_n30_m200_k{k_}")
        if k_==4 and not d:
            dd = load("ehr_edit_basic_n50_m200_k4")
            d = {"results": dd["results"][:30]} if dd else None
        if not d: continue
        mm = R.compute_metrics(d["results"], k_)
        print(f"  {k_:>2} | {mm['EN']:>8} {mm['RN']:>4} | {PAPER_EHR_K[k_]}")

    section("COSINE  EHRAgent (bge-m3 SUBSTITUTE for MiniLM) — DEVIATION")
    for tag,label in [("ehr_cos_advan_n30_m200_k4","advanced Iadvan(cosine_specific)"),
                      ("ehr_cos_basic_n30_m200_k4","basic Ibasic(general)")]:
        d = load(tag)
        if not d: continue
        mm = R.compute_metrics(d["results"], 4)
        print(f"  [{label}] REPRO: {fmt(mm)}   (paper MiniLM cos m200 EN≈20)")

    section("RAP (REDUCED: mocked webshop + bge-m3) — DEVIATION")
    for tag,label in [("rap_cos_basic_n30_m200_k3","basic Ibasic"),
                      ("rap_cos_advan_n30_m200_k3","advanced category")]:
        d = load(tag)
        if not d: continue
        mm = R.compute_metrics(d["results"], 3)
        c, reasons = failbreak(d)
        print(f"  [{label}] REPRO: {fmt(mm)}  outcomes={c} reasons={reasons}")
        print(f"     (paper RAP MEXTRA n30 m200: EN=26 RN=27 EE=0.29 CER=0.87 AER=0.90)")

if __name__ == "__main__":
    main()
