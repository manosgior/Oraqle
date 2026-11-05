"""
Extended quantitative comparison for the MCM/QEC evaluation panels.
Reads results/mcm_latency_error_ler.csv (combined MCMit/HERQULES x current/futuristic).
Prints many cuts: same-duration MCMit-vs-HERQULES, MCMit duration sweeps,
HERQULES crossover, current-vs-futuristic, and best/worst per system.
"""
import pandas as pd
import numpy as np

CSV = "results/mcm_latency_error_ler.csv"
LAT = "measure_latency_ns"


def pct_lower(ref, val):
    return np.nan if ref == 0 else (ref - val) / ref * 100.0


def get(df, system, noise, d, dur):
    s = df[(df.system == system) & (df.noise_model == noise) &
           (df.distance == d) & (df[LAT] == dur)]
    return None if s.empty else float(s.logical_error_rate.values[0])


def main():
    df = pd.read_csv(CSV)
    df.logical_error_rate = pd.to_numeric(df.logical_error_rate)
    noises = ["current", "futuristic"]
    dists = sorted(df.distance.unique())
    durs = sorted(df[LAT].unique())

    for noise in noises:
        print(f"\n{'='*70}\nNOISE MODEL: {noise}\n{'='*70}")

        # 1) Same duration: MCMit vs HERQULES, per distance + avg
        print("\n[1] Same readout length -- MCMit LER lower than HERQULES by:")
        for dur in durs:
            imps = []
            for d in dists:
                m, h = get(df, "MCMit", noise, d, dur), get(df, "HERQULES", noise, d, dur)
                if m is None or h is None:
                    continue
                p = pct_lower(h, m)
                line = f"    {dur:>4}ns d={d:<2}: MCMit={m:.5f} HERQULES={h:.5f}"
                if np.isnan(p):
                    print(line + "  -> N/A (HERQULES=0)")
                else:
                    imps.append(p)
                    print(line + f"  -> {p:6.2f}% lower  ({h/m:.1f}x)" if m > 0 else line)
            if imps:
                print(f"    {dur:>4}ns AVG: {np.mean(imps):6.2f}% lower")

        # 2) MCMit duration sweep: each duration vs 1000ns, and vs the worst (250ns)
        print("\n[2] MCMit shorter-readout gains (vs 1000ns, and vs 250ns):")
        for d in dists:
            l1000 = get(df, "MCMit", noise, d, 1000)
            l250 = get(df, "MCMit", noise, d, 250)
            best_dur, best = min(((dur, get(df, "MCMit", noise, d, dur)) for dur in durs
                                  if get(df, "MCMit", noise, d, dur) is not None),
                                 key=lambda x: x[1])
            row = f"    d={d:<2}: best={best:.5f}@{best_dur}ns"
            if l1000:
                row += f" | vs1000ns({l1000:.5f}): {pct_lower(l1000, best):.2f}% lower"
            if l250:
                row += f" | vs250ns({l250:.5f}): {pct_lower(l250, best):.2f}% lower"
            print(row)

        # 3) Specific: 500ns vs 1000ns for both systems
        print("\n[3] 500ns vs 1000ns (same system):")
        for system in ["MCMit", "HERQULES"]:
            for d in dists:
                a, b = get(df, system, noise, d, 500), get(df, system, noise, d, 1000)
                if a is None or b is None:
                    continue
                print(f"    {system:8} d={d:<2}: 500ns={a:.5f} 1000ns={b:.5f} "
                      f"-> 500ns is {pct_lower(b, a):.2f}% {'lower' if a < b else 'higher'}")

        # 4) HERQULES crossover: shortest duration where HERQULES beats MCMit@500
        print("\n[4] HERQULES vs MCMit@500ns -- where does HERQULES catch up?")
        for d in dists:
            m500 = get(df, "MCMit", noise, d, 500)
            if m500 is None:
                continue
            beat = [dur for dur in durs if (get(df, "HERQULES", noise, d, dur) or 1) <= m500]
            tag = f"{min(beat)}ns" if beat else "never (within swept range)"
            print(f"    d={d:<2}: MCMit@500ns={m500:.5f}; HERQULES reaches it at {tag}")

    # 5) Current vs futuristic (MCMit), per distance/duration
    print(f"\n{'='*70}\nMCMit: futuristic improvement over current\n{'='*70}")
    for d in dists:
        for dur in durs:
            c, fut = get(df, "MCMit", "current", d, dur), get(df, "MCMit", "futuristic", d, dur)
            if c is None or fut is None:
                continue
            print(f"    d={d:<2} {dur:>4}ns: current={c:.5f} futuristic={fut:.5f} "
                  f"-> {pct_lower(c, fut):.2f}% lower")

    # 6) Headline best/worst
    print(f"\n{'='*70}\nHeadline numbers\n{'='*70}")
    for noise in noises:
        for system in ["MCMit", "HERQULES"]:
            sub = df[(df.system == system) & (df.noise_model == noise)]
            if sub.empty:
                continue
            bi = sub.logical_error_rate.idxmin()
            print(f"    {system:8} {noise:10}: best LER={sub.logical_error_rate.min():.5f} "
                  f"@ d={int(sub.loc[bi].distance)} {int(sub.loc[bi][LAT])}ns")


if __name__ == "__main__":
    main()
