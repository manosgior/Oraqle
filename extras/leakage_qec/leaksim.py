"""leaksim.py -- readout-induced ancilla leakage on top of the lattice-sim circuits.

lattice-sim has no leakage and uses one circuit for sampling and decoding. Here a shot's leakage history is drawn from a
Markov chain and compiled into two circuits built from the same lattice-sim circuit:

  truth circuit    what happens to the qubits; only sampled
  belief circuit   what the decoder is told; only used for its detector error model

Physics of an ancilla that leaves a readout in L (see leakage/docs/LEAKAGE_REPORT.md):

  created          during a readout, mostly while reading |1>; the bit of that readout is still right
  while leaked     not touched by the conditional-X reset; no coupling to its data qubits, so its next readouts carry no
                   stabilizer information ("blind"); each two-qubit gate gives the partner a random Pauli
  what it reads    |1> (97-99% measured), the same value as the readout that created it, so its detectors stay silent.
                   This is modelled exactly with measurement feedback: the blind readout copies the previous outcome of
                   that ancilla (CX rec[-k] q). An episode created while reading |0> also has one flipped readout.
  ends             decays to |1> during a readout, which is then read and reset correctly

A history is fixed per pattern, and each pattern gets its own sampler and decoders, as in system/sweep_soft.py but with many
more patterns. Shot-to-shot variation of the history is covered by averaging over patterns.

Assumed rather than measured, so swept: the Pauli on the partner (p_ld), the reduction unit (eta, e_lru, t_lru), no transport
of leakage to data qubits, and half of the readouts returning |1>.
"""
import dataclasses
import sys

import numpy as np
import stim


@dataclasses.dataclass
class LeakModel:
    p_create: float            # P(ancilla leaves a readout in L | it was not leaked), averaged over the two outcomes
    survive: float             # P(a leaked ancilla is still leaked after the next round)
    frac_from_0: float = 0.07  # share of episodes created while reading |0>
    p_ld: float = 0.75         # DEPOLARIZE1 strength on the partner of a blocked gate; 0.75 = uniformly random Pauli
    tpr: float = 1.0           # P(flag | leaves the readout in L)
    fpr: float = 0.0           # P(flag | not in L)
    precision: float = 1.0     # P(in L | flag), what the decoder is told
    lru: str = "none"          # none, flag or always
    eta: float = 0.99          # P(the reduction unit removes leakage)
    e_lru: float = 0.002       # error of a reduction unit on a qubit that is not leaked

    @property
    def occupancy(self):
        """Stationary share of rounds an ancilla is leaked, without reduction."""
        return self.p_create / (1 - self.survive + self.p_create)


class Layout:
    """A lattice-sim circuit parsed once into operations tagged with their round."""

    def __init__(self, circuit):
        self.ops = []                  # (name, args, targets, round); targets are stim targets or qubit ints
        self.ancillas = {}             # round -> list of ancillas measured with MR in that round
        self.read_errors = []          # per measurement: the readout error of the original circuit
        r, in_mr = 0, False
        flat = list(circuit.flattened())
        pending = 0.0
        for k, inst in enumerate(flat):
            name = inst.name
            if name == "X_ERROR" and k + 1 < len(flat) and flat[k + 1].name in ("M", "MR") \
                    and inst.targets_copy() == flat[k + 1].targets_copy():
                pending = inst.gate_args_copy()[0]     # the readout error: position, not value, identifies it
                continue
            if name == "QUBIT_COORDS":
                continue
            if name == "TICK":
                if in_mr:
                    r += 1
                    in_mr = False
                self.ops.append(("TICK", (), [], r))
                continue
            args = tuple(inst.gate_args_copy())
            if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                self.ops.append((name, args, inst.targets_copy(), r))
                continue
            tg = [t.value for t in inst.targets_copy()]
            if name == "MR":
                in_mr = True
                self.ancillas.setdefault(r, []).extend(tg)
            if name in ("M", "MR"):
                self.read_errors += [pending] * len(tg)
                pending = 0.0
            self.ops.append((name, args, tg, r))
        self.rounds = r + 1
        self.mr_rounds = sorted(self.ancillas)
        self.qubit_coords = circuit.get_final_qubit_coordinates()
        self.coord_ops = [i for i in circuit.flattened() if i.name == "QUBIT_COORDS"]
        all_ancillas = sorted({q for v in self.ancillas.values() for q in v})
        self.all_ancillas = np.array(all_ancillas, dtype=np.int64)
        self.index = {q: i for i, q in enumerate(all_ancillas)}


def draw_history(layout, model, rng):
    """Returns (leaked, created_from_0, flags).

    leaked[r]         ancillas that are leaked during round r (blocked gates, blind readout)
    from0             set of (r, q): first round of an episode created while reading |0>
    flags[r]          ancillas flagged by the readout of round r-1, so known before round r
    lru_ok[r]         ancillas on which a reduction unit ran before round r
    """
    state = {}                         # ancilla -> leaked now
    leaked, flags, lru_ran = {}, {}, {}
    from0 = set()
    for r in layout.mr_rounds:
        next_r = r + 1
        leaked_next, flagged, ran = set(), set(), set()
        for q in layout.ancillas[r]:
            was = state.get(q, False)
            new_episode = False
            if was and rng.random() < model.survive:
                now = True
            else:
                now = rng.random() < model.p_create
                new_episode = now
            if now:
                fl = rng.random() < model.tpr
            else:
                fl = rng.random() < model.fpr
            if fl:
                flagged.add(q)
            run_lru = model.lru == "always" or (model.lru == "flag" and fl)
            if run_lru:
                ran.add(q)
                if now and rng.random() < model.eta:
                    now, new_episode = False, False
            if new_episode and rng.random() < model.frac_from_0:
                from0.add((next_r, q))
            state[q] = now
            if now:
                leaked_next.add(q)
        leaked[next_r], flags[next_r], lru_ran[next_r] = leaked_next, flagged, ran
    return dict(leaked=leaked, from0=from0, flags=flags, lru_ran=lru_ran)


def _append(c, name, tg, args=()):
    c.append(name, tg, list(args) if args else None)


def truth_circuit(layout, hist, model, soft=None):
    """The circuit that is sampled. soft: optional dict {(measurement index): error} replacing the readout error."""
    leaked, from0, lru_ran = hist["leaked"], hist["from0"], hist["lru_ran"]
    return _rewrite(layout, leaked, from0, lru_ran, model, soft, blind=True)


def belief_circuit(layout, hist, model, soft=None, mode="flag"):
    """The circuit the decoder's weights come from.

    mode='base'        no leakage knowledge
    mode='uniform'     every ancilla inflated by the stationary occupancy (a calibrated decoder without flags)
    mode='flag'        flagged ancillas inflated by the flag's precision
    mode='oracle'      the true leakage history, with certainty

    The CX gates stay in the decoder's circuit: a false flag must not remove edges from the matching graph.
    """
    if mode == "base":
        weights = None
    elif mode == "uniform":
        occ = model.occupancy
        weights = {r: {q: occ for q in layout.ancillas.get(r, [])} for r in layout.mr_rounds}
    elif mode == "flag":
        weights = {r: {q: model.precision for q in s} for r, s in hist["flags"].items()}
    elif mode == "oracle":
        weights = {r: {q: 1.0 for q in s} for r, s in hist["leaked"].items()}
    else:
        raise ValueError(mode)
    if model.lru != "none" and mode == "flag":
        # after a reduction unit, a flagged ancilla is leaked only if the unit failed
        weights = {r: {q: w * (1 - model.eta) for q, w in d.items()} for r, d in weights.items()}
    return _rewrite(layout, {}, set(), hist["lru_ran"], model, soft, blind=False, weights=weights)


def _rewrite(layout, leaked, from0, lru_ran, model, soft, blind, weights=None):
    c = stim.Circuit()
    last_meas = {}
    n_meas = 0
    done_lru = set()
    for name, args, tg, r in layout.ops:
        L = leaked.get(r, ()) if blind else ()
        W = weights.get(r, {}) if weights else {}
        if name == "TICK":
            c.append("TICK")
            continue
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            c.append(name, tg, list(args))
            continue
        if r in lru_ran and r not in done_lru and name != "X_ERROR":
            # reduction units run between the readout of round r-1 and the gates of round r
            done_lru.add(r)
            on = sorted(lru_ran[r])
            if on and model.e_lru > 0:
                c.append("DEPOLARIZE1", on, min(0.75, model.e_lru))
        if name == "CX":
            keep, depol, infl = [], [], []
            for a, b in zip(tg[0::2], tg[1::2]):
                if a in L or b in L:
                    for x, other in ((a, b), (b, a)):
                        if x not in L and other in L:
                            depol.append(x)
                else:
                    keep += [a, b]
                    for x, other in ((a, b), (b, a)):
                        w = W.get(other)
                        if w:
                            infl.append((x, w))
            if keep:
                c.append("CX", keep)
            if depol and model.p_ld > 0:
                c.append("DEPOLARIZE1", depol, model.p_ld)
            for x, w in infl:
                c.append("DEPOLARIZE1", [x], min(0.75, w * model.p_ld))
            continue
        if name == "DEPOLARIZE2":
            pairs = [x for a, b in zip(tg[0::2], tg[1::2]) if a not in L and b not in L for x in (a, b)]
            if pairs:
                c.append(name, pairs, list(args))
            continue
        if name == "H" or name == "DEPOLARIZE1":
            keep = [q for q in tg if q not in L]
            if keep:
                c.append(name, keep, list(args))
            continue
        if name in ("M", "MR"):
            for q in tg:
                if q in L:
                    c.append("CX", [stim.target_rec(last_meas[q] - n_meas), q])
                    if (r, q) in from0:
                        c.append("X_ERROR", [q], 1.0)
                e = soft[n_meas] if soft is not None else layout.read_errors[n_meas]
                w = W.get(q)
                if w:                                  # a readout that is blind with probability w
                    e = e * (1 - w) + 0.5 * w
                if e > 0:
                    c.append("X_ERROR", [q], min(0.5, e))
                c.append(name, [q])
                last_meas[q] = n_meas
                n_meas += 1
            continue
        c.append(name, tg, list(args) if args else None)
    return c


def soft_table(table, bits):
    """Merge a fine |logit| table into 2**bits levels with thresholds chosen for equal |logit| width up to the 99.9th
    percentile of the distribution. Returns [(fraction, error)], least confident first.

    Hardware would quantize |logit| the same way; the error of each level is what the decoder is told."""
    fr, er = np.array(table["frac"]), np.array(table["err"])
    edges = np.array(table["abs_logit_edges"])
    if bits == 0:
        return [(1.0, float((fr * er).sum()))]
    cum = np.cumsum(fr)
    top = edges[np.searchsorted(cum, 0.999) + 1]
    levels = np.linspace(0, top, 2 ** bits + 1)
    levels[-1] = np.inf
    out = []
    for lo, hi in zip(levels[:-1], levels[1:]):
        m = (edges[:-1] >= lo) & (edges[:-1] < hi)
        f = fr[m].sum()
        out.append((float(f), float((fr[m] * er[m]).sum() / f) if f > 0 else 0.0))
    return [x for x in out if x[0] > 0]


def flag_fraction_table(table, f):
    """The 1-bit flag of system/sweep_soft.py on the fine table: least confident fraction f vs the rest.

    Returns [(fraction, error)] for the two levels; the flagged level uses its measured error."""
    fr, er = np.array(table["frac"]), np.array(table["err"])
    cum = np.cumsum(fr)
    k = int(np.searchsorted(cum, f)) + 1
    f1, f2 = fr[:k].sum(), fr[k:].sum()
    return [(float(f1), float((fr[:k] * er[:k]).sum() / f1)), (float(f2), float((fr[k:] * er[k:]).sum() / f2))]


def draw_soft(n_meas, levels, rng):
    """Per-measurement error rates for k-bit soft readout: a level is drawn independently per measurement."""
    p = np.array([x[0] for x in levels])
    e = np.array([x[1] for x in levels])
    return e[rng.choice(len(levels), size=n_meas, p=p / p.sum())]


if __name__ == "__main__":
    sys.exit("library module")
