# e5vrepro: independent demodulation of raw TRAIN/TEST traces (own IF estimate, own binning).
# v2: global DC offset removed before demodulation (v1 IF refinement for q1 locked onto DC).
# Output: complex64 memmaps (N,5,100) of 20-ns boxcar bins of the demodulated signal per qubit tone,
# plus y arrays and the IF json. No shared features or other agents' code are used.
import os as _os, sys as _sys  # Oraqle unify: path plumbing (lkpaths.py)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402
import h5py, numpy as np, json, os, sys, time
from multiprocessing import Pool
FS = 500e6; NS = 1000; BIN = 10; NB = NS // BIN
OUT = LP.outdir(LP.stage('em/e5vrepro') / 'out')
FILES = {'train': (str(LP.RAW_TRAIN), 'X_train', 'y_train'),
         'test': (str(LP.RAW_TEST), 'X_test', 'y_test')}
t = np.arange(NS) / FS

def strat_sample(path, xk, yk, nblk=120, blk=1000):
    h = h5py.File(path, 'r'); N = h[xk].shape[0]
    starts = np.linspace(0, N - blk, nblk).astype(int)
    xs, ys = [], []
    for s0 in starts:
        xs.append(h[xk][s0:s0 + blk].astype(np.float32)); ys.append(h[yk][s0:s0 + blk])
    x = np.concatenate(xs); y = np.concatenate(ys)
    return x[..., 0] + 1j * x[..., 1], y

def estimate_if():
    s, y = strat_sample(*FILES['train'])
    dc = complex(s.mean())
    s = s - dc
    print('IF sample', s.shape, 'states present', len(np.unique(y)), 'DC', dc, flush=True)
    NF = 1 << 18; F = np.fft.fftfreq(NF, 1 / FS)
    res = {}
    for q in range(5):
        b = (y >> q) & 1
        D = s[b == 1].mean(0) - s[b == 0].mean(0)
        f0 = F[np.argmax(np.abs(np.fft.fft(D, NF)))]
        r = dict(f_coarse=float(f0), dc_re=dc.real, dc_im=dc.imag)
        # refine: phase slope of smoothed demodulated prepared-|0> (and |1>) mean, samples 200-999
        f = f0
        for it in range(4):
            sl = {}
            for bb in (0, 1):
                z = (s[b == bb] * np.exp(-2j * np.pi * f * t)).mean(0)
                k = np.ones(50) / 50
                zs = np.convolve(z, k, mode='same')
                tt = np.arange(200, 950)
                ph = np.unwrap(np.angle(zs[tt]))
                sl[bb] = np.polyfit(tt / FS, ph, 1)[0] / (2 * np.pi)
            f = f + sl[0]
            r['iter%d' % it] = dict(slope0_Hz=float(sl[0]), slope1_Hz=float(sl[1]))
        r['f_if'] = float(f)
        res[q] = r
        print('q', q, json.dumps(r), flush=True)
    return res

def work(args):
    split, s0, s1, fifs, dc = args
    path, xk, yk = FILES[split]
    h = h5py.File(path, 'r')
    x = h[xk][s0:s1].astype(np.float32)
    s = (x[..., 0] - dc.real) + 1j * (x[..., 1] - dc.imag)
    out = np.empty((s1 - s0, 5, NB), np.complex64)
    for q, f in enumerate(fifs):
        ph = np.exp(-2j * np.pi * f * t).astype(np.complex64)
        out[:, q, :] = (s * ph).reshape(s1 - s0, NB, BIN).mean(-1)
    mm = np.load(os.path.join(OUT, 'em_e5vrepro_bins2_%s.npy' % split), mmap_mode='r+')
    mm[s0:s1] = out; mm.flush(); del mm
    return s0

if __name__ == '__main__':
    t0 = time.time()
    ifp = os.path.join(OUT, 'em_e5vrepro_if2.json')
    if not os.path.exists(ifp):
        res = estimate_if(); json.dump(res, open(ifp, 'w'), indent=1)
    res = json.load(open(ifp)); fifs = [res[str(q)]['f_if'] for q in range(5)]
    dc = complex(res['0']['dc_re'], res['0']['dc_im'])
    print('IFs (MHz):', [round(f / 1e6, 5) for f in fifs], 'elapsed', time.time() - t0, flush=True)
    for split in ('train', 'test'):
        path, xk, yk = FILES[split]
        h = h5py.File(path, 'r'); N = h[xk].shape[0]; y = h[yk][:]; h.close()
        np.save(os.path.join(OUT, 'em_e5vrepro_y_%s.npy' % split), y)
        fn = os.path.join(OUT, 'em_e5vrepro_bins2_%s.npy' % split)
        mm = np.lib.format.open_memmap(fn, mode='w+', dtype=np.complex64, shape=(N, 5, NB)); del mm
        CH = 10000
        jobs = [(split, a, min(a + CH, N), fifs, dc) for a in range(0, N, CH)]
        with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as p:
            for i, r in enumerate(p.imap_unordered(work, jobs)):
                if i % 20 == 0: print(split, i, len(jobs), 'elapsed', round(time.time() - t0), flush=True)
        print(split, 'done', time.time() - t0, flush=True)
