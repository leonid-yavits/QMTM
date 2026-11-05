"""Thin wrapper around the patched pyTsetlinMachine (non-indexed path).
Select the library with env QMTM_LIB (default: patched build)."""
import os, sys, ctypes as C, numpy as np

LIB = os.environ.get("QMTM_LIB", "/home/claude/qmtm/qm")
sys.path.insert(0, LIB)
import pyTsetlinMachine.tm as _tm  # noqa: E402

_lib = _tm._lib
_lib.qm_seed.restype = None
_lib.qm_seed.argtypes = [C.c_ulonglong]
HAS_QM = hasattr(_lib, "mc_tm_set_qm")
if HAS_QM:
    _lib.mc_tm_set_qm.restype = None
    _lib.mc_tm_set_qm.argtypes = [_tm.mc_ctm_pointer, C.c_int, C.c_int, C.c_int]
HAS_FB = hasattr(_lib, "mc_tm_set_qm_fb")
if HAS_FB:
    _lib.mc_tm_set_qm_fb.restype = None
    _lib.mc_tm_set_qm_fb.argtypes = [_tm.mc_ctm_pointer, C.c_int]


HAS_FAST = hasattr(_lib, "mc_tm_set_fast")
if HAS_FAST:
    _lib.mc_tm_set_fast.restype = None
    _lib.mc_tm_set_fast.argtypes = [_tm.mc_ctm_pointer, C.c_int]


def seed(s):
    _lib.qm_seed(int(s))


class QMTM(_tm.MultiClassTsetlinMachine):
    """mode 0 = baseline, 1 = relative QM (theta_pm per mille of included
    literals), 2 = absolute QM (abs_tol violations allowed per clause)."""

    def __init__(self, clauses, T, s, mode=0, theta_pm=1000, abs_tol=0, fb=0, fast=0, **kw):
        super().__init__(clauses, T, s, indexed=False, **kw)
        self.qm = (mode, theta_pm, abs_tol)
        self.fb = fb
        self.fast = fast

    def init_machine(self, X, Y):
        # Allocates the machine without training. Only one row (carrying the largest label) is
        # encoded here, so the class count and feature count match a full fit(epochs=0).
        i = int(np.argmax(Y))
        super().fit(X[i:i + 1], Y[i:i + 1], epochs=0)
        self.set_qm(*self.qm)
        if self.fb:
            if not HAS_FB:
                raise RuntimeError("library has no QM feedback support")
            _lib.mc_tm_set_qm_fb(self.mc_tm, int(self.fb))
        if self.fast:
            if not HAS_FAST:
                raise RuntimeError("library has no fast-stream support")
            _lib.mc_tm_set_fast(self.mc_tm, int(self.fast))

    def set_qm(self, mode, theta_pm=1000, abs_tol=0):
        self.qm = (mode, theta_pm, abs_tol)
        if HAS_QM:
            _lib.mc_tm_set_qm(self.mc_tm, int(mode), int(theta_pm), int(abs_tol))
        elif mode != 0:
            raise RuntimeError("library has no QM support")

    def train_epoch(self, X, Y):
        super().fit(X, Y, epochs=1, incremental=True)

    # Memory-lean path: pack once with numpy (bit-identical to tm_encode with append_negated=1),
    # then train/predict on the packed words. Avoids the n x M uint32 copy made by the library.
    def encode(self, X, batch=2048):
        n, M = X.shape
        assert 2 * M == self.number_of_features
        W = self.number_of_ta_chunks
        out = np.zeros((n, W), dtype=np.uint32)
        for a in range(0, n, batch):
            x = X[a:a + batch].astype(bool)
            lit = np.concatenate([x, ~x], axis=1)                       # literal k<M: x_k ; k>=M: not x_k
            pad = np.zeros((lit.shape[0], 32 * W - 2 * M), dtype=bool)
            b = np.packbits(np.concatenate([lit, pad], axis=1), axis=1, bitorder="little")
            out[a:a + batch] = b.view("<u4")
        return np.ascontiguousarray(out.reshape(-1))

    def train_epoch_encoded(self, E, Y, epochs=1):
        _tm._lib.mc_tm_fit(self.mc_tm, E, np.ascontiguousarray(Y).astype(np.uint32), len(Y), epochs)

    def predict_encoded(self, E):
        n = E.size // self.number_of_ta_chunks
        Yp = np.ascontiguousarray(np.zeros(n, dtype=np.uint32))
        _tm._lib.mc_tm_predict(self.mc_tm, E, Yp, n)
        return Yp

    def include_sets(self):
        """Included literals as COO arrays (class, clause, literal); literal < number_of_features//2 is positive."""
        nb, W, F = self.number_of_state_bits, self.number_of_ta_chunks, self.number_of_features
        cls, cl, lit = [], [], []
        for c in range(self.number_of_classes):
            _, ta = self.get_state()[c]
            msb = ta.reshape(self.number_of_clauses, W, nb)[:, :, nb - 1].copy()
            bits = np.unpackbits(msb.view(np.uint8), axis=1, bitorder="little")[:, :F]
            j, i = np.nonzero(bits)
            cls.append(np.full(len(j), c, np.int32)); cl.append(j.astype(np.int32)); lit.append(i.astype(np.int32))
        return np.concatenate(cls), np.concatenate(cl), np.concatenate(lit)

    def include_counts(self):
        """Included-literal count per clause, shape (classes, clauses)."""
        out = np.zeros((self.number_of_classes, self.number_of_clauses), dtype=np.int32)
        nb = self.number_of_state_bits
        for c in range(self.number_of_classes):
            _, ta = self.get_state()[c]
            ta = ta.reshape(self.number_of_clauses, self.number_of_ta_chunks, nb)[:, :, nb - 1].copy()
            last = self.number_of_features % 32
            if last:
                ta[:, -1] &= np.uint32((1 << last) - 1)
            out[c] = np.unpackbits(ta.view(np.uint8), axis=1).sum(1)
        return out
