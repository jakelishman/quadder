import numpy as np
from .utility import plus_minus, rule_sets
from . import QuadratureRule

def _u_3_14_1(*_):
    # 14-1, p302, Stroud 1971
    r = np.sqrt(0.1 * (5 - np.sqrt(5)))
    s = np.sqrt(0.1 * (5 + np.sqrt(5)))
    V = 4*np.pi
    B = 125*V / 10080
    C = 143*V / 10080
    z = np.sqrt(np.polynomial.Polynomial((9, -3562, 115115, -1043900, 3578575,
                                          -5112250, 2556125)).roots())
    u = np.array((-z[2] + z[3],
                  -z[4] + z[1],
                  -z[1] + z[5],
                  -z[5] + z[2],
                  -z[3] + z[4])) / (2*s)
    v = np.array((z[4] + z[5],
                  z[5] + z[3],
                  z[2] + z[4],
                  z[3] + z[1],
                  z[1] + z[2])) / (2*s)
    w = (z[0] + z[1:]) / (2*s)
    return [(B, plus_minus((r, s, 0))),
            (B, plus_minus((0, r, s))),
            (B, plus_minus((s, 0, r))),
            (C, zip(u, v, w)),
            (C, zip(u, -v, -w)),
            (C, zip(-u, -v, w)),
            (C, zip(-u, v, -w)),
            (C, zip(v, w, u)),
            (C, zip(v, -w, -u)),
            (C, zip(-v, -w, u)),
            (C, zip(-v, w, -u)),
            (C, zip(w, u, v)),
            (C, zip(w, -u, -v)),
            (C, zip(-w, -u, v)),
            (C, zip(-w, u, -v))]
u_3_14_1 = QuadratureRule('spherical-surface',
                          rule_sets({3}, {14}),
                          lambda *_: 72,
                          _u_3_14_1)
