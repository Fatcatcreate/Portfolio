#!/usr/bin/env python3
import math

def ComputeCutoffLambdaForNgon(n):
    """Return cutoff lambda for regular n-gon using the theorem in the notes."""
    theta = math.pi / n
    def f(k):
        return math.sin(k*theta) - (k + n) * math.tan(theta) * math.cos(k*theta)
    K = 0
    for k in range(0, n+1):
        if f(k) < 0:
            K = k
    denom = (K + n) * math.tan(theta)
    R = 2.0 * math.sin(K*theta) / denom if denom != 0 else 0.0
    inside = R - math.cos(K*theta)
    # numerical safety
    if inside > 1.0: inside = 1.0
    if inside < -1.0: inside = -1.0
    alpha = 0.5 * (K*theta + math.acos(inside))
    lam = 1.0 / math.cos(alpha)
    return lam

lam = ComputeCutoffLambdaForNgon(6)
print(f"{lam:.8f}")
