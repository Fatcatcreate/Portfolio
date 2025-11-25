#!/usr/bin/env python3
"""
code/Euler248/code.py

Project Euler 248 — find the 150,000th number n such that φ(n) = 13!
"""

import math
from collections import defaultdict

def Divisors(n):
    divs, root = [], int(math.isqrt(n))
    for i in range(1, root + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

def IsPrime(n):
    if n < 2: return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0: return False
    return True

def Valuation(n, p):
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v

def InverseEulerPhi(n):
    r = defaultdict(list)
    r[1] = [1]
    for d in Divisors(n):
        if IsPrime(d + 1):
            temp = defaultdict(list)
            max_k = Valuation(n, d + 1) + 1
            for k in range(1, max_k + 1):
                u = d * (d + 1) ** (k - 1)
                v = (d + 1) ** k
                for f in Divisors(n // u):
                    if f in r:
                        temp[f * u].extend([v * x for x in r[f]])
            for i, vlist in temp.items():
                r[i].extend(vlist)
    return sorted(r[n]) if n in r else []

if __name__ == "__main__":
    FACT = math.factorial(13)
    TARGET_INDEX = 150_000 - 1
    numbers = InverseEulerPhi(FACT)
    print(numbers[TARGET_INDEX])
