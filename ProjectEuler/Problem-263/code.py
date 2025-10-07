#!/usr/bin/env python3
"""
code/Euler263/code.py

Project Euler 263 — find the sum of the first four "engineers' paradises".
"""

import random
import math
def IsPrime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    def Check(a, s, d, n):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False
    for a in bases:
        if a % n == 0:
            continue
        if not Check(a, s, d, n):
            return False
    return True

# Pollard-Rho factorisation
def PollardRho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randrange(2, n - 1)
        y = x
        c = random.randrange(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
            if d == n:
                break
        if 1 < d < n:
            return d

def Factorize(n: int) -> dict:
    if n == 1:
        return {}
    if IsPrime(n):
        return {n: 1}
    factors = {}
    # trial divide small primes
    for p in (2,3,5,7,11,13,17,19,23,29):
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n == 1:
        return factors
    if IsPrime(n):
        factors[n] = factors.get(n, 0) + 1
        return factors
    d = PollardRho(n)
    if d is None:
        factors[n] = factors.get(n,0) + 1
        return factors
    left = Factorize(d)
    right = Factorize(n // d)
    for k, v in left.items():
        factors[k] = factors.get(k, 0) + v
    for k, v in right.items():
        factors[k] = factors.get(k, 0) + v
    return factors

# Stewart–Sierpiński criterion
def SigmaPrimePower(p: int, a: int) -> int:
    return (p ** (a + 1) - 1) // (p - 1)

def IsPractical(n: int) -> bool:
    if n == 1:
        return True
    fac = Factorize(n)
    primes_sorted = sorted(fac.items(), key=lambda x: x[0]) 
    if primes_sorted[0][0] != 2:
        return False
    running_sigma = SigmaPrimePower(primes_sorted[0][0], primes_sorted[0][1])
    for p, e in primes_sorted[1:]:
        if p > running_sigma + 1:
            return False
        running_sigma *= SigmaPrimePower(p, e)
    return True

def NextPrimeAbove(n: int) -> int:
    if n < 2:
        return 2
    m = n + 1
    if m <= 2:
        return 2
    if m % 2 == 0:
        if m == 2:
            return 2
        m += 1
    while not IsPrime(m):
        m += 2
    return m

def IsConsecutiveSexyPair(a: int, b: int) -> bool:
    if b - a != 6:
        return False
    if not (IsPrime(a) and IsPrime(b)):
        return False
    return NextPrimeAbove(a) == b

def FindEngineersParadises(required: int = 4):
    found = []
    k = 0
    while len(found) < required:
        for offset in (20, 820):  
            n = 840 * k + offset
            if n <= 9:
                continue
            if not (IsConsecutiveSexyPair(n - 9, n - 3) and
                    IsConsecutiveSexyPair(n - 3, n + 3) and
                    IsConsecutiveSexyPair(n + 3, n + 9)):
                continue
            ok = True
            for off in (-8, -4, 0, 4, 8):
                if not IsPractical(n + off):
                    ok = False
                    break
            if ok:
                found.append(n)
                if len(found) >= required:
                    break
        k += 1
    return found


paradises = FindEngineersParadises(4)
print(paradises)
print(sum(paradises))

