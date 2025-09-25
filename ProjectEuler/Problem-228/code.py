#!/usr/bin/env python3
from fractions import Fraction

def count_dirs(a, b):
    s = set()
    for n in range(a, b+1):
        for k in range(n):
            s.add(Fraction(2*k, n))
    return len(s)

def main():
    print(count_dirs(1864, 1909))

if __name__ == "__main__":
    main()
