"""
Project Euler Problem 180: Golden Triplets
Complexity: O(k⁴) time, O(k²) space
"""

from fractions import Fraction
from math import gcd

def GenerateReducedRationals(MaxDenominator):
    Rationals = set()
    for Denominator in range(2, MaxDenominator + 1):
        for Numerator in range(1, Denominator):
            if gcd(Numerator, Denominator) == 1:
                Rationals.add(Fraction(Numerator, Denominator))
    return Rationals

def IsRationalSquare(Value):
    Numerator = Value.numerator
    Denominator = Value.denominator
    NumSqrt = int(Numerator ** 0.5)
    DenSqrt = int(Denominator ** 0.5)
    if NumSqrt * NumSqrt == Numerator and DenSqrt * DenSqrt == Denominator:
        return Fraction(NumSqrt, DenSqrt)
    return None


def GoldenTriples(Order):
    Rationals = GenerateReducedRationals(Order)
    GoldenTriples = set()
    for X in Rationals:
        for Y in Rationals:
            # Case 1: z = x + y
            Z = X + Y
            if Z in Rationals:
                GoldenTriples.add(tuple(sorted([X, Y, Z])))
            # Case 2: z = sqrt(x² + y²)
            SumOfSquares = X * X + Y * Y
            SqrtValue = IsRationalSquare(SumOfSquares)
            if SqrtValue and SqrtValue in Rationals:
                GoldenTriples.add(tuple(sorted([X, Y, SqrtValue])))
            # Case 3: z = xy/(x + y)
            if X + Y != 0:
                Z = (X * Y) / (X + Y)
                if Z in Rationals:
                    GoldenTriples.add(tuple(sorted([X, Y, Z])))
            # Case 4: z = xy/sqrt(x² + y²)
            if SqrtValue and SqrtValue != 0:
                Z = (X * Y) / SqrtValue
                if Z in Rationals:
                    GoldenTriples.add(tuple(sorted([X, Y, Z])))
    return GoldenTriples


def Compute(Order):
    GoldenTriples = GoldenTriples(Order)
    SumValues = {X + Y + Z for X, Y, Z in GoldenTriples}
    TotalSum = sum(SumValues)
    return TotalSum.numerator + TotalSum.denominator


Result = Compute(35)
print(Result)