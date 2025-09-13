#!/usr/bin/env python3
MAX = 1000

def Solve(M):
    N = M + 1
    size = N * N
    Won = False
    Lost = True
    One = [Won] * size
    Two = [Won] * size
    All = [Won] * size

    def Id(a, b): return a * N + b
    Count = 0
    for x in range(0, M + 1):
        for y in range(x, M + 1):
            if One[Id(x, y)] == Lost:
                continue
            for z in range(y, M + 1):
                if One[Id(y, z)] == Lost or One[Id(x, z)] == Lost or One[Id(x, y)] == Lost:
                    continue
                if Two[Id(y - x, z)] == Lost or Two[Id(z - y, x)] == Lost or Two[Id(z - x, y)] == Lost:
                    continue
                if All[Id(y - x, z - x)] == Lost:
                    continue
                Count += x + y + z
                One[Id(y, z)] = Lost
                One[Id(x, z)] = Lost
                One[Id(x, y)] = Lost
                Two[Id(y - x, z)] = Lost
                Two[Id(z - y, x)] = Lost
                Two[Id(z - x, y)] = Lost
                All[Id(y - x, z - x)] = Lost
    return Count


print(Solve(MAX))
