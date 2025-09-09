#!/usr/bin/env python3

L=10**6

def primes():
    yield 2
    ps=[2];c=3
    while 1:
        ok=1;r=int(c**0.5)
        for p in ps:
            if p>r:break
            if c%p==0:ok=0;break
        if ok:ps.append(c);yield c
        c+=2

def bestN(lim):
    n=1;ps=[]
    for p in primes():
        if n*p>lim:break
        n*=p;ps.append(p)
    return n,ps

def phi(n,ps):
    f=n
    for p in ps:f=(f//p)*(p-1)
    return f

def all():
    n,ps=bestN(L)
    f=phi(n,ps)
    print(n)
    print("primes:",ps)
    print("phi:",f)
    print("ratio:",n/f)

all()
