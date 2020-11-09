import sympy as sp;


x = sp.Symbol('x')
y = 3*x**x

print(x, y)


#求导数
y1 = 3*x
f1 = sp.diff(y1)

print(f1)

y2 = 3*x**3+2*x**2+1
f2 = sp.diff(y2)

print(f2)


#求积分
F1 = sp.integrate(f1, x)
F2 = sp.integrate(f2, x)
print(F1, F2)


#求极限

L1 = sp.limit(y1, x, 0)
L2 = sp.limit(y2, x, 0)

print(L1, L2)