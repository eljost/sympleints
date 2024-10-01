import sympy as sym


class TwoCenter3dFunc:
    def __init__(self, ax, A, bx, B, R, r):
        self.ax = ax
        self.A = A
        self.bx = bx
        self.B = B
        self.R = sym.Matrix(R).reshape(1, 3)
        self.r = r

    @property
    def p(self):
        """Total exponent p."""
        return self.ax + self.bx

    @property
    def mu(self):
        """Reduced exponent mu."""
        return self.ax * self.bx / self.p

    @property
    def AB(self):
        """Relative coordinate/Gaussian separation X_AB."""
        return self.A - self.B

    @property
    def P(self):
        """Center-of-charge coordinate."""
        return (self.ax * self.A + self.bx * self.B) / self.p

    @property
    def rA(self):
        return self.r - self.A

    @property
    def rB(self):
        return self.r - self.B

    @property
    def rR(self):
        return self.r - self.R

    def prefact(self, coords, i, j, k):
        return coords[0] ** i * coords[1] ** j * coords[2] ** k

    def prefact_rA(self, i, j, k):
        return self.prefact(self.rA, i, j, k)

    def prefact_rB(self, i, j, k):
        return self.prefact(self.rB, i, j, k)

    @property
    def K(self):
        return sym.exp(-self.mu * self.AB.dot(self.AB))

    @property
    def Pr(self):
        """Relative coordinate/Gaussian separation X_Pr."""
        return self.P - self.r

    @property
    def product(self):
        return self.K * sym.exp(-self.p * self.Pr.dot(self.Pr))

    def full_product(self, i, k, m, j, l, n):
        return self.prefact_rA(i, k, m) * self.prefact_rB(j, l, n) * self.product
