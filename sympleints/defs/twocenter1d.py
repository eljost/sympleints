from sympy import exp


class TwoCenter1d:
    def __init__(self, a, A, b, B, C=None):
        self.a = a
        self.A = A
        self.b = b
        self.B = B
        self.C = C

    @property
    def p(self):
        """Total exponent p."""
        return self.a + self.b

    @property
    def mu(self):
        """Reduced exponent mu."""
        return self.a * self.b / self.p

    @property
    def AB(self):
        """Relative coordinate/Gaussian separation X_AB."""
        return self.A - self.B

    @property
    def P(self):
        """Center-of-charge coordinate."""
        return (self.a * self.A + self.b * self.B) / self.p

    @property
    def PA(self):
        """Relative coordinate/Gaussian separation X_PA."""
        return self.P - self.A

    @property
    def PB(self):
        """Relative coordinate/Gaussian separation X_PB."""
        return self.P - self.B

    @property
    def PC(self):
        """Relative coordinate/Gaussian separation X_PC."""
        return self.P - self.C

    @property
    def K(self):
        return exp(-self.mu * self.AB**2)
