from sympy import exp


class TwoCenter1d:
    def __init__(self, ax, A, bx, B, R=None):
        self.ax = ax
        self.A = A
        self.bx = bx
        self.B = B
        self.R = R

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
    def PA(self):
        """Relative coordinate/Gaussian separation X_PA."""
        return self.P - self.A

    @property
    def PB(self):
        """Relative coordinate/Gaussian separation X_PB."""
        return self.P - self.B

    @property
    def PR(self):
        """Relative coordinate/Gaussian separation X_PR."""
        return self.P - self.R

    @property
    def K(self):
        return exp(-self.mu * self.AB**2)
