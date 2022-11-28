from sympy import exp


class FourCenter1d:
    def __init__(self, ax, A, bx, B, cx=None, C=None, dx=None, D=None, R=None):
        self.ax = ax
        self.A = A
        self.bx = bx
        self.B = B
        self.cx = cx
        self.C = C
        self.dx = dx
        self.D = D
        self.R = R

    @property
    def p(self):
        """Total exponent p."""
        return self.ax + self.bx

    @property
    def q(self):
        """Total exponent q."""
        return self.cx + self.dx

    @property
    def mu(self):
        """Reduced exponent mu."""
        return self.ax * self.bx / self.p

    @property
    def nu(self):
        """Reduced exponent mu."""
        return self.cx * self.dx / self.q

    @property
    def AB(self):
        """Relative coordinate/Gaussian separation X_AB."""
        return self.A - self.B

    @property
    def CD(self):
        """Relative coordinate/Gaussian separation X_AB."""
        return self.C - self.D

    @property
    def P(self):
        """Center-of-charge coordinate for AB."""
        return (self.ax * self.A + self.bx * self.B) / self.p

    @property
    def Q(self):
        """Center-of-charge coordinate for CD."""
        return (self.cx * self.C + self.dx * self.D) / self.q

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
    def QC(self):
        """Relative coordinate/Gaussian separation X_QC."""
        return self.Q - self.C

    @property
    def QD(self):
        """Relative coordinate/Gaussian separation X_QD."""
        return self.Q - self.D

    @property
    def QR(self):
        """Relative coordinate/Gaussian separation X_QR."""
        return self.Q - self.R

    @property
    def K(self):
        return exp(-self.mu * self.AB**2)

    @property
    def L(self):
        return exp(-self.nu * self.CD**2)
