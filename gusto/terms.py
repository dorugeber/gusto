from abc import ABCMeta, abstractmethod
from firedrake import FacetNormal, div, dx, inner


class Term(object, metaclass=ABCMeta):

    off_centering = 0.5

    def __init__(self, state):
        self.state = state
        self.parameters = state.parameters
        self.n = FacetNormal(state.mesh)

    @abstractmethod
    def evaluate(self, test, q, fields):
        pass

    def __call__(self, test, q, fields):
        return self.evaluate(test, q, fields)


class ShallowWaterPressureGradientTerm(Term):

    def evaluate(self, test, q, fields):
        g = self.parameters.g
        D = fields("D")
        return g*div(test)*D*dx


class ShallowWaterCoriolisTerm(Term):

    def evaluate(self, test, q, fields):
        f = self.parameters.coriolis
        return -f*inner(test, self.state.perp(q))*dx


class ShallowWaterTopographyTerm(Term):

    def evaluate(self, test, q, fields):
        g = self.parameters.g
        b = self.parameters("topography")
        return g*div(test)*b*dx
