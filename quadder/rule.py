from .utility import organise

class QuadratureRule:
    """
    Get the points and weights to perform a quadrature rule of a particular type
    valid to a specific degree.

    General usage is to call the class as a function with the signature
        rule(degree: int, dimension: int)
        -> (points: np.array(shape=(n, dimension)),
            weights: np.array(shape=(n,)))
    where `n` is the number of points in the rule.

    Use `QuadratureRule.n_points(dimension, degree)` to find the number of
    points without calculating the points themselves.

    Use `QuadratureRule.allowed(dimension, degree)` to get a Boolean return of
    whether the pair are supported.

    If the `(dimension, degree)` pair is not valid for this quadrature rule, a
    `ValueError` will be raised when calling any calculation function.

    This class typically need not be instantiated---there are other pre-created
    quadrature rules in this package.
    """
    def __init__(self, type, allowed, n_points, rule):
        self.type = type
        self._allowed = allowed
        self._n_points = n_points
        self._rule = rule

    def allowed(self, dimension, degree):
        """
        Boolean return whether the specified `(dimension, degree)` pair is
        allowed for this quadrature rule.
        """
        try:
            self._allowed(dimension, degree)
            return True
        except ValueError:
            return False

    def n_points(self, dimension, degree):
        self._allowed(dimension, degree)
        return self._n_points(dimension, degree)

    def __call__(self, dimension, degree):
        self._allowed(dimension, degree)
        return organise(self._rule(dimension, degree))
