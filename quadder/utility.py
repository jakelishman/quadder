import itertools
import collections
import numpy as np

def symmetric(iterable):
    """
    Find all distrinct permutations of `iterable`, i.e. accounting for
    indistinguishable elements.  The aim was to make it lazily evaluated, except
    for two caveats:
        - the iterable must be consumed to find the indistinct elements
        - it uses `itertools.combinations`, which is not lazy
    """
    elements = tuple(collections.Counter(iterable).items())
    length = sum(map(lambda t: t[1], elements))
    def reduction(state, elements):
        if len(elements) is 0:
            return state
        value, count = elements[0]
        def fill(permutation):
            available = (i for i, x in enumerate(permutation) if x is None)
            def set(indices):
                out = permutation.copy()
                for index in indices:
                    out[index] = value
                return out
            return map(set, itertools.combinations(available, count))
        return reduction(itertools.chain.from_iterable(map(fill, state)),
                         elements[1:])
    return reduction([[None] * length], elements)

# For reference: the output of the below is essentially the same, except it runs
# in `n! lg(n!)` time, whereas the previous version is `n! / prod_i(m_i!)`,
# where `m_i` is how many times the `i`th distinct value occurs.
def symmetric_descriptive(iterable):
    prev = None
    for cur in sorted(itertools.permutations(iterable)):
        if cur != prev:
            yield cur
        prev = cur

def plus_minus(iterable):
    state = [((x,) if x == 0.0 else (x, -x)) for x in iterable]
    return itertools.product(*state)

def full_symmetric(iterable):
    state = [((x,) if x == 0.0 else (x, -x)) for x in iterable]
    return itertools.chain.from_iterable(itertools.starmap(itertools.product, symmetric(state)))

def organise(iterable):
    ordered = ((weight, tuple(points))\
               for weight, points in sorted(iterable, key=lambda t:t[0]))
    matched = (((weight,) * len(points), points)
               for weight, points in ordered)
    joined = map(lambda t: tuple(itertools.chain.from_iterable(t)),
                 zip(*matched))
    weights, points = map(np.array, joined)
    return points, weights

def rule_sets(dimensions, degrees):
    degrees = set(degrees)
    dimensions = set(dimensions)
    def allowed(dimension, degree):
        if degree not in degrees:
            raise ValueError(f"Degree must be in {degrees}.")
        if dimension not in dimensions:
            raise ValueError(f"Dimension must be in {dimensions}.")
    return allowed
