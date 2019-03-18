import sys
import inspect
import numpy as np
import itertools
import operator
from .utility import full_symmetric, symmetric, plus_minus, rule_sets
from . import spherical_surface, QuadratureRule

__all__ = ['plane_r2']

def _e_n_7_1(dimension, _):
    # 7-1, p318, Stroud 1971
    n = dimension
    r = np.sqrt((3*(8-n)+(n-2)*np.sqrt(3*(8-n))) / (2*(5-n)))
    s = np.sqrt((3*n+2*np.sqrt(3*(8-n)))/(2*(3*n-8)))
    t = np.sqrt((6-np.sqrt(3*(8-n)))/2)
    V = np.pi**(n/2)
    B = (8 - n)*V / (8 * r**6)
    C = V / (2**(n+3) * s**6)
    D = V / (16 * t**6)
    A = V - 2*n*B - (2**n * C) - 2*n*(n-1)*D
    return [(A, ((0.0,) * n,)),
            (B, full_symmetric((r,) + (0,)*(n-1))),
            (C, full_symmetric((s,)*n)),
            (D, full_symmetric((t, t) + (0,)*(n-2)))]
e_n_7_1 = QuadratureRule('plane_r2',
                         rule_sets({3, 4, 6, 7}, {7}),
                         lambda n, _: (1<<n) + ((n*n)<<1) + 1,
                         _e_n_7_1)

_e_n_9_1_B = {3: ( 0.67644_87344_29924,     0.51198_91062_91551e-2,
                   0.44859_57234_93744,     0.23522_34545_95606e-3,
                   0.91539_07130_80005e-1,  0.13920_81999_20793e-1,
                   0.23522_34545_95606e-3,  0.91539_07130_80008e-1),
              4: (-0.86045_29450_07048,    -0.40551_19985_33795e-1,
                   0.10702_64754_49715e1,   0.13897_42393_07092e-3,
                  -0.16224_87794_48181,     0.24674_01100_27234e-1,
                   0.13897_42393_07094e-3,  0.16224_87794_48181,
                   0.13897_42393_07094e-3),
              5: (-0.82734_70062_00826e1,  -0.16082_01745_30905,
                   0.35349_98637_58467e1,   0.73897_62769_09564e-3,
                  -0.86273_54218_12943,     0.43733_54581_90621e-1,
                  -0.24632_54256_36523e-3,  0.28757_84739_37648,
                   0.24632_54256_36523e-3),
              6: (-0.36184_04341_43098e2,  -0.44793_65291_38517,
                   0.11207_78630_04144e2,   0.39294_04043_20855e-2,
                  -0.25485_97867_84158e1,   0.77515_69170_07496e-1,
                  -0.13098_01347_73619e-2,  0.50971_95735_68315,
                   0.43660_04492_45395e-3)}
def _e_n_9_1(dimension, _):
    u = 2.02018_28704_5609
    v = 0.95857_24646_13819
    B = _e_n_9_1_B[dimension]
    items = [(B[0], ((0,) * dimension,)),
             (B[1], full_symmetric((u,) + (0,)*(dimension - 1))),
             (B[2], full_symmetric((v,) + (0,)*(dimension - 1))),
             (B[3], full_symmetric((u,u) + (0,)*(dimension - 2))),
             (B[4], full_symmetric((v,v) + (0,)*(dimension - 2))),
             (B[5], full_symmetric((u,v) + (0,)*(dimension - 2))),
             (B[6], full_symmetric((u,u,u) + (0,)*(dimension - 3))),
             (B[7], full_symmetric((v,v,v) + (0,)*(dimension - 3)))]
    if dimension > 3:
        items.append((B[8], full_symmetric((u,u,u,u) + (0,)*(dimension - 4))))
    return items
e_n_9_1 = QuadratureRule('plane_r2',
                         rule_sets({3, 4, 5, 6}, {9}),
                         lambda n, _: (2*n**4 - 4*n**3 + 22*n**2 - 8*n + 3)//3,
                         _e_n_9_1)


def _e_1(_, degree):
    # Standard plane_r2 quadrature for one dimension.
    points, weights = np.polynomial.hermite.hermgauss((degree + 1) // 2)
    return zip(weights, map(lambda t: ((t,),), points))
def _e_1_allowed(dimension, degree):
    if dimension is not 1:
        raise ValueError("Dimension must be 1.")
    if degree % 2 == 0:
        raise ValueError("Degree must be odd.")
e_1 = QuadratureRule('plane_r2',
                     _e_1_allowed,
                     lambda _, degree: (degree + 1) // 2,
                     _e_1)


def _e_2_7_1(*_):
    # p324, Stroud 1971
    r = np.sqrt(3)
    s = np.sqrt(0.125 * (9 - 3*np.sqrt(5)))
    t = np.sqrt(0.125 * (9 + 3*np.sqrt(5)))
    V = np.pi
    A = V / 36
    B = (5 + 2*np.sqrt(5))*V / 45
    C = (5 - 2*np.sqrt(5))*V / 45
    return [(A, full_symmetric((r, 0))),
            (B, plus_minus((s, s))),
            (C, plus_minus((t, t)))]
e_2_7_1 = QuadratureRule('plane_r2',
                         rule_sets({2}, {7}),
                         lambda *_: 12,
                         _e_2_7_1)


def _e_2_9_1(*_):
    # p324, Stroud 1971
    r = (1.53818_90013_20852,   1.22474_48713_91589,
         0.48171_65220_01144_3, 2.60734_98911_95855_4)
    s = (0, r[1], r[2], 0.96632_17712_79414_9)
    B = (0.12372_22328_85734_7, 0.06544_98469_49786_97,
         0.59352_80476_18087_5, 0.00134_90179_71918_148)
    return zip(B, map(full_symmetric, zip(r, s)))
e_2_9_1 = QuadratureRule('plane_r2',
                         rule_sets({2}, {9}),
                         lambda *_: 20,
                         _e_2_9_1)

def _e_2_11_1(*_):
    # p325, Stroud 1971
    r = (2.75781_63962_57008, 1.73205_08075_68877,
         0.52805_15301_59755_9, 1.22474_48713_91589, 0.70710_67811_86547_5)
    s = (0, 0, 0, 2.12132_03435_59643, 1.22474_48713_91589)
    B = (0.81766_45817_67541_7e-3, 0.43633_23129_98582_4e-1,
         0.53732_55214_49817_4, 0.36361_02608_32152e-2,
         0.98174_77042_46810_3e-1)
    return zip(B, map(full_symmetric, zip(r, s)))
e_2_11_1 = QuadratureRule('plane_r2',
                          rule_sets({2}, {11}),
                          lambda *_: 28,
                          _e_2_11_1)

def _e_2_13_1(*_):
    # p325, Stroud 1971
    r = (2.40315_17650_01966, 1.29847_99733_15986, 1.91242_82057_69905,
         0.94788_54439_69822_3, 0.31888_24732_57654_7, 3.32565_78296_63178,
         1.88222_84018_23884)
    s = (0, 0, r[2], r[3], r[4], 1.14552_72856_99371, 0.88260_73082_88965_9)
    A = -0.74829_13219_38036_3
    B = (0.35215_09661_09866_8e-2, 0.16500_55872_539264,
         0.85378_25937_94640_4e-3, 0.13269_38806_78933_6,
         0.64477_19928_48153_9, 0.17992_66413_50774_7e-4,
         0.12794_12775_88899_8e-1)
    return [(A, ((0, 0),)),
            *zip(B, map(full_symmetric, zip(r, s)))]
e_2_13_1 = QuadratureRule('plane_r2',
                          rule_sets({2}, {13}),
                          lambda *_: 37,
                          _e_2_13_1)

def _e_2_15_1(*_):
    # p326, Stroud 1971
    r = (3.53838_87281_21807, 2.35967_64168_77929, 1.31280_18446_20926,
         0.53895_59482_11420_5, 2.30027_99498_05658, 1.58113_88300_84189,
         0.84185_04335_81927_9, 2.68553_35817_55341, 1.74084_75143_97403)
    s = (0, 0, 0, 0, r[4], r[5], r[6], 1.11238_44317_71456, 0.72108_26504_86896)
    B = (0.80064_83569_65962_8e-5, 0.36045_77420_83826_4e-2,
         0.11876_09330_75913_7, 0.43724_88543_79140_2,
         0.36717_35075_83298_9e-4, 0.56548_66776_46162_7e-2,
         0.17777_74268_42424, 0.27354_49647_85329e-3, 0.20879_84556_93859_4e-1)
    return zip(B, map(full_symmetric, zip(r, s)))
e_2_15_1 = QuadratureRule('plane_r2',
                          rule_sets({2}, {15}),
                          lambda *_: 44,
                          _e_2_15_1)




def _e_3_7_1(*_):
    V = np.pi**1.5
    r = np.sqrt(0.25 * (15 + np.sqrt(15)))
    s = np.sqrt(0.5 * (6 - np.sqrt(15)))
    t = np.sqrt(0.5 * (9 + 2*np.sqrt(15)))
    A = ((720 + 8*np.sqrt(15)) / 2205) * V
    B = ((270 - 46*np.sqrt(15)) / 15435) * V
    C = ((162 + 41*np.sqrt(15)) / 6174) * V
    D = ((783 - 202*np.sqrt(15)) / 24696) * V
    return [(A, ((0, 0, 0),)),
            (B, full_symmetric((r, 0, 0))),
            (C, full_symmetric((s, s, 0))),
            (D, plus_minus((t, t, t)))]
e_3_7_1 = QuadratureRule('plane_r2',
                         rule_sets({3}, {7}),
                         lambda *_: 27,
                         _e_3_7_1)



def _e_3_14_1(*_):
    r = (0.7235510187, 1.468553289, 2.266580584, 3.190993201)
    A = np.array((0.2265043732, 0.1908084800, 0.02539731378, 0.0004032955750))
    points, weights = spherical_surface.u_3_14_1(3, 14)
    return zip(itertools.starmap(operator.mul, itertools.product(A, weights)),
               ((ri*np.array(vi),) for ri, vi in itertools.product(r, points)))
e_3_14_1 = QuadratureRule('plane_r2',
                          rule_sets({3}, {14}),
                          lambda *_: 288,
                          _e_3_14_1)


def plane_r2(dimension, degree):
    """
    Get the points and weights for a quadrature rule spanning all of a
    `dimension`-dimensional space (i.e. from -\\infty to \\infty in every
    dimension) with the weight function
        w(r) = exp(-r.r),  where r = (x_1, x_2, ..., x_n).

    If no suitable rule is known to the package, then `(None, None)` will be
    returned instead of `(points, weights)`.
    """
    scope = sys.modules[__name__]
    def predicate(cls):
        return isinstance(cls, QuadratureRule)\
               and cls.type == 'plane_r2'\
               and cls.allowed(dimension, degree)
    rules = [cls for _, cls in inspect.getmembers(scope, predicate)]
    if len(rules) is 0:
        return None, None
    rule = min(rules, key=lambda cls: cls.n_points(dimension, degree))
    return rule(dimension, degree)
