import numpy as np

class Permutation (object):
    secure = False          # additional checks during init
    print_cyclic = False      # cyclic notation for printing (slow!)
    def __init__(self, permlist=None, size=None):
        """Two valid inits:
            (1) permlist = [idx_0, idx_1, idx_2, .. idx_(N-1)]
            (2) size = N: equals perlist = [0,1,..,N-1]
           Note that permlist must exactly contain the indices 0,..,N-1.
           Class variable 'secure' [True|False] performs additional checks that slow down execution speed.
        """
        assert (permlist is None) != (size is None), " > Error: Either permlist or size must be given (mutually exclusive)!"
        if isinstance(permlist, int):   # convenience function
            size = permlist
            permlist = None
        if self.__class__.secure is False:
            if size:
                permlist = np.arange(size, dtype=int)
            else:
                permlist = np.array(permlist, dtype=int)
        elif self.__class__.secure is True:
            if permlist is not None:
                assert isinstance(permlist, (list, tuple, np.ndarray)), " > Error: permlist must be type list, tuple or ndarray."
                for n in permlist:
                    assert isinstance(n, (int, np.integer)), " Error: Indices must be type int."
                N = max(permlist) + 1
                assert (np.sort(permlist) == range(N)).all(), " Error: Must contain all indices from 0 to %d exactly once." % (N-1)
                permlist = np.array(permlist, dtype=int)
            else:
                assert isinstance(size, int) and size > 0, " Error: size must be positive integer."
                permlist = np.arange(size, dtype=int)
        self.perm = permlist

    def __str__(self):
        if self.__class__.print_cyclic:
            from sympy.combinatorics import Permutation as SympyPermutation
            return SympyPermutation(self.perm).__str__()
        else:
            return self.perm.__str__()

    def __call__(self, target):
        if self.__class__.secure:
            assert isinstance(target, np.ndarray), " Error: Permutation must be applied to type ndarray."
        return target[self.perm]

    def __mul__(self, other):
        return self.__class__(permlist=self(other.perm))

    def __eq__(self, other):
        return (self.perm == other.perm).all()

    def __ne__(self, other):
        return not (self == other)

    def inverse(self):
        idx = np.arange(self.perm.size)
        idx[self.perm] = np.arange(self.perm.size)
        return self.__class__(idx)




# # # # # # # # # # # # # # #
# # #   T E S T I N G   # # #
# # # # # # # # # # # # # # #

if __name__ == "__main__":
    from sympy.combinatorics import Permutation as SympyPermutation

    Permutation.secure = True
    Permutation.print_cyclic = True

    # Init
    l = [2,0,1,3,4]
    p = Permutation(l)
    q = SympyPermutation(l)

    print("\n__Contruct 1__")
    print("MyPer:", p)
    print("Sympy:", q)

    # Apply
    target = np.arange(10, 15)

    print("\n__perm(Target)__")
    print("MyPer:", p(target))
    print("Sympy:", q(target))

    # Multiply
    l2 = [4,0,2,1,3]
    r = Permutation(l2)
    s = SympyPermutation(l2)
    print("\n__Construct 2__")
    print("MyPer:", r)
    print("Sympy:", s)

    print("\n__Multiply__")
    print("MyPer: p*r =", p*r, "\t r*p =", r*p)
    print("Sympy: q*s =", q*s, "\t s*q =", s*q)

    print("\n__Multiplied(Target)__")
    print("MyPer: p*r :", (p*r)(target), "\t r*p :", (r*p)(target))
    print("Sympy: q*s :", (q*s)(target), "\t s*q :", (s*q)(target))


    # Equality
    print("\n__Equality__")
    print("MyPer: (p == p * p^-1 * p) is\t" , (p == p * p.inverse() * p) )
    print("MyPer: (p == p * p * p^-1) is\t" , (p == p * p * p.inverse()) )
    print("MyPer: (id == p * p * p) is\t" , (Permutation(p.perm.size) == p * p * p) )
    print("MyPer: (p == p * p) is\t" , (p == p * p) )

