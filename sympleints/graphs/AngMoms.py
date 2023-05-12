from dataclasses import dataclass
from enum import Enum
import functools
import random
from typing import Tuple

from sympleints.config import ARR_BASE_NAME
from sympleints.helpers import BFKind
from sympleints.graphs.triangle_rec import get_pos_dict


# TODO: don't hardcode L_max=8 here, but use a more sensible value?!
reduce_at_pos = get_pos_dict(L_max=10)


@functools.total_ordering
class LKind(Enum):
    def __init__(self, L, kind):
        self.L = L
        self.kind = kind

    sCART = 0, BFKind.CART
    pCART = 1, BFKind.CART
    dCART = 2, BFKind.CART
    fCART = 3, BFKind.CART
    gCART = 4, BFKind.CART
    hCART = 5, BFKind.CART
    iCART = 6, BFKind.CART
    jCART = 7, BFKind.CART
    kCART = 8, BFKind.CART
    lCART = 9, BFKind.CART
    mCART = 10, BFKind.CART

    sSPH = 0, BFKind.SPH
    pSPH = 1, BFKind.SPH
    dSPH = 2, BFKind.SPH
    fSPH = 3, BFKind.SPH
    gSPH = 4, BFKind.SPH
    hSPH = 5, BFKind.SPH
    iSPH = 6, BFKind.SPH
    jSPH = 7, BFKind.SPH
    kSPH = 8, BFKind.SPH
    lSPH = 9, BFKind.SPH
    mSPH = 10, BFKind.CART

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return (self.kind < other.kind) or (self.L < other.L)


LKindMap = {(lk.L, lk.kind): lk for lk in LKind}


@dataclass(frozen=True, order=True)
class SphAngMom:
    l: int
    m: int

    is_aux = False
    is_sph = True

    @staticmethod
    def from_L_tot(L_tot):
        return SphAngMom(L_tot, 0)

    def valid(self):
        l = self.l
        return (l >= 0) and (-l <= self.m <= l)

    def as_short_str(self):
        return f"s{self.l}"

    def compressed(self, positive=True):
        if positive:
            compr = self.l
        else:
            compr = -self.l
        return compr

    def to_LKind(self):
        return LKindMap[(self.l, BFKind.SPH)]

    def __getitem__(self, index):
        return (self.l, self.m)[index]

    def __str__(self):
        return f"({self.l},{self.m})"


@dataclass(frozen=True, order=True)
class CartAngMom:
    x: int
    y: int
    z: int

    is_aux = False
    is_sph = False
    kind = BFKind.CART

    @staticmethod
    def from_L_tot(L_tot):
        return CartAngMom(L_tot, 0, 0)

    @property
    def L(self):
        return self.x + self.y + self.z

    def modify(self, index, mod):
        mod_ams = [self.x, self.y, self.z]
        mod_ams[index] += mod
        return CartAngMom(*mod_ams)

    def as_short_str(self):
        return f"{self.L}"

    def non_zero_indices(self):
        return [i for i, am in enumerate((self.x, self.y, self.z)) if am > 0]

    def valid(self):
        return (self.x >= 0) and (self.y >= 0) and (self.z >= 0)

    def compressed(self, positive=True):
        return self.L

    # TODO: remove this. It is already provided as dataclasses.astuple
    def as_tuple(self):
        return (self.x, self.y, self.z)

    def to_LKind(self):
        return LKindMap[(self.L, BFKind.CART)]

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

    def __str__(self):
        return f"({self.x},{self.y},{self.z})"


@dataclass(frozen=True, order=True)
class AuxIndex:
    n: int

    is_aux = True
    is_sph = False

    def modify(self, _, mod):
        return AuxIndex(self.n + mod)

    @staticmethod
    def from_L_tot(L_tot):
        return AuxIndex(L_tot)

    def as_short_str(self):
        return f"a{self.n}"

    def valid(self):
        return self.n >= 0

    def compressed(self, positive=True):
        return self.n

    def __neg__(self):
        return -self.n

    def __str__(self):
        return f"({self.n},)"


AngMomMap = {
    BFKind.CART: CartAngMom,
    BFKind.SPH: SphAngMom,
}


@dataclass(frozen=True, order=True)
class KindedLTots:
    Ls: Tuple[int]
    kinds: Tuple[BFKind]

    def __post_init__(self):
        assert len(self.Ls) == len(self.kinds)

    def __len__(self):
        return len(self.Ls)

    def as_key(self):
        key = list()
        for L, k in zip(self.Ls, self.kinds):
            key.append(LKindMap[(L, k)])
        return tuple(key)


def name_for_ang_moms(ang_moms, base=ARR_BASE_NAME):
    prefix = "".join([am.as_short_str() for am in ang_moms])
    return f"{base}_{prefix}"


def name_for_L_tots_kinds(L_tots, kinds, base=ARR_BASE_NAME, with_aux=True):
    ang_moms = list()
    for L_tot, kind in zip(L_tots, kinds):
        ang_moms.append(AngMomMap[kind].from_L_tot(L_tot))
    if with_aux:
        ang_moms.append(AuxIndex(0))
    return name_for_ang_moms(ang_moms, base=base)


class AngMoms:
    def __init__(self, ang_moms):
        self.ang_moms = tuple(ang_moms)

        self.with_aux = any([type(am) == AuxIndex for am in self.ang_moms])

    """
    def as_tuple(self, drop_aux=False):
        if drop_aux and self.with_aux:
            return tuple([am for am in self.ang_moms if not am.is_aux])
        else:
            return self.ang_moms
    """

    def drop_aux(self):
        return AngMoms([am for am in self.ang_moms if not am.is_aux])

    def add_aux(self, aux: AuxIndex):
        return AngMoms(self.ang_moms + (aux,))

    def with_ang_mom_at(self, ang_mom, index):
        new_ang_moms = list()
        for i, am in enumerate(self.ang_moms):
            if i == index:
                am = ang_mom
            new_ang_moms.append(am)
        return AngMoms(new_ang_moms)

    @staticmethod
    def from_iterables(iterables):
        kinds = {
            1: AuxIndex,
            2: SphAngMom,
            3: CartAngMom,
        }
        ang_moms = [kinds[len(am)](*am) for am in iterables]
        return AngMoms(ang_moms)

    # TODO: deprecate this function
    @staticmethod
    def from_tuples(tuples):
        return AngMoms.from_iterables(tuples)

    @staticmethod
    def from_str(str_):
        return AngMoms.from_tuples(eval(str_))

    def compressed(self, drop_aux=False, positive=True):
        comp = list()
        for entry in self.ang_moms:
            if drop_aux and entry.is_aux:
                continue
            comp.append(entry.compressed(positive=positive))
        return tuple(comp)

    def compressed_unique(self, drop_aux=False):
        return self.compressed(drop_aux=drop_aux, positive=False)

    def with_order(self, order):
        ang_moms = [self.ang_moms[o] for o in order]
        if self.with_aux:
            ang_moms += [
                self.aux,
            ]
        return AngMoms(ang_moms)

    def as_key(self, drop_aux=False):
        key = list()
        for entry in self.ang_moms:
            if entry.is_aux:
                if drop_aux:
                    continue
                lk = entry.n
            else:
                lk = entry.to_LKind()
            key.append(lk)
        return tuple(key)

    def to_kinded_tot_ang_moms(self):
        comp = list()
        kinds = list()
        for entry in self.ang_moms:
            if entry.is_aux:
                continue
            elif entry.is_sph:
                kind = BFKind.SPH
            else:
                kind = BFKind.CART
            comp.append(entry.compressed(positive=True))
            kinds.append(kind)
        return KindedLTots(tuple(comp), tuple(kinds))

    def name(self, base=ARR_BASE_NAME):
        return name_for_ang_moms(self.ang_moms, base)

    @property
    def aux(self):
        return self.ang_moms[-1] if self.with_aux else None

    @property
    def aux_index(self):
        return len(self.ang_moms) - 1 if self.with_aux else None

    def is_spherical(self, index) -> bool:
        return type(self.ang_moms[index]) == SphAngMom

    def get_reduce_index(self, index, prefer_index=None, L_target=None):
        # AuxInd has only 1 entry, so we always return 0.
        if index == self.aux_index:
            reduce_index = 0
        else:
            ams = self.ang_moms[index]
            try:
                non_zero_indices = ams.non_zero_indices()
                # If a preferred index was supplied try to use it, if possible.
                if prefer_index in non_zero_indices:
                    reduce_index = prefer_index
                elif L_target is not None:
                    L = sum(ams)
                    reduce_index = reduce_at_pos[L_target][L][ams]
                # Otherwise pick a random index
                else:
                    reduce_index = random.choice(non_zero_indices)
            # IndexError will be raised if the basecase is reached
            except (AttributeError, IndexError):
                reduce_index = None
        return reduce_index

    def get_reduce_index(self, index, prefer_index=None, L_target=None, name=""):
        # AuxInd has only 1 entry, so we always return 0.
        if index == self.aux_index:
            reduce_index = 0
        else:
            ams = self.ang_moms[index]
            L = sum(ams)
            if L == 0:
                return None
            non_zero_indices = ams.non_zero_indices()
            # If a preferred index was supplied try to use it, if possible.
            if prefer_index in non_zero_indices:
                reduce_index = prefer_index
            elif L_target is not None:
                try:
                    reduce_index = reduce_at_pos[L_target][L][ams.as_tuple()]
                # This except block is triggered in cases, where angular momentum is built
                # up beyond the actually target angular momentum, e.g., in the VRR of
                # 3-center-2-electron integrals, where the angular momentum of A has to be
                # built up to L_A + L_B, while only L_A is actually required for A.
                except KeyError:
                    reduce_index = reduce_at_pos[L][L][ams.as_tuple()]
            # Otherwise pick a random index
            else:
                reduce_index = random.choice(non_zero_indices)
        return reduce_index

    def apply_mask(self, mask, reduce_index):
        if reduce_index is None:
            return None

        mod_ang_moms = list(self.ang_moms)
        for i, mod in enumerate(mask):
            if mod == 0:
                continue
            mod_ang_moms[i] = mod_ang_moms[i].modify(reduce_index, mod)
        if not all([am.valid() for am in mod_ang_moms]):
            mod_ang_moms = None
        else:
            mod_ang_moms = AngMoms(tuple(mod_ang_moms))
        return mod_ang_moms

    def __len__(self):
        return len(self.ang_moms)

    def __getitem__(self, index):
        return self.ang_moms[index]

    def __hash__(self):
        return hash(self.ang_moms)

    def __eq__(self, other):
        return all(
            [this == other for this, other in zip(self.ang_moms, other.ang_moms)]
        )

    # Previously, __lt__ did not return anything, so I guess the function was not
    # needed ...
    # def __lt__(self, other):
    # return self.ang_moms < other.ang_moms

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "(" + ",".join(map(str, self.ang_moms)) + ")"
