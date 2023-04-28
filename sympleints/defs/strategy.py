from enum import Enum
from typing import Callable, List


class Strategy(Enum):
    FIRST = 1
    HIGHEST = 2
    ZERO = 3


class RecurStrategy:
    def __init__(
        self,
        base_case: Callable,
        funcs: List[Callable],
        strategy: Strategy = Strategy["FIRST"],
    ):
        self.base_case = base_case
        self.funcs = funcs
        self.strategy = strategy

    def recur(self, *args, strategy="first"):
        # Negative angular momenta always yield 0
        if min(args) < 0:
            return 0
        # After the first conditional all angular momenta are non-negative.

        # If all of them are 0 we reached the base case.
        if sum(args) == 0:
            return self.base_case()
        # Otherwise, utilize the provided recurrence relations, according to the
        # chosen strategy.

        # Decrement highest angular momentum
        if self.strategy == Strategy.HIGHEST:
            ind = args.index(max(args))
        # Decrement first non-zero angular momentum
        elif self.strategy == Strategy.FIRST:
            for ind, angmom in enumerate(args):
                if angmom > 0:
                    break
        # Decrement smallest angular momentum
        elif self.strategy == Strategy.ZERO:
            min_non_zero_arg = min([arg for arg in args if arg > 0])
            ind = args.index(min_non_zero_arg)
        mod_args = list(args).copy()
        mod_args[ind] -= 1
        return self.funcs[ind](*mod_args)
