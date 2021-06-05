#
# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
# This file is part of the pdaggerq package.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typing import List, Tuple, Union, Sequence
import copy
import numpy as np


class Index:

    def __init__(self, name: str, support: str):
        """
        Generate an index that acts on a particular space

        Later we might  want to generate the axis over which this index
        is used to generate einsum code.  See numpypsi4 for this.

        :param name: how the index shows up
        :param support: where the index ranges. Options: occ,virt,all
        """
        self.name = name
        self.support = support

    def __repr__(self):
        return "{}".format(self.name)

    def __eq__(self, other):
        if not isinstance(other, Index):
            raise TypeError("Can't compare non Index object to Index")
        return other.name == self.name and other.support == self.support

    def __ne__(self, other):
        return not self.__eq__(other)


class BaseTerm:
    """
    Base object for building named abstract tensors

    These objects one can ONLY be composed by multiplication with other BaseTerms
    to produce new TensorTerms.  They can also be checked for equality.
    """
    def __init__(self, *, indices: Tuple[Index, ...], name: str):
        self.name = name
        self.indices = indices

    def __repr__(self):
        return "{}".format(self.name) + "(" + ",".join(
            repr(xx) for xx in self.indices) + ")"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __mul__(self, other):
        # what about numpy floats  and such?
        if isinstance(other, BaseTerm):
            return TensorTerm((copy.deepcopy(self), other))
        elif isinstance(other, TensorTerm):
            self_copy = copy.deepcopy(self)
            return other.__mul__(self_copy)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        name_equality = other.name == self.name
        if len(self.indices) == len(other.indices):
            index_equal = all([self.indices[xx] == other.indices[xx] for xx in
                               range(len(self.indices))])
            return name_equality and index_equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class TensorTerm:
    """
    collection  of BaseTerms that can be translated to a einsnum contraction
    """

    def __init__(self, base_terms: Tuple[BaseTerm, ...], coefficient=1.0):
        self.base_terms = base_terms
        self.coefficient = coefficient

    def __repr__(self):
        return "{: 5.4f} ".format(self.coefficient) + "*".join(
            xx.__repr__() for xx in self.base_terms)

    def __mul__(self, other):
        self_copy = copy.deepcopy(self)
        if isinstance(other, (int, float, complex)):
            self_copy.coefficient *= other
        elif isinstance(other, BaseTerm):
            self_copy.base_terms = self_copy.base_terms + (other,)
        return self_copy

    def __rmul__(self, other):
        return self.__mul__(other)

    def einsum_string(self, update_val,
                      output_variables=None,
                      occupied=['i', 'j', 'k', 'l', 'm', 'n'],
                      virtual=['a', 'b', 'c', 'd', 'e', 'f'],
                      occ_char=None,
                      virt_char=None,):
        einsum_out_strings = ""
        einsum_tensors = []
        tensor_out_idx = []
        einsum_strings = []
        if occ_char is None:
            # in our code this will be a slice. o = slice(None, nocc)
            occ_char = 'o'
        if virt_char is None:
            virt_char = 'v'  # v = slice(nocc, None)

        for bt in self.base_terms:
            tensor_index_ranges = []
            string_indices = [xx.name for xx in bt.indices]
            for idx_type in string_indices:
                if idx_type in occupied:
                    if bt.name in ['h', 'g', 'f']:
                        tensor_index_ranges.append(occ_char)
                    else:
                        tensor_index_ranges.append(':')
                    if output_variables is not None:
                        if idx_type in output_variables:
                            tensor_out_idx.append(idx_type)
                elif idx_type in virtual:  # virtual
                    if bt.name in ['h', 'g', 'f']:
                        tensor_index_ranges.append(virt_char)
                    else:
                        tensor_index_ranges.append(':')
                    if output_variables is not None:
                        if idx_type in output_variables:
                            tensor_out_idx.append(idx_type)
                else:  # route to output with ->
                    tensor_index_ranges.append(idx_type)

            if bt.name in ['t1', 't2', 'l2', 'l1', 'r1', 'r2']:
                einsum_tensors.append(bt.name)
            else:
                einsum_tensors.append(
                    bt.name + "[" + ", ".join(tensor_index_ranges) + "]")
            einsum_strings.append("".join(string_indices))
        if tensor_out_idx:
            einsum_out_strings += "->{}".format("".join(sorted(tensor_out_idx)))

        teinsum_string = "+= {} * einsum(\'".format(self.coefficient)

        if len(einsum_strings) > 2:
            sorbs = 8
            nocc = 4
            nvirt = sorbs - nocc
            o = slice(0, nocc, 1)
            v = slice(nocc, sorbs, 1)
            h = np.zeros((sorbs, sorbs))
            f = np.zeros((sorbs, sorbs))
            g = np.zeros((sorbs, sorbs, sorbs, sorbs))
            t1 = np.zeros((nvirt, nocc))
            t2 = np.zeros((nvirt, nvirt, nocc, nocc))
            l2 = np.zeros((nocc, nocc, nvirt, nvirt))
            l1 = np.zeros((nocc, nvirt))
            einsum_path_string = "np.einsum_path(\'".format(self.coefficient)
            einsum_path_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ", optimize=\'optimal\')"
            einsum_optimal_path = eval(einsum_path_string)
            # print(einsum_optimal_path[1])
            teinsum_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ", optimize={})".format(einsum_optimal_path[0])
        else:
            teinsum_string += ",".join(
                einsum_strings) + einsum_out_strings + "\', " + ", ".join(
                einsum_tensors) + ")"
        if update_val is not None:
            teinsum_string = update_val + " " + teinsum_string
        return teinsum_string


class Right0amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='r0'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r0"


class Right1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='r1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r1({},{})".format(self.indices[0], self.indices[1])


class Right2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='r2'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "r2({},{})".format(self.indices[0], self.indices[1],
                                  self.indices[2], self.indices[3])


class Left0amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='l0'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l0"


class Left1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='l1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l1({},{})".format(self.indices[0], self.indices[1])


class Left2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='l2'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "l2({},{},{},{})".format(self.indices[0], self.indices[1],
                                  self.indices[2], self.indices[3])


class D1(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='d1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "d1({},{})".format(self.indices[0], self.indices[1])


class T1amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='t1'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "t1({},{})".format(self.indices[0], self.indices[1])


class T2amps(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='t2'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "t2({},{},{},{})".format(self.indices[0], self.indices[1],
                                        self.indices[2], self.indices[3])


class OneBody(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='h'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "h({},{})".format(self.indices[0], self.indices[1])

class FockMat(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='f'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "f({},{})".format(self.indices[0], self.indices[1])


class TwoBody(BaseTerm):

    def __init__(self, *, indices=Tuple[Index,...], name='g'):
        super().__init__(indices=indices, name=name)

    def __repr__(self):
        return "<{},{}||{},{}>".format(self.indices[0], self.indices[1],
                                       self.indices[2], self.indices[3])


class Delta(BaseTerm):
    def __repr__(self):
        return "d({},{})".format(self.indices[0], self.indices[1])