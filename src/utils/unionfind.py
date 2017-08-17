#
#   Copyright (c) 2010-2017, MIT Probabilistic Computing Project
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
#

from collections import OrderedDict


class Node(object):
    """Union-find node."""
    def __init__(self, element):
        self.element = element
        self.parent = None
        self.rank = 0

def find(node):
    """Find current canonical representative equivalent to node.

    Adjust the parent pointer of each node along the way to the root
    to point directly at the root for inverse-Ackerman-fast access.
    """
    if node.parent is None:
        return node
    root = node
    while root.parent is not None:
        root = root.parent
    parent = node
    while parent.parent is not root:
        grandparent = parent.parent
        parent.parent = root
        parent = grandparent
    return root

def union(a, b):
    """Assert equality of two nodes a and b so find(a) is find(b)."""
    a = find(a)
    b = find(b)
    if a is not b:
        if a.rank < b.rank:
            a.parent = b
        elif b.rank < a.rank:
            b.parent = a
        else:
            b.parent = a
            a.rank += 1

def classes(equivalences):
    """Compute mapping from element to list of equivalent elements.

    `equivalences` is an iterable of (x, y) tuples representing
    equivalences x ~ y.

    Returns an OrderedDict mapping each x to the list of elements
    equivalent to x.
    """
    node = OrderedDict()
    def N(x):
        if x in node:
            return node[x]
        n = node[x] = Node(x)
        return n
    for x, y in equivalences:
        union(N(x), N(y))
    eqclass = OrderedDict()
    for x, n in node.iteritems():
        x_ = find(n).element
        if x_ not in eqclass:
            eqclass[x_] = []
        eqclass[x_].append(x)
        eqclass[x] = eqclass[x_]
    return eqclass
