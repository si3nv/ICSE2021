
Format
------
The benchmark is distributed as a CSV file. 
The first line is a header with column names: 
`A,B,A_ecosystem,B_ecosystem,relation_type,comment`

Columns `A` and `B` contain namespaces, i.e. strings used to import a software library.
In many ecosystems namespace is indistinguishable from the package name.
For example, in npm import statement `requre('react')` the string `react` 
used to import package is the same as the package name on npm.
However, Python statement `import sklearn` implies package `scikit-learn`.
In this example, `sklearn` is a namespace, while `scikit-learn` is a package name.

Columns `A_ecosystem` and `B_ecosystem` identify the ecosystem of the namespaces.
In the first version of the benchmark, only Python packages are used (`PY`).

Column `relation_type` indicates relations between A and B.
As of version 1.0, three values are possible: `competing`, `complementary` and `orthogonal`.
See more details in section Relation types.



Relation types
--------------
Important note: relations are directional. 
In many cases, relation A -> B does not imply B -> A.

A *competes* with B if A can replace B. 
The reverse might not be true, e.g. Python `click` (a package facilitating 
creation of CLI programs) can replace `docopt`, which only provides CLI interface
description, but `docopt` cannot replace `click` because it lacks argument parsing functionality. 

B is complementary to A if B extends functionality of A without changing its scope.
Example of complementary relation is `pandas` extending functionality of `numpy`.
However, `geopandas`, even though preserving some degree of similarity,
is arguably implementing a separate functionality than `pandas`.

A is orthogonal to B if they are not directly related.
For example, astronomical package `astropy` is not directly related to libraries
facilitating game creation, such as Cocos2D.
However, this cannot be said about template engines (e.g. `genshi`) in relation
to web frameworks, as those often use templates.

These definitions certainly leave a lot of gray area.
For example, `django` template engine can be used to replace `jinja2`,
but also `jinja2` is often used with `django` to extend its functionality 
(mostly for performance reasons).


Criteria
--------

Projects used in the benchmark should satisfy the following criteria:
- some evidence of usage. Originally, the threshold of 100 Github projects was used.
- competing packages are not trivial, i.e. they can be extended to actually compete.
  These packages should have at least three releases over at least six months.
  Ideally, the last release should not be more than two years old 
  for this competition to be observable. 
- namespace should not be common for local imports (e.g., `app`)
- uniqueness, or at least clear prevalence, of one package providing the namespace.
  This criterion is automatically satisfied in ecosystems using package name
  as the namespace, such as npm and Go.
  In others, like PyPI (Python) and Maven (Java) a namespace can be served by multiple packages.
  The only exception is when one package is clearly more common than others and
  use of the namespace in question with high probability can be related to one package.
  For example, in Python, namespace `numpy` is served by packages 
  `numpy`, `intel-numpy` and `gnumpy`, however in most cases statement `import numpy`
  implies use of the original `numpy` package.
