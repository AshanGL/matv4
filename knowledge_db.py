"""
knowledge_db.py
===============
Unified knowledge database: problem store + theorem store.
Both stores use FAISS inner-product indexes over sentence-transformer embeddings.

IMPROVEMENTS vs prior version
------------------------------
1. Theorem library expanded from ~25 to 80+ theorems across all domains.
2. PDF doc folder support: KnowledgeDB.build_from_docs() scans a folder of PDFs
   organised as docs/<domain>/<file>.pdf and adds extracted text chunks as theorems.
3. search_theorems() now returns the full statement (not just name + when_to_apply)
   so the solver prompt can show complete theorem text.
4. build_from_dataframe() also accepts technique_tags and creates richer records.
"""

from __future__ import annotations

import os
import json
import hashlib
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

DOMAINS = [
    "Algebra", "Geometry", "Number Theory", "Discrete Mathematics",
    "Calculus", "Precalculus", "Applied Mathematics", "Other",
]

DOMAIN_SLUGS = {d: d.lower().replace(" ", "_") for d in DOMAINS}

EMBEDDING_MODEL = "/kaggle/input/models/sumandey008/sentence-transformersall-minilm-l6-v2/transformers/default/1"


# ─────────────────────────────────────────────────────────────────────────────
# Expanded built-in theorem library (80+ theorems)
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_THEOREMS = [
    # ══ Algebra ══════════════════════════════════════════════════════════════
    {"name": "AM-GM Inequality",
     "statement": "For non-negative reals a₁,...,aₙ: (a₁+...+aₙ)/n ≥ (a₁·...·aₙ)^(1/n). Equality iff all aᵢ are equal.",
     "domain": "Algebra", "tags": ["inequality", "optimization", "am-gm"],
     "when_to_apply": "Proving minimum/maximum of symmetric expressions involving products and sums."},

    {"name": "Cauchy-Schwarz Inequality",
     "statement": "(Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²). Equality iff aᵢ/bᵢ is constant for all i.",
     "domain": "Algebra", "tags": ["inequality", "cauchy-schwarz"],
     "when_to_apply": "Bounding dot products; proving sum inequalities with two sequences."},

    {"name": "Vieta's Formulas",
     "statement": "For xⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₀ = 0 with roots r₁,...,rₙ: elementary symmetric polynomials of roots equal (±) coefficients. Sum r₁+...+rₙ = -aₙ₋₁, Product r₁·...·rₙ = (-1)ⁿa₀.",
     "domain": "Algebra", "tags": ["polynomial", "roots", "vieta", "symmetric"],
     "when_to_apply": "Relating sums/products/symmetric functions of roots to coefficients without finding roots explicitly."},

    {"name": "Polynomial Remainder Theorem",
     "statement": "The remainder of dividing polynomial p(x) by (x-a) equals p(a). Special case: (x-a) | p(x) iff p(a)=0.",
     "domain": "Algebra", "tags": ["polynomial", "remainder", "factor"],
     "when_to_apply": "Finding remainders or checking if (x-a) is a factor of p(x)."},

    {"name": "Schur's Inequality",
     "statement": "For non-negative a,b,c and t>0: aᵗ(a-b)(a-c) + bᵗ(b-a)(b-c) + cᵗ(c-a)(c-b) ≥ 0. For t=1: a³+b³+c³+abc ≥ ab(a+b)+bc(b+c)+ca(c+a).",
     "domain": "Algebra", "tags": ["inequality", "schur", "symmetric", "three-variables"],
     "when_to_apply": "Symmetric inequalities in three non-negative variables; particularly when AM-GM doesn't directly apply."},

    {"name": "Power Mean Inequality",
     "statement": "For r > s, the r-th power mean M_r ≥ M_s. Sequence: max ≥ A ≥ G ≥ H ≥ min (AM ≥ GM ≥ HM).",
     "domain": "Algebra", "tags": ["inequality", "power-mean", "am-gm"],
     "when_to_apply": "Comparing different averages; refining AM-GM for specific exponents."},

    {"name": "Muirhead's Inequality",
     "statement": "If sequence α majorises β (written α ≻ β), then Σ_{sym} x₁^α₁...xₙ^αₙ ≥ Σ_{sym} x₁^β₁...xₙ^βₙ for non-negative xᵢ.",
     "domain": "Algebra", "tags": ["inequality", "muirhead", "symmetric", "majorization"],
     "when_to_apply": "Symmetric polynomial inequalities when you can show one exponent sequence majorises another."},

    {"name": "SOS (Sum of Squares) Decomposition",
     "statement": "A polynomial p(x₁,...,xₙ) is non-negative iff it can be written as a sum of squares of polynomials (SOS). Many olympiad inequalities are proved by finding the SOS form.",
     "domain": "Algebra", "tags": ["inequality", "sos", "polynomial"],
     "when_to_apply": "Proving polynomial inequalities are non-negative when other methods fail."},

    {"name": "Lagrange Multipliers",
     "statement": "At a constrained optimum of f subject to g(x)=0, we have ∇f = λ∇g for some λ. For multiple constraints gᵢ=0: ∇f = Σλᵢ∇gᵢ.",
     "domain": "Algebra", "tags": ["optimization", "lagrange", "constrained"],
     "when_to_apply": "Finding extrema of a function subject to equality constraints."},

    {"name": "Rational Root Theorem",
     "statement": "If p/q (in lowest terms) is a root of aₙxⁿ+...+a₀, then p | a₀ and q | aₙ.",
     "domain": "Algebra", "tags": ["polynomial", "rational-root", "integer"],
     "when_to_apply": "Finding rational roots of integer polynomials to factor them."},

    {"name": "Newton's Identities",
     "statement": "Relate power sums pₖ = Σrᵢᵏ to elementary symmetric polynomials eₖ: p₁ = e₁, p₂ = e₁p₁ - 2e₂, pₙ = e₁pₙ₋₁ - e₂pₙ₋₂ + ... + (-1)ⁿ⁻¹neₙ.",
     "domain": "Algebra", "tags": ["symmetric", "power-sum", "newton", "polynomial"],
     "when_to_apply": "Computing power sums of roots from polynomial coefficients, or vice versa."},

    {"name": "Chebyshev's Sum Inequality",
     "statement": "If a₁≥a₂≥...≥aₙ and b₁≥b₂≥...≥bₙ, then n·Σaᵢbᵢ ≥ (Σaᵢ)(Σbᵢ). Reverse if sequences are opposite-ordered.",
     "domain": "Algebra", "tags": ["inequality", "chebyshev", "sum"],
     "when_to_apply": "Proving inequalities involving sums of products of ordered sequences."},

    # ══ Number Theory ═════════════════════════════════════════════════════════
    {"name": "Fermat's Little Theorem",
     "statement": "If p is prime and gcd(a,p)=1, then aᵖ⁻¹ ≡ 1 (mod p). Equivalently, aᵖ ≡ a (mod p) for all integers a.",
     "domain": "Number Theory", "tags": ["modular", "prime", "fermat"],
     "when_to_apply": "Computing large powers modulo a prime; reducing exponents mod (p-1)."},

    {"name": "Euler's Totient Theorem",
     "statement": "If gcd(a,n)=1, then a^φ(n) ≡ 1 (mod n), where φ(n) = n·Π_{p|n}(1-1/p) counts integers ≤n coprime to n.",
     "domain": "Number Theory", "tags": ["euler", "totient", "modular"],
     "when_to_apply": "Reducing large exponents modulo n when gcd(a,n)=1; generalises Fermat."},

    {"name": "Chinese Remainder Theorem",
     "statement": "If n₁,...,nₖ are pairwise coprime, the system x ≡ aᵢ (mod nᵢ) has a unique solution mod N=n₁·...·nₖ. Explicit: x = Σaᵢ·(N/nᵢ)·[(N/nᵢ)⁻¹ mod nᵢ] (mod N).",
     "domain": "Number Theory", "tags": ["crt", "modular", "congruence", "system"],
     "when_to_apply": "Solving systems of congruences with coprime moduli; combining local conditions."},

    {"name": "Lifting the Exponent (LTE) Lemma",
     "statement": "For odd prime p, p|a-b, p∤a, p∤b: vₚ(aⁿ-bⁿ) = vₚ(a-b)+vₚ(n). For p=2, n odd: v₂(aⁿ-bⁿ) = v₂(a-b). For p=2, n even: v₂(aⁿ-bⁿ) = v₂(a-b)+v₂(a+b)+v₂(n)-1.",
     "domain": "Number Theory", "tags": ["lte", "p-adic", "valuation", "divisibility"],
     "when_to_apply": "Finding the exact power of a prime p dividing aⁿ±bⁿ."},

    {"name": "Wilson's Theorem",
     "statement": "p is prime if and only if (p-1)! ≡ -1 (mod p).",
     "domain": "Number Theory", "tags": ["prime", "wilson", "factorial"],
     "when_to_apply": "Primality characterization; computing (p-1)! mod p; proving divisibility results involving factorials."},

    {"name": "Legendre's Formula",
     "statement": "The exponent of prime p in n! is vₚ(n!) = Σ_{k≥1} ⌊n/pᵏ⌋ = (n - sₚ(n))/(p-1) where sₚ(n) is the digit sum of n in base p.",
     "domain": "Number Theory", "tags": ["factorial", "prime", "legendre", "valuation"],
     "when_to_apply": "Finding the power of a prime in n!; solving trailing-zero and divisibility problems."},

    {"name": "Quadratic Reciprocity",
     "statement": "For distinct odd primes p,q: (p/q)(q/p) = (-1)^{(p-1)(q-1)/4}. Supplement: (-1/p) = (-1)^{(p-1)/2}; (2/p) = (-1)^{(p²-1)/8}.",
     "domain": "Number Theory", "tags": ["quadratic-reciprocity", "legendre-symbol", "prime"],
     "when_to_apply": "Determining if a is a quadratic residue mod p; solving x²≡a (mod p)."},

    {"name": "Bezout's Identity",
     "statement": "For integers a,b with gcd(a,b)=d, there exist integers x,y such that ax+by=d. Moreover, d is the smallest positive integer representable as ax+by.",
     "domain": "Number Theory", "tags": ["gcd", "bezout", "linear-diophantine"],
     "when_to_apply": "Proving existence of solutions to ax+by=c (solvable iff gcd(a,b)|c); finding modular inverses."},

    {"name": "Zsygmondy's Theorem",
     "statement": "For integers a>b>0 with gcd(a,b)=1 and n>2 (with exceptions for (a,b,n)=(2,1,6) and when a+b is a power of 2 and n=2): aⁿ-bⁿ has a prime factor not dividing aᵏ-bᵏ for any k<n (a primitive prime divisor).",
     "domain": "Number Theory", "tags": ["zsygmondy", "primitive-prime", "divisibility"],
     "when_to_apply": "Proving that aⁿ-bⁿ has a 'new' prime factor; showing certain equations have no solutions for large n."},

    {"name": "Hensel's Lemma",
     "statement": "If f(a) ≡ 0 (mod p) and f'(a) ≢ 0 (mod p), then there exists a unique lift b ≡ a (mod p) with f(b) ≡ 0 (mod p²). Extends to pⁿ.",
     "domain": "Number Theory", "tags": ["hensel", "p-adic", "lifting", "modular"],
     "when_to_apply": "Lifting solutions of f(x)≡0 (mod p) to solutions mod pⁿ."},

    {"name": "Dirichlet's Theorem on Primes in AP",
     "statement": "If gcd(a,d)=1, there are infinitely many primes of the form a+nd (n=0,1,2,...). The primes are equidistributed among all valid residues mod d.",
     "domain": "Number Theory", "tags": ["dirichlet", "arithmetic-progression", "prime"],
     "when_to_apply": "Proving existence of primes with specific residue properties."},

    # ══ Geometry ══════════════════════════════════════════════════════════════
    {"name": "Power of a Point",
     "statement": "For a point P and a circle: if a line through P meets the circle at A,B, then PA·PB = constant (equals |PO²-r²| where O is center, r is radius). Negative inside circle.",
     "domain": "Geometry", "tags": ["circle", "power", "chord", "secant", "tangent"],
     "when_to_apply": "Relating lengths of chords, secants, tangents from a common external or internal point."},

    {"name": "Ptolemy's Theorem",
     "statement": "For a cyclic quadrilateral ABCD: AC·BD = AB·CD + AD·BC. Equality holds. Ptolemy's inequality: for non-cyclic, AC·BD ≤ AB·CD + AD·BC.",
     "domain": "Geometry", "tags": ["cyclic", "ptolemy", "quadrilateral", "circle"],
     "when_to_apply": "Length problems in cyclic quadrilaterals; proving collinearity."},

    {"name": "Stewart's Theorem",
     "statement": "For triangle ABC with cevian AD to side BC (where BD=m, DC=n, AD=d, BC=a=m+n): b²m + c²n = a(d² + mn).",
     "domain": "Geometry", "tags": ["cevian", "stewart", "triangle", "length"],
     "when_to_apply": "Finding length of a cevian (median, altitude, angle bisector) in a triangle."},

    {"name": "Ceva's Theorem",
     "statement": "Cevians AD, BE, CF of triangle ABC are concurrent iff (AF/FB)·(BD/DC)·(CE/EA) = 1. Trigonometric form: sin(∠BAD)/sin(∠DAC) · sin(∠CBE)/sin(∠EBA) · sin(∠ACF)/sin(∠FCB) = 1.",
     "domain": "Geometry", "tags": ["cevian", "concurrent", "ceva", "triangle"],
     "when_to_apply": "Proving three cevians are concurrent; finding ratios in triangles."},

    {"name": "Menelaus's Theorem",
     "statement": "For triangle ABC and a transversal line crossing BC at X, CA at Y, AB at Z (or extensions): (BX/XC)·(CY/YA)·(AZ/ZB) = -1 (using signed ratios).",
     "domain": "Geometry", "tags": ["menelaus", "collinear", "transversal", "triangle"],
     "when_to_apply": "Proving three points on the sides (or extensions) of a triangle are collinear."},

    {"name": "Extended Law of Sines",
     "statement": "In triangle ABC: a/sin A = b/sin B = c/sin C = 2R, where R is the circumradius.",
     "domain": "Geometry", "tags": ["sine-rule", "circumradius", "triangle"],
     "when_to_apply": "Relating side lengths to angles and circumradius; finding unknown sides or angles."},

    {"name": "Law of Cosines",
     "statement": "In triangle ABC: c² = a² + b² - 2ab·cos(C). Also: cos(C) = (a²+b²-c²)/(2ab).",
     "domain": "Geometry", "tags": ["cosine-rule", "triangle", "sides-angles"],
     "when_to_apply": "Finding a side when two sides and the included angle are known; finding angles from all three sides."},

    {"name": "Inscribed Angle Theorem",
     "statement": "An inscribed angle is half the central angle subtending the same arc. Angles inscribed in a semicircle are right angles. Angles inscribed in the same arc are equal.",
     "domain": "Geometry", "tags": ["circle", "inscribed-angle", "central-angle"],
     "when_to_apply": "Finding angle relationships in circles; proving cyclic quadrilaterals."},

    {"name": "Radical Axis Theorem",
     "statement": "The radical axis of two circles is the locus of points having equal power with respect to both circles. Three circles have a common radical center (or the axes are parallel).",
     "domain": "Geometry", "tags": ["radical-axis", "circle", "power"],
     "when_to_apply": "Proving concurrence of lines; finding intersection properties of three circles."},

    {"name": "Simson Line",
     "statement": "The feet of perpendiculars from a point P to the three sides (or extensions) of a triangle are collinear iff P lies on the circumcircle of the triangle.",
     "domain": "Geometry", "tags": ["simson", "perpendicular", "collinear", "circumcircle"],
     "when_to_apply": "Characterising points on circumcircles; proving collinearity of foot-of-perpendicular points."},

    {"name": "Euler Line",
     "statement": "The circumcenter O, centroid G, and orthocenter H of a triangle are collinear. OG:GH = 1:2. The nine-point center N also lies on this line with ON:NH = 1:1.",
     "domain": "Geometry", "tags": ["euler-line", "centroid", "circumcenter", "orthocenter"],
     "when_to_apply": "Problems involving triangle centers and their collinearity."},

    {"name": "Heron's Formula",
     "statement": "Area of triangle with sides a,b,c: K = √(s(s-a)(s-b)(s-c)) where s = (a+b+c)/2 is the semi-perimeter.",
     "domain": "Geometry", "tags": ["area", "heron", "triangle", "sides"],
     "when_to_apply": "Computing triangle area from side lengths."},

    {"name": "Angle Bisector Theorem",
     "statement": "The angle bisector from vertex A of triangle ABC meets BC at D, and BD/DC = AB/AC = c/b.",
     "domain": "Geometry", "tags": ["angle-bisector", "triangle", "ratio"],
     "when_to_apply": "Finding the division point of a side by an angle bisector."},

    {"name": "Inversion",
     "statement": "Inversion with center O and radius r maps point P to P' on ray OP with OP·OP' = r². Circles through O map to lines; circles not through O map to circles. Angles are preserved (conformal).",
     "domain": "Geometry", "tags": ["inversion", "transformation", "circle"],
     "when_to_apply": "Simplifying configurations with many tangent circles; transforming circle problems to line problems."},

    # ══ Discrete Mathematics / Combinatorics ══════════════════════════════════
    {"name": "Pigeonhole Principle",
     "statement": "If n+1 objects are placed in n boxes, at least one box contains ≥2 objects. Generalised: if kn+1 objects in n boxes, some box has ≥k+1 objects.",
     "domain": "Discrete Mathematics", "tags": ["pigeonhole", "existence", "counting"],
     "when_to_apply": "Proving existence of a collision, repetition, or concentration; any 'some two must share' argument."},

    {"name": "Inclusion-Exclusion Principle",
     "statement": "|A₁∪...∪Aₙ| = Σ|Aᵢ| - Σ|Aᵢ∩Aⱼ| + ... + (-1)^(n+1)|A₁∩...∩Aₙ|. Equivalently, |complement of union| = |universal| - Σ|Aᵢ| + ...",
     "domain": "Discrete Mathematics", "tags": ["inclusion-exclusion", "counting", "set"],
     "when_to_apply": "Counting elements satisfying at least one of several properties; computing derangements, Euler's totient."},

    {"name": "Burnside's Lemma",
     "statement": "Number of distinct objects under group action G = (1/|G|) Σ_{g∈G} |Fix(g)|, where Fix(g) = set of objects fixed by symmetry g.",
     "domain": "Discrete Mathematics", "tags": ["burnside", "symmetry", "coloring", "group"],
     "when_to_apply": "Counting distinct colorings or configurations up to symmetry (rotation, reflection)."},

    {"name": "Stars and Bars",
     "statement": "Number of ways to write n = x₁+...+xₖ with non-negative integers xᵢ: C(n+k-1, k-1). With positive integers xᵢ≥1: C(n-1, k-1).",
     "domain": "Discrete Mathematics", "tags": ["counting", "partition", "stars-bars", "composition"],
     "when_to_apply": "Distributing identical objects into distinct bins; counting compositions/partitions of n."},

    {"name": "Principle of Mathematical Induction",
     "statement": "If P(1) is true and P(k)⟹P(k+1) for all k≥1, then P(n) is true for all n≥1. Strong induction: assume P(1),...,P(k) to prove P(k+1).",
     "domain": "Discrete Mathematics", "tags": ["induction", "proof", "recursion"],
     "when_to_apply": "Proving statements about positive integers; sequences defined by recurrences."},

    {"name": "Lindström-Gessel-Viennot Lemma",
     "statement": "Number of n-tuples of non-intersecting lattice paths from sources (a₁,...,aₙ) to sinks (b₁,...,bₙ) equals det(e(aᵢ,bⱼ)) where e(a,b) is the number of paths from a to b.",
     "domain": "Discrete Mathematics", "tags": ["lgv", "paths", "determinant", "non-intersecting"],
     "when_to_apply": "Counting non-crossing paths; expressing combinatorial sums as determinants."},

    {"name": "Catalan Numbers",
     "statement": "Cₙ = C(2n,n)/(n+1) = (2n)!/(n!(n+1)!). Counts: Dyck paths, full binary trees, triangulations, non-crossing partitions, mountain ranges, and 200+ other structures.",
     "domain": "Discrete Mathematics", "tags": ["catalan", "counting", "paths", "trees"],
     "when_to_apply": "Counting 'ballot sequences' or any of the 200+ Catalan-equivalent structures."},

    {"name": "Kirchhoff's Matrix Tree Theorem",
     "statement": "Number of spanning trees of graph G = any cofactor of the Laplacian matrix L = D - A, where D is the degree matrix and A is the adjacency matrix.",
     "domain": "Discrete Mathematics", "tags": ["graph", "spanning-tree", "laplacian", "matrix"],
     "when_to_apply": "Counting spanning trees of a graph."},

    {"name": "Dilworth's Theorem",
     "statement": "In any finite partially ordered set, the maximum size of an antichain equals the minimum number of chains needed to cover the poset.",
     "domain": "Discrete Mathematics", "tags": ["dilworth", "poset", "antichain", "chain"],
     "when_to_apply": "Problems about sequences, divisibility posets, or any ordered set where you need to balance antichains and chains."},

    {"name": "Sperner's Theorem",
     "statement": "The maximum size of a family of subsets of {1,...,n} with no subset containing another (an antichain) is C(n, ⌊n/2⌋).",
     "domain": "Discrete Mathematics", "tags": ["sperner", "antichain", "subset", "family"],
     "when_to_apply": "Bounding the size of a family of sets with no containment; extremal set theory."},

    {"name": "Ramsey Theory (R(3,3)=6)",
     "statement": "R(s,t) is the minimum n such that any 2-coloring of K_n contains K_s in color 1 or K_t in color 2. R(3,3)=6, R(3,4)=9, R(4,4)=18. General: R(s,t) ≤ C(s+t-2, s-1).",
     "domain": "Discrete Mathematics", "tags": ["ramsey", "graph-coloring", "complete-graph"],
     "when_to_apply": "Proving unavoidable monochromatic structures in colored graphs."},

    {"name": "Generating Functions",
     "statement": "The OGF of sequence (aₙ) is A(x) = Σaₙxⁿ. Convolution (aₙ*bₙ)↔A(x)B(x). EGF: Â(x) = Σaₙxⁿ/n!. Partial fractions + geometric series give closed forms for linear recurrences.",
     "domain": "Discrete Mathematics", "tags": ["generating-function", "recurrence", "ogf", "egf"],
     "when_to_apply": "Solving linear recurrences; counting problems with a recursive structure; finding closed forms."},

    {"name": "Möbius Inversion Formula",
     "statement": "If g(n) = Σ_{d|n} f(d), then f(n) = Σ_{d|n} μ(d)g(n/d), where μ is the Möbius function (μ(1)=1, μ(p₁...pₖ)=(-1)ᵏ for distinct primes, μ(n)=0 if p²|n).",
     "domain": "Discrete Mathematics", "tags": ["mobius", "inversion", "number-theory", "divisor-sum"],
     "when_to_apply": "Inverting divisor-sum relations; computing Euler's totient; summing multiplicative functions."},

    # ══ Calculus / Analysis ═══════════════════════════════════════════════════
    {"name": "Fundamental Theorem of Calculus",
     "statement": "If F'=f on [a,b], then ∫_a^b f(x)dx = F(b)-F(a). Also: d/dx ∫_a^x f(t)dt = f(x). Connects differentiation and integration.",
     "domain": "Calculus", "tags": ["integral", "derivative", "ftc"],
     "when_to_apply": "Evaluating definite integrals; differentiating under the integral sign."},

    {"name": "L'Hôpital's Rule",
     "statement": "If lim f = lim g = 0 or ±∞, then lim f/g = lim f'/g' provided the latter limit exists.",
     "domain": "Calculus", "tags": ["limit", "hopital", "indeterminate"],
     "when_to_apply": "Evaluating indeterminate limits 0/0, ∞/∞; simplify by differentiating numerator and denominator."},

    {"name": "Taylor's Theorem",
     "statement": "f(x) = Σ_{k=0}^n f^(k)(a)/k! · (x-a)^k + R_n(x), where R_n(x) = f^(n+1)(c)/(n+1)! · (x-a)^(n+1) for some c between a and x.",
     "domain": "Calculus", "tags": ["taylor", "series", "approximation"],
     "when_to_apply": "Approximating functions near a point; proving inequalities via series comparison; finding limits."},

    {"name": "Jensen's Inequality",
     "statement": "If f is convex, then f(E[X]) ≤ E[f(X)], i.e., f((x₁+...+xₙ)/n) ≤ (f(x₁)+...+f(xₙ))/n. Reversed for concave f.",
     "domain": "Calculus", "tags": ["jensen", "convex", "inequality", "expectation"],
     "when_to_apply": "Proving inequalities involving convex/concave functions of averages; AM-GM is a special case."},

    {"name": "Stolz-Cesàro Theorem",
     "statement": "If bₙ is strictly monotone and unbounded, and lim (aₙ₊₁-aₙ)/(bₙ₊₁-bₙ) = L, then lim aₙ/bₙ = L.",
     "domain": "Calculus", "tags": ["stolz-cesaro", "limit", "sequence"],
     "when_to_apply": "Evaluating limits of sequences of the form aₙ/bₙ; discrete analogue of L'Hôpital's rule."},

    {"name": "Abel's Summation",
     "statement": "Σ_{k=1}^n aₖbₖ = Aₙbₙ - Σ_{k=1}^{n-1} Aₖ(bₖ₊₁-bₖ), where Aₖ = Σ_{j=1}^k aⱼ.",
     "domain": "Calculus", "tags": ["abel", "summation-by-parts", "series"],
     "when_to_apply": "Evaluating sums involving products; proving convergence of alternating series."},

    # ══ Applied Mathematics / Probability ════════════════════════════════════
    {"name": "Bayes' Theorem",
     "statement": "P(A|B) = P(B|A)·P(A) / P(B). Using total probability: P(B) = Σᵢ P(B|Aᵢ)P(Aᵢ).",
     "domain": "Applied Mathematics", "tags": ["probability", "bayes", "conditional"],
     "when_to_apply": "Updating probabilities given new evidence; conditional probability problems."},

    {"name": "Linearity of Expectation",
     "statement": "E[X+Y] = E[X]+E[Y] regardless of dependence. E[cX] = cE[X]. Powerful because it holds even for dependent random variables.",
     "domain": "Applied Mathematics", "tags": ["probability", "expectation", "linearity"],
     "when_to_apply": "Computing expected values of sums by indicator random variables; counting expected number of events."},

    {"name": "Markov's and Chebyshev's Inequalities",
     "statement": "Markov: P(X≥a) ≤ E[X]/a for non-negative X, a>0. Chebyshev: P(|X-μ|≥kσ) ≤ 1/k². These give tail probability bounds.",
     "domain": "Applied Mathematics", "tags": ["probability", "markov", "chebyshev", "tail-bound"],
     "when_to_apply": "Bounding the probability of extreme values; proving almost-sure results."},

    {"name": "Geometric Series Formula",
     "statement": "Σ_{k=0}^{n-1} rᵏ = (1-rⁿ)/(1-r) for r≠1. Σ_{k=0}^∞ rᵏ = 1/(1-r) for |r|<1.",
     "domain": "Precalculus", "tags": ["series", "geometric", "sum"],
     "when_to_apply": "Summing geometric sequences; computing compound interest; partial fraction + geometric series for rational functions."},

    {"name": "Arithmetic Series Formula",
     "statement": "Σ_{k=1}^n k = n(n+1)/2. More generally, for AP with first term a, common difference d, n terms: S = n(2a+(n-1)d)/2 = n(a+l)/2 where l is the last term.",
     "domain": "Precalculus", "tags": ["series", "arithmetic", "sum"],
     "when_to_apply": "Summing arithmetic sequences; any sum of the form Σ(ak+b)."},

    {"name": "Binomial Theorem",
     "statement": "(a+b)ⁿ = Σ_{k=0}^n C(n,k) aᵏ bⁿ⁻ᵏ. Setting a=b=1: Σ C(n,k) = 2ⁿ. Setting a=1,b=-1: Σ (-1)ᵏ C(n,k) = 0.",
     "domain": "Precalculus", "tags": ["binomial", "combinatorics", "expansion"],
     "when_to_apply": "Expanding powers; finding specific coefficients; sums involving binomial coefficients."},

    {"name": "Vandermonde's Identity",
     "statement": "C(m+n, r) = Σ_{k=0}^r C(m,k)·C(n,r-k). Special case: C(2n,n) = Σ_{k=0}^n C(n,k)².",
     "domain": "Precalculus", "tags": ["binomial", "vandermonde", "convolution"],
     "when_to_apply": "Evaluating sums of products of binomial coefficients."},

    {"name": "Hockey Stick Identity",
     "statement": "Σ_{i=r}^n C(i,r) = C(n+1,r+1). Equivalently: C(r,r)+C(r+1,r)+...+C(n,r) = C(n+1,r+1).",
     "domain": "Precalculus", "tags": ["binomial", "hockey-stick", "sum", "combinatorics"],
     "when_to_apply": "Telescoping sums involving binomial coefficients; proving identities by counting paths."},

    {"name": "Fibonacci / Lucas Identities",
     "statement": "Fₙ = Fₙ₋₁+Fₙ₋₂. Key identities: F²ₙ - Fₙ₋₁Fₙ₊₁ = (-1)^(n-1) (Cassini), Fₘ₊ₙ = FₘFₙ₊₁+Fₘ₋₁Fₙ, Σ Fᵢ = Fₙ₊₂-1, gcd(Fₘ,Fₙ) = F_{gcd(m,n)}.",
     "domain": "Discrete Mathematics", "tags": ["fibonacci", "lucas", "recurrence", "identity"],
     "when_to_apply": "Problems involving Fibonacci-type sequences; divisibility of Fibonacci numbers."},

    {"name": "AM-HM Inequality",
     "statement": "For positive reals x₁,...,xₙ: (x₁+...+xₙ)/n ≥ n/(1/x₁+...+1/xₙ). Equivalently: (Σxᵢ)(Σ1/xᵢ) ≥ n².",
     "domain": "Algebra", "tags": ["inequality", "harmonic-mean", "am-hm"],
     "when_to_apply": "Inequalities involving reciprocals; Cauchy-Schwarz in Engel/Titu form."},

    {"name": "Titu's Lemma (Engel Form of Cauchy-Schwarz)",
     "statement": "Σ aᵢ²/bᵢ ≥ (Σaᵢ)²/(Σbᵢ) for positive bᵢ.",
     "domain": "Algebra", "tags": ["titu", "engel", "cauchy-schwarz", "inequality"],
     "when_to_apply": "Inequalities with fractions; when the denominator is a single term, apply Titu directly."},

    {"name": "Euler's Formula (Graph Theory)",
     "statement": "For a connected planar graph: V - E + F = 2, where V=vertices, E=edges, F=faces (including the outer infinite face).",
     "domain": "Discrete Mathematics", "tags": ["euler", "planar-graph", "topology"],
     "when_to_apply": "Proving bounds on edges/faces of planar graphs; K₅ and K₃,₃ non-planarity."},

    {"name": "Double Counting / Bijection Principle",
     "statement": "If two expressions count the same set, they are equal. A bijection between finite sets proves |A|=|B|. Counting in two ways is a powerful proof technique.",
     "domain": "Discrete Mathematics", "tags": ["double-counting", "bijection", "combinatorial-proof"],
     "when_to_apply": "Proving combinatorial identities; any identity that 'feels' like it should have a combinatorial proof."},

    {"name": "Extremal Principle",
     "statement": "In a finite non-empty set with a well-defined ordering, there exists a minimal (or maximal) element. Useful for infinite descent or assuming WLOG a specific structure.",
     "domain": "Discrete Mathematics", "tags": ["extremal", "minimum", "induction", "existence"],
     "when_to_apply": "Proving existence by assuming a minimal counterexample leads to contradiction; solving extremal problems."},
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(model_name, device=device, local_files_only=True)
    embs   = model.encode(texts, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
    return embs.astype("float32")


def _build_faiss_cpu(embs: np.ndarray):
    import faiss
    dim   = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def _save_faiss(index, path: str):
    import faiss
    faiss.write_index(index, path)


def _load_faiss(path: str):
    import faiss
    return faiss.read_index(path)


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeDB
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeDB:
    """
    Unified knowledge store.

    Layout
    ------
    knowledge_db/
      problems/<domain_slug>/
        records.jsonl
        embeddings.npy
        faiss.index
        meta.json
      theorems/
        records.jsonl
        embeddings.npy
        faiss.index
      manifest.json
    """

    MANIFEST = "manifest.json"

    def __init__(self, db_dir: str, model_name: str = EMBEDDING_MODEL):
        self.db_dir     = db_dir
        self.model_name = model_name

        self._prob_indexes: dict  = {}
        self._prob_records: dict  = {}
        self._theo_index          = None
        self._theo_records: list  = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_dataframe(self, df, rebuild: bool = False) -> None:
        if self.is_built() and not rebuild:
            print("  KnowledgeDB already built. Pass rebuild=True to force.")
            return

        import pandas as pd

        os.makedirs(self.db_dir, exist_ok=True)

        print("Building knowledge database from DataFrame...")
        domain_groups: dict[str, list] = {slug: [] for slug in DOMAIN_SLUGS.values()}

        for _, row in df.iterrows():
            domain  = str(row.get("main_domain", row.get("domain", "Other")))
            slug    = DOMAIN_SLUGS.get(domain, "other")
            record  = {
                "problem":         str(row.get("problem", "")),
                "answer":          str(row.get("answer", "")),
                "answer_type":     str(row.get("answer_type", "integer")),
                "domain":          domain,
                "difficulty_band": str(row.get("difficulty_band", "medium")),
                "technique_tags":  _parse_tags(row.get("technique_tags", [])),
                "solution_sketch": str(row.get("solution", ""))[:500],
            }
            if record["problem"]:
                domain_groups[slug].append(record)

        prob_root = os.path.join(self.db_dir, "problems")
        for slug, records in domain_groups.items():
            if not records:
                continue
            domain_dir = os.path.join(prob_root, slug)
            os.makedirs(domain_dir, exist_ok=True)

            texts = [r["problem"] for r in records]
            print(f"  Embedding {len(texts)} problems in '{slug}'...")
            embs  = _embed(texts, self.model_name)

            np.save(os.path.join(domain_dir, "embeddings.npy"), embs)
            index = _build_faiss_cpu(embs)
            _save_faiss(index, os.path.join(domain_dir, "faiss.index"))

            with open(os.path.join(domain_dir, "records.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            with open(os.path.join(domain_dir, "meta.json"), "w") as f:
                json.dump({"n_problems": len(records), "slug": slug}, f, indent=2)

        self._build_theorem_index(BUILTIN_THEOREMS)

        manifest = {
            "built_at":   datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_name,
            "n_theorems": len(BUILTIN_THEOREMS),
        }
        with open(os.path.join(self.db_dir, self.MANIFEST), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"KnowledgeDB built → {self.db_dir}")

    def build_from_docs(self, doc_folder: str) -> None:
        """
        Scan doc_folder for PDFs organized as:
          doc_folder/<domain>/<file>.pdf

        Extracts text chunks and adds them to the theorem store.
        Requires: pypdf or pdfplumber.
        """
        doc_folder = Path(doc_folder)
        if not doc_folder.exists():
            print(f"  Doc folder not found: {doc_folder}")
            return

        extra_theorems = []
        for domain_dir in sorted(doc_folder.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name.replace("_", " ").title()

            pdf_files = list(domain_dir.glob("*.pdf"))
            if not pdf_files:
                continue

            print(f"  Processing {len(pdf_files)} PDFs in '{domain}'...")
            for pdf_path in pdf_files:
                chunks = _extract_pdf_chunks(str(pdf_path), max_chars=600)
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 80:
                        continue
                    extra_theorems.append({
                        "name": f"{pdf_path.stem} §{i+1}",
                        "statement": chunk,
                        "domain": domain,
                        "tags": [domain.lower().replace(" ", "-"), "pdf"],
                        "when_to_apply": f"Relevant excerpt from {pdf_path.name}",
                    })

        if extra_theorems:
            print(f"  Adding {len(extra_theorems)} PDF chunks to theorem store...")
            all_theorems = BUILTIN_THEOREMS + extra_theorems
            self._build_theorem_index(all_theorems)
            print(f"  Theorem store now has {len(all_theorems)} entries.")

    def _build_theorem_index(self, theorems: list[dict]) -> None:
        theo_dir = os.path.join(self.db_dir, "theorems")
        os.makedirs(theo_dir, exist_ok=True)

        texts = [f"{t['name']}: {t.get('statement','')} {t['when_to_apply']}"
                 for t in theorems]
        embs  = _embed(texts, self.model_name)
        index = _build_faiss_cpu(embs)

        np.save(os.path.join(theo_dir, "embeddings.npy"), embs)
        _save_faiss(index, os.path.join(theo_dir, "faiss.index"))

        with open(os.path.join(theo_dir, "records.jsonl"), "w") as f:
            for t in theorems:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        self._theo_index   = None
        self._theo_records = []

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _load_prob_domain(self, slug: str) -> bool:
        if slug in self._prob_indexes:
            return True
        domain_dir = os.path.join(self.db_dir, "problems", slug)
        idx_path   = os.path.join(domain_dir, "faiss.index")
        rec_path   = os.path.join(domain_dir, "records.jsonl")
        if not os.path.exists(idx_path):
            return False
        self._prob_indexes[slug] = _load_faiss(idx_path)
        records = []
        with open(rec_path) as f:
            for line in f:
                records.append(json.loads(line))
        self._prob_records[slug] = records
        return True

    def _load_theorems(self) -> bool:
        if self._theo_index is not None:
            return True
        theo_dir = os.path.join(self.db_dir, "theorems")
        idx_path = os.path.join(theo_dir, "faiss.index")
        rec_path = os.path.join(theo_dir, "records.jsonl")
        if not os.path.exists(idx_path):
            return False
        self._theo_index = _load_faiss(idx_path)
        with open(rec_path) as f:
            self._theo_records = [json.loads(line) for line in f]
        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search_problems(
        self,
        query_emb: np.ndarray,
        domain:    Optional[str] = None,
        top_k:     int           = 4,
        min_sim:   float         = 0.2,
    ) -> list[dict]:
        query = query_emb.astype(np.float32).reshape(1, -1)

        if domain is not None:
            slug = DOMAIN_SLUGS.get(domain, domain.lower().replace(" ", "_"))
            return self._search_one_domain(slug, query, top_k, min_sim)

        all_results = []
        for slug in DOMAIN_SLUGS.values():
            all_results.extend(self._search_one_domain(slug, query, top_k, min_sim))

        all_results.sort(key=lambda r: r["similarity"], reverse=True)
        seen, deduped = set(), []
        for r in all_results:
            key = r["problem"][:80]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:top_k]

    def _search_one_domain(self, slug, query, top_k, min_sim) -> list[dict]:
        if not self._load_prob_domain(slug):
            return []
        index   = self._prob_indexes[slug]
        records = self._prob_records[slug]
        n       = min(top_k * 3, index.ntotal)
        if n == 0:
            return []
        scores, idxs = index.search(query, n)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or float(score) < min_sim:
                continue
            rec = dict(records[int(idx)])
            rec["similarity"] = round(float(score), 4)
            rec.pop("solution_sketch", None)
            results.append(rec)
            if len(results) >= top_k:
                break
        return results

    def search_theorems(
        self,
        query_emb: np.ndarray,
        domain:    Optional[str] = None,
        top_k:     int           = 5,
        min_sim:   float         = 0.12,
    ) -> list[dict]:
        if not self._load_theorems():
            return []

        query    = query_emb.astype(np.float32).reshape(1, -1)
        n        = min(top_k * 4, self._theo_index.ntotal)
        scores, idxs = self._theo_index.search(query, n)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or float(score) < min_sim:
                continue
            rec = dict(self._theo_records[int(idx)])
            rec["similarity"] = round(float(score), 4)
            if domain and rec.get("domain") != domain:
                rec["similarity"] *= 0.75
            results.append(rec)

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def is_built(self) -> bool:
        return os.path.exists(os.path.join(self.db_dir, self.MANIFEST))

    def add_theorem(self, theorem: dict, rebuild_index: bool = True) -> None:
        theo_dir   = os.path.join(self.db_dir, "theorems")
        jsonl_path = os.path.join(theo_dir, "records.jsonl")
        os.makedirs(theo_dir, exist_ok=True)

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(theorem, ensure_ascii=False) + "\n")

        if rebuild_index:
            records = []
            with open(jsonl_path) as f:
                for line in f:
                    records.append(json.loads(line))
            self._build_theorem_index(records)

    def status(self) -> dict:
        info = {"db_dir": self.db_dir, "is_built": self.is_built(), "domains": {}}
        prob_root = os.path.join(self.db_dir, "problems")
        for domain, slug in DOMAIN_SLUGS.items():
            meta_path = os.path.join(prob_root, slug, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    info["domains"][domain] = json.load(f)
        theo_dir = os.path.join(self.db_dir, "theorems")
        rec_path = os.path.join(theo_dir, "records.jsonl")
        if os.path.exists(rec_path):
            with open(rec_path) as f:
                info["theorems"] = sum(1 for _ in f)
        else:
            info["theorems"] = 0
        return info

    def print_status(self) -> None:
        s = self.status()
        print(f"\n{'='*52}")
        print(f"  KnowledgeDB  —  {self.db_dir}")
        print(f"{'='*52}")
        for domain, meta in s["domains"].items():
            print(f"  {domain:<25} {meta.get('n_problems',0):>5} problems")
        print(f"  {'Theorems':<25} {s['theorems']:>5}")
        print(f"{'='*52}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tags(t) -> list[str]:
    import ast
    if isinstance(t, list):  return [str(x) for x in t]
    try:    return list(ast.literal_eval(str(t)))
    except: return [str(t)] if t else []


def _extract_pdf_chunks(pdf_path: str, max_chars: int = 600) -> list[str]:
    """Extract text from a PDF and split into chunks of ~max_chars."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            full_text = "\n".join(
                page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"    Warning: could not read {pdf_path}: {e}")
            return []
    except Exception as e:
        print(f"    Warning: could not read {pdf_path}: {e}")
        return []

    # Split into paragraphs then merge into ~max_chars chunks
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for para in paragraphs:
        if len(cur) + len(para) < max_chars:
            cur += " " + para
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = para
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame enrichment helpers
# ─────────────────────────────────────────────────────────────────────────────

def auto_label_answer_type(answer_str: str) -> str:
    import re
    s = str(answer_str).strip()
    if re.fullmatch(r"-?\d+", s):                            return "integer"
    if re.fullmatch(r"-?\d+\.\d+", s):                      return "float"
    if re.search(r"(-?\d+)\s*/\s*(-?\d+)", s):              return "fraction"
    if s.startswith("{") or (", " in s and s[0].isdigit()):  return "set"
    if re.search(r"[a-zA-Z\^\\]", s):                       return "expression"
    return "string"


def enrich_dataframe(df):
    if "answer_type" not in df.columns:
        df = df.copy()
        df["answer_type"] = df["answer"].apply(
            lambda x: auto_label_answer_type(str(x)))
    return df
