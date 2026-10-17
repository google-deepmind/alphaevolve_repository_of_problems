"""
Exact Symbolic Computer-Assisted Proof in Q(sqrt(5)) for Problem 35:
Packing 11 Unit Cubes into a Cube of Side Length S* = 2 + 2/sqrt(5).

Proves using exact symbolic arithmetic (SymPy over Q(sqrt(5)), zero floating-point)
that at:
  w = 1/2 + 1/sqrt(5),  u = 3/(10*sqrt(5)),  v = 1/(10*sqrt(5)),
  cos(theta) = 2/sqrt(5),  sin(theta) = 1/sqrt(5),
all 11 unit cubes:
  1. Are contained within [-S*/2, S*/2]^3 where S* = 2 + 2/sqrt(5), and
  2. Have disjoint open interiors across all 55 pairs (i, j), identifying the
     exact algebraic separating axis and symbolic separation gap in Q(sqrt(5))
     for every pair.
"""

import sympy as sp

s5 = sp.sqrt(5)
half = sp.Rational(1, 2)
w = half + 1 / s5
u = sp.Rational(3, 10) / s5
v = sp.Rational(1, 10) / s5
c = 2 / s5
s = 1 / s5

# 8 local vertices of a unit cube [-1/2, 1/2]^3
local_verts = [
    sp.Matrix([sx * half, sy * half, sz * half])
    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
]

I3 = sp.eye(3)
# Cube 9: rot = Rz(-theta)
R9 = sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
# Cube 10: rot = Ry(+theta)
R10 = sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
# Cube 11: rot = Rx(+theta)
R11 = sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])

centers = [
    sp.Matrix([-w, -w, -w]),
    sp.Matrix([-w, -w,  w]),
    sp.Matrix([-w,  w, -w]),
    sp.Matrix([-w,  w,  w]),
    sp.Matrix([ w, -w, -w]),
    sp.Matrix([ w, -w,  w]),
    sp.Matrix([ w,  w, -w]),
    sp.Matrix([ w,  w,  w]),
    sp.Matrix([ u,  v,  w]),
    sp.Matrix([ v, -w, -u]),
    sp.Matrix([-w,  u, -v]),
]

rots = [I3] * 8 + [R9, R10, R11]

# Compute exact symbolic world vertices and axes for all 11 cubes
all_verts = []
all_axes = []
half_S = 1 + 1 / s5  # S*/2 = 1 + 1/sqrt(5)

for idx in range(11):
    R_mat = rots[idx]
    c_vec = centers[idx]
    verts = [R_mat * lv + c_vec for lv in local_verts]
    all_verts.append(verts)
    all_axes.append([R_mat.col(k) for k in range(3)])
    # Verify exact containment in [-half_S, +half_S]^3
    for v_pt in verts:
        for coord in v_pt:
            diff_hi = sp.simplify(half_S - coord)
            diff_lo = sp.simplify(coord + half_S)
            assert diff_hi >= 0 and diff_lo >= 0, f"Cube {idx+1} vertex outside box!"

print("1. EXACT SYMBOLIC CONTAINMENT VERIFIED:")
print(f"   All 88 vertices of the 11 cubes lie in [-S*/2, S*/2]^3 with S* = {2 * half_S} = {float(2 * half_S):.16f}")

# Verify all 55 pairs of cubes in exact symbolic Q(sqrt(5)) arithmetic
touching_pairs = []
strictly_separated_pairs = []

for i in range(11):
    for j in range(i + 1, 11):
        cand_axes = list(all_axes[i]) + list(all_axes[j])
        for a1 in all_axes[i]:
            for a2 in all_axes[j]:
                cr = a1.cross(a2)
                if not cr.is_zero_matrix:
                    cand_axes.append(cr)
        best_gap = None
        best_axis = None
        for ax in cand_axes:
            norm_sq = sp.simplify(ax.dot(ax))
            if norm_sq == 0:
                continue
            unit_ax = ax / sp.sqrt(norm_sq)
            p1 = [sp.simplify(v_pt.dot(unit_ax)) for v_pt in all_verts[i]]
            p2 = [sp.simplify(v_pt.dot(unit_ax)) for v_pt in all_verts[j]]
            min1, max1 = min(p1), max(p1)
            min2, max2 = min(p2), max(p2)
            gap = sp.simplify(sp.Max(min2 - max1, min1 - max2))
            if best_gap is None or float(gap) > float(best_gap):
                best_gap = gap
                best_axis = sp.simplify(unit_ax)
        assert best_gap >= 0, f"Overlap between Cube {i+1} and Cube {j+1}: {best_gap}"
        if best_gap == 0:
            touching_pairs.append((i + 1, j + 1, best_axis, best_gap))
        else:
            strictly_separated_pairs.append((i + 1, j + 1, best_axis, best_gap))

print(f"\n2. EXACT SYMBOLIC SEPARATION VERIFIED ACROSS ALL 55 PAIRS:")
print(f"   - Strictly separated pairs (gap > 0 in Q(sqrt(5))): {len(strictly_separated_pairs)}")
print(f"   - Exact contact pairs      (gap = 0 in Q(sqrt(5))): {len(touching_pairs)}")
print("\n   Exact contact pairs (i, j) and their unit separating normals n_{i,j}:")
for i_id, j_id, ax, gap in touching_pairs:
    ax_str = "[" + ", ".join(str(sp.simplify(x)) for x in ax) + "]"
    print(f"     Cubes ({i_id:2d}, {j_id:2d}): gap = {gap}, normal = {ax_str}")
