# ========================
# Part 1: Vector from scratch
# ========================
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


# ========================
# Part 2: Matrix from scratch
# ========================
import random

class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


# ========================
# Part 3: Projection & Gram-Schmidt
# ========================
def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])

def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


if __name__ == "__main__":
    print("=== Part 1: Vector ===")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a + b = {a + b}")
    print(f"a · b = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"cosine similarity = {a.cosine_similarity(b):.4f}")

    print("\n=== Part 2: Matrix ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])
    point = Vector([3, 1])
    rotated = rotation_90 @ point
    print(f"Original: {point}")
    print(f"Rotated 90°: {rotated}")

    random.seed(42)
    weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
    input_vector = Vector([1.0, 0.5, -0.3])
    output = weights @ input_vector
    print(f"\nInput (3D): {input_vector}")
    print(f"Output (2D): {output}")
    print("This is what a neural network layer does.")

    print("\n=== Part 3: Projection & Gram-Schmidt ===")
    a = Vector([3, 4])
    b = Vector([1, 0])
    proj = project(a, b)
    print(f"Проекция {a} на {b}: {proj}")
    print(f"Остаток (перпендикуляр): {a - proj}")

    v1 = Vector([1, 0, 0])
    v2 = Vector([1, 1, 0])
    v3 = Vector([1, 1, 1])
    basis = gram_schmidt([v1, v2, v3])
    print(f"\nOrthonormal basis:")
    for i, u in enumerate(basis):
        print(f"  u{i+1} = {u}, |u| = {u.magnitude():.6f}")
    print(f"u1·u2 = {basis[0].dot(basis[1]):.6f}")
    print(f"u1·u3 = {basis[0].dot(basis[2]):.6f}")
    print(f"u2·u3 = {basis[1].dot(basis[2]):.6f}")


# ========================

import math

def angle_between(a, b):
    cos_angle = a.cosine_similarity(b)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))

print("\n=== Exercise 1: Angle between vectors ===")
ea = Vector([1, 0])
eb = Vector([0, 1])
ec = Vector([1, 1])
print(f"Angle [1,0] and [0,1]: {angle_between(ea, eb):.2f}°")
print(f"Angle [1,0] and [1,1]: {angle_between(ea, ec):.2f}°")
print(f"Angle [1,0] and [1,0]: {angle_between(ea, ea):.2f}°")

print("\n=== Exercise 2: Scaling matrix ===")
scale = Matrix([[2, 0], [0, 3]])
sp1 = Vector([1, 1])
print(f"Original: {sp1}, Scaled (2x,3y): {scale @ sp1}")
sp2 = Vector([3, 2])
print(f"Original: {sp2}, Scaled: {scale @ sp2}")

print("\n=== Exercise 3: Most similar word vectors ===")
random.seed(1)
word_vectors = [Vector([random.gauss(0, 1) for _ in range(50)]) for _ in range(5)]
best_pair, best_sim = None, -2
for i in range(len(word_vectors)):
    for j in range(i+1, len(word_vectors)):
        sim = word_vectors[i].cosine_similarity(word_vectors[j])
        print(f"  v{i} vs v{j}: {sim:.4f}")
        if sim > best_sim:
            best_sim, best_pair = sim, (i, j)
print(f"Most similar: v{best_pair[0]} and v{best_pair[1]} (similarity={best_sim:.4f})")

print("\n=== Exercise 4: Verify Gram-Schmidt orthonormality ===")
gv1, gv2, gv3 = Vector([2, 1, 0]), Vector([1, 2, 1]), Vector([0, 1, 3])
gbasis = gram_schmidt([gv1, gv2, gv3])
for i in range(len(gbasis)):
    for j in range(i+1, len(gbasis)):
        print(f"  u{i+1}·u{j+1} = {gbasis[i].dot(gbasis[j]):.10f} (should be ~0)")
for i, u in enumerate(gbasis):
    print(f"  |u{i+1}| = {u.magnitude():.10f} (should be 1)")

print("\n=== Exercise 5: Rank-2 matrix in 3x3 ===")
import numpy as np
M = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
print(f"Matrix:\n{M}")
print(f"Rank: {np.linalg.matrix_rank(M)}")
print("Row 2 = 2 * Row 1 -> dependent. Columns span a 2D plane in 3D space.")

print("\n=== Exercise 6: Project [1,2,3] onto [1,1,1] ===")
pa, pb = Vector([1, 2, 3]), Vector([1, 1, 1])
pproj = project(pa, pb)
print(f"Projection: {pproj}")
scalar = pa.dot(pb) / pb.dot(pb)
print(f"Scalar = {scalar:.4f} = mean of components of a = {sum(pa.components)/3:.4f}")
print("Geometrically: extracts how much of 'a' points equally along all 3 axes.")
