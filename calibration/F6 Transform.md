
# From Spatial Transform Formula to the F6 Group Structure: A Concise and Friendly Path of Expression

We begin with a classic spatial transform formula:

```plaintext
p = t + R * p′
```

This formula expresses that a point's position `p` in the **global coordinate system** can be computed from its position `p′` in the **local coordinate system**, using a rotation `R` and a translation `t`.

This expression is classic, but we aim to abstract from it a more fundamental idea: a spatial transform can be expressed as a structured pair `(t, R)`.

---

## From Formula to Abstraction

In this formula, `t` is a 3D vector representing translation, and `R` is a 3×3 rotation matrix. Together, they form a rigid transformation:

```plaintext
(t, R)
```

This abstraction is the first step in understanding spatial transforms. But when the transforms are nested multiple layers deep, this `(t, R)` expression becomes cumbersome.

Let’s look at its expression in a multi-transform scenario:

```plaintext
p = t + R * (t₁ + R₁ * (t₂ + R₂ * (t₃ + R₃ * p₃)))
```

While accurate, the nesting is exhausting to read, especially when there are `n` layers.

---

## Homogeneous Matrix Q: A Linearized Scheme

To resolve the complexity of nested expressions, people introduced the homogeneous matrix `Q`, a 4×4 matrix that combines translation and rotation in a unified framework:

```plaintext
Q = Q₁ · Q₂ · ... · Qₙ
```

This solution makes spatial transforms linear and convenient—especially suitable for matrix libraries and graphics pipelines. But it also brings new issues:

- 4×4 matrices consume more memory, not performance/storage friendly;
- It’s hard to visually separate the translation and rotation components;
- Not intuitive for beginners.

---

## F6 Notation: An Attempt to Hide the Rotation Matrix

We introduce the F6 notation. Its essence is still `(t, R)`, and it introduces no new mathematical entity. It simply makes one **user-friendly move**: hiding the explicit rotation matrix `R`.

This is done via two transform functions:

- `F62Q`: converts F6 to homogeneous matrix Q (for computation);
- `Q2F6`: converts Q back to F6 (for expression).

In this way, F6 becomes a **concise interface** for expressing spatial transforms, suitable for human reading and explanation.

---

## F6 Forms a Mathematical Group

We can view spatial transforms as a series of arrows from local coordinate frames to global ones:

```plaintext
a[i−1,i]: the transform from frame i−1 to frame i
```

A chain of such transforms `A` can be expressed as:

```plaintext
A = Πⁿᵢ₌₁ a[i−1,i]
```

Where Π denotes multiplication from i=1 to n.

In this sense, F6 forms a **mathematical group**:

- **Closure**: The combination of two F6s is still an F6;
- **Inverses**: Each transform has an inverse;
- **Identity**: There exists a unit transform;
- **Associativity**: Satisfied by composition.

This means F6 is not just a concise expression—it’s also **algebraically rigorous**, ready for generalization.

---

## Use Case: Recording and Analyzing a Robot’s Motion Path

With F6, we can obtain **each moment’s absolute orientation**, and any **relative orientation** between moments.

Imagine a scenario: a robot moves around in a factory. At each moment, its pose (position and direction) is recorded by sensors as an F6 transform.

Then we have a time series of F6 transforms:

```plaintext
T₀, T₁, T₂, ..., Tₙ
```

Each `Tᵢ` is the spatial transform from the world frame to the robot’s frame at time `i`.

With this data, we can:

- Get the robot’s **absolute pose** at any time;
- Compute the **relative motion** from time `m` to `n` via `invF6(Tₘ) * Tₙ`;
- Analyze the robot’s **local movement pattern**, like drift, shaking, or reversal;
- Compare the motion of two robots by comparing their F6 sequences.

This showcases the power of F6’s group structure: **composition** represents combined motion, **inverse** represents relative comparison. F6 is not just an expression—it’s a structure ready for computation and reasoning.

---

## Real-world Implementation and Library

F6 is not just a theoretical framework. A full Python implementation has been published on PyPI:

- Project: `f6transform`
- Installation:

  ```bash
  pip install f6transform
  ```

- GitHub: [https://github.com/xu-xiangxing/f6transform](https://github.com/xu-xiangxing/f6transform)

The library embodies the minimalist F6 form, hiding rotation matrices while supporting both teaching and engineering. Core APIs include:

- `combine_F6(a, b)`: combine two F6 transforms
- `invF6(a)`: compute inverse transform
- `combine_F6(a, {0})`: apply F6 to a point `p` (where `{0}` is p’s local pose)
- `F62Q(a)` / `Q2F6(Q)`: convert between F6 and Q

---

## Conclusion and Expectations

In spatial transform studies, classic formulas and homogeneous matrices have dominated. They are rigorous, yet inadvertently create entry barriers. Most engineers are familiar with `p = t + R*p′`, a few understand 4×4 matrices, but **almost no one realizes a friendlier alternative exists**.

The introduction of F6 aims to **break this silent constraint**. It does not reinvent the wheel—but rather "**builds bridges and opens paths**" through simplification. Its value lies not only in **conciseness**, but in **lowering the entry point** and **unleashing creativity**.

We hope:

- It can be included in `numpy` extension tools as a standard transform notation;
- It can be featured in future robotics or kinematics textbooks as a **human-friendly first step** in learning spatial transforms.
