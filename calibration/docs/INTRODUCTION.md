# F6Transform: A New Philosophy for Spatial Transformation and 3D Perception

Welcome to the F6Transform project. This is more than just a software library; it is a new, intuitive, and powerful way to understand and manipulate spatial relationships in 3D space.

Our core mission is to provide an elegant and mathematically sound framework that bridges the gap between complex 3D vision tasks and a human-friendly algebraic structure.

## The Core Idea: What is F6?

In 3D graphics and robotics, spatial transforms are traditionally handled by cumbersome nested formulas or non-intuitive 4x4 homogeneous matrices. These methods, while functional, create a high barrier to entry and obscure the simple elegance of the underlying geometry.

**F6 is a new notation and a mathematical group** that represents a spatial transform as a simple, structured pair:

`{t, o}`

*   `t`: A vector representing the **translation**.
*   `o`: An abstract representation of the **orientation** (e.g., Euler angles, a quaternion, or another intuitive format).

The key innovation of F6 is that the complex **3x3 rotation matrix (`R`) is completely hidden from the user**. We provide robust, internal functions that seamlessly convert between the user-friendly orientation `o` and the internal rotation matrix `R`. This encapsulation is a cornerstone of the F6 philosophy:

*   **User-Friendly**: You interact with an intuitive concept of orientation, not a complex matrix.
*   **Mathematically Superior**: By encapsulating the `o <-> R` conversion, we create a clean, high-level algebraic system. All operations (`combine`, `inverse`, etc.) are defined on the `{t, o}` structure itself.

All operations, from transforming a coordinate system to locating a single point, exist within the F6 group. This creates a closed, self-consistent, and exceptionally elegant system.

For a deeper dive into the philosophy, please read [F6 Transform.md](F6 Transform.md).

## The Grand Vision: From a Single Point to a 3D Perception Mesh

This project demonstrates the power of the F6 philosophy through a clear, two-stage journey:

### Stage 1: Mastering the Single View (The "Back Projection" Foundation)

The first goal is to achieve high-precision 3D measurement from a single camera, even one with significant lens distortion. We accomplish this by integrating three novel techniques:

1.  **Self-Calibration**: An algorithm to precisely determine a camera's intrinsic focal length (`id`), laying the foundation for accurate measurement.
2.  **Ray Redirection**: A groundbreaking, non-parametric method for distortion correction. Instead of relying on a few polynomial coefficients, we build a high-resolution mapping grid that can model and correct any type of complex distortion (even "funhouse mirror" effects), far surpassing traditional methods.
3.  **Back Projection within F6**: We use the corrected 2D image points and the calibrated `id` to back-project them into 3D space. Crucially, the final 3D point is represented as an F6 transform, demonstrating the closure of the F6 group.

The result of this stage is a robust system that can take a pixel coordinate from a distorted image and transform it into a precise, F6-represented 3D world coordinate.

### Stage 2: Building the Network (The "Mesh" Vision)

The true power of F6 is revealed when we network multiple cameras. If two or more cameras can see a common reference object (e.g., a calibration target), we can use F6 group operations (`combine` and `inverse`) to calculate the exact spatial relationship (`F6` transform) between them.

By chaining these relationships, we can build a **3D Perception Mesh**: a network of cameras that are all spatially aware of each other. This creates a powerful, distributed 3D monitoring system where information from any camera can be seamlessly fused and translated into the perspective of any other camera in the network, or into a single global coordinate system.

## Project Structure

This repository is organized to reflect the two stages of our vision:

*   `f6transform/`: The core Python library, suitable for `pip install`. It contains the fundamental F6 group operations and the calibration/vision algorithms.
*   `examples/`: A series of scripts that demonstrate how to use the library.
    *   `1_single_camera_transform.py`: A complete walkthrough of Stage 1.
    *   `2_multi_camera_network.py`: A conceptual demonstration of Stage 2.
*   `scripts/`: Standalone tools for running calibration tasks.
*   `data/`: Sample data, including calibration images and output files.
*   `tests/`: Unit and integration tests to ensure correctness and stability.

## Getting Started

We invite you to explore this new way of thinking. Start by running the first example to see how a single camera can be empowered by the F6Transform system.

```bash
# (Setup instructions will be added here)
python examples/1_single_camera_transform.py
```
