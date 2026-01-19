Universal Approximation Theorem

To understand the **Universal Approximation Theorem**, you have to stop thinking of a neuron as just a "line" and start thinking of it as a **"Lego Brick."**

The magic of  is that the activation function transforms a boring straight line into a **shape** that can be stacked.

---

### 1. What does the Activation look like?

The activation function is indeed a function of the input (). In the 2D world, imagine  is the horizontal axis and  is the vertical.

* **ReLU (The "Corner"):** The most common today. It is 0 for any negative input and a straight line for positive. It looks like a **hinge**.
* **Sigmoid (The "Step"):** It looks like an **S-curve**. It transitions smoothly from 0 to 1.
* **Step Function (The "Cliff"):** The simplest version. It’s 0, then suddenly jumps to 1.

---

### 2. How to fit a Parabola (抛物线)

A single neuron can't be a parabola. But **two ReLUs** working together can approximate one!

1. **Neuron 1 (Left Side):** . This creates a line that goes *down* and then hits zero.
2. **Neuron 2 (Right Side):** . This creates a line that starts at zero and goes *up*.
3. **The Result:** If you add them together (), you get a **"V" shape**.

To make it look like a smooth parabola, you just add more neurons. Each neuron adds a new "segment" or "bend." With 10 neurons, your "V" becomes a multi-faceted bowl that looks perfectly smooth to the human eye.

---

### 3. How to fit a Circle (圆)

Fitting a circle in 2D ( inputs) is the ultimate test. To "carve" a circle out of the world, a neural network creates a **"Bump."**

1. **The Step:** One neuron () creates a "cliff" (everything on the left is 0, right is 1).
2. **The Slice:** A second neuron creates a cliff in the opposite direction. When you add them, you get a **ridge** (a raised strip).
3. **The Box:** Two more neurons create a ridge going the other way (top to bottom). Where the two ridges intersect, you get a **square bump**.
4. **The Circle:** If you add dozens of these "cliffs" at different angles (using weights  for  and ), you aren't building a square anymore—you are building a **polygon** with so many sides it becomes a circle.

> **The Core Logic:** The Universal Approximation Theorem says that if you have enough of these "bumps," you can tile them together like pixels on a screen to create **any shape**, no matter how complex.

---

### 4. Summary: The Recipe for Knowledge

* ****: Sets the **angle** and **position** of your "Lego brick."
* **Activation**: Turns that line into a **shape** (a hinge or a step).
* **The Layer**: **Sums** these shapes together to create a complex landscape.

In an LLM, the "shape" isn't a circle in 2D; it's a "concept" in 4,096-dimensional space. The model is essentially drawing a incredibly complex "map" of human thought using billions of these tiny bent lines.

Would you like to see how the **Gradient Descent** actually "pushes" these lines around until they perfectly wrap around a shape like a circle?

---

[Neural Networks and the Universal Approximation Theorem](https://www.youtube.com/watch?v=wen3221_3gU)
This video provides a very simple explanation of how neural networks act as universal approximators by combining simple "Lego-like" building blocks to fit any function.
