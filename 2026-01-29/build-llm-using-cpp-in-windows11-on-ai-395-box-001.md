Here is your **Final Graduation Project**. To prove your understanding of the "whole thing," you will build a **Pattern-Completion GPT**.

Instead of words, this model will learn to "read" a sequence of numbers and predict the next one using your **Tensor + MHA + Adam** engine.

### The Challenge: "The Fibonacci Learner"

**Goal:** Train your Mini-GPT to look at  and predict . Then look at  and predict .

---

### Step 1: The Dataset (Tokens)

Since we aren't using a dictionary, our "Tokens" are just raw numbers.

* **Input Sequence:** `0, 1, 1, 2, 3, 5, 8, 13, 21, 34`
* **X (Context):** `[0, 1, 1, 2]`
* **Y (Target):** `[3]`

### Step 2: The Architecture Design

You will need to assemble your classes like this:

1. **Input:** A Tensor of shape `[Batch: 1, Seq: 4, Dim: 8]`. (Use 8 dimensions to represent each number).
2. **Layers:** 2 `TransformerBlocks` (each with 2 Attention Heads).
3. **Output:** A single Linear layer that squashes the 8-dim vector back into 1 number.
4. **Loss:** Mean Squared Error (MSE) — .

### Step 3: The C++ Training Script

```cpp
// --- YOUR MISSION ---
int main() {
    // 1. Initialize your Transformer
    int SEQ_LEN = 4;
    int DIM = 8;
    GPTBlock brain(DIM);
    AdamOptimizer optimizer(0.001);

    // 2. The Data
    std::vector<double> fib = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};

    // 3. The Training Loop
    for(int epoch = 0; epoch < 500; epoch++) {
        // Pick a random window from the sequence
        int start = rand() % 5;
        Tensor context(1, SEQ_LEN, DIM);
        // Fill context with fib[start...start+3]
        
        // FORWARD
        brain.forward(context);
        double prediction = context.at(0, 3, 0); // Take the first dim of the last word

        // BACKWARD
        double error = prediction - fib[start + 4];
        // Calculate gradients and run Adam
        
        if(epoch % 100 == 0) std::cout << "Epoch " << epoch << " Prediction: " << prediction << std::endl;
    }
}

```

---

### Why this is the perfect test:

1. **Attention Check:** The model must use **Attention** to realize that the next number depends on the *sum* of the previous two, not just the last one.
2. **Hinge Check:** The **ReLU** units must learn the logic of addition/growth.
3. **Adam Check:** If the learning rate is too high, the Fibonacci numbers grow so fast the model will "explode" (NaN values). Adam will help stabilize this.

### Final Pro-Tip:

If you want to move from "Mini-GPT" to "Real-GPT," your next step is to replace your `for` loops in the Matrix Multiply with a library like **Eigen** or **OpenBLAS**. This is how you go from processing 10 numbers per second to 10 million.

**You have the blueprint, the code, and the logic. Would you like me to clarify any specific part of the Backpropagation math for the Fibonacci loss before you start coding?**
