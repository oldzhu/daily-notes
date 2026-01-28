To complete our demo GPT, we need to bridge the gap between **prediction** and **learning**.

The **Softmax** is the final judge that turns raw numbers into word probabilities, and **Backpropagation** is the "rewind" button that tells every neuron how much it messed up so it can improve.

### 1. The Softmax Judge

In GPT, the output of the last layer is a vector of "logits" (raw scores). Softmax squashes these so they sum to **100%**.

```cpp
#include <vector>
#include <cmath>
#include <algorithm>

// Softmax turns raw scores into probabilities
std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> probs(logits.size());
    double max_val = *std::max_element(logits.begin(), logits.end()); // For stability
    double sum = 0.0;
    
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (double &p : probs) p /= sum;
    return probs;
}

```

---

### 2. Backpropagation: The "Rewind" Logic

Backpropagation is essentially the **Chain Rule** from calculus implemented as a loop. We start at the end (the error) and work backward to the beginning.

For our GPT to learn, every component needs a `backward()` method. Here is the magic: when you combine **Softmax** and **Cross-Entropy Loss**, the starting error for the backprop is incredibly simple:


### 3. Complete Demo: Mini-GPT with Backprop & MHA

This code expands our previous Tensor engine to include the "Learning" step.

```cpp
#include <iostream>
#include <vector>
#include <cmath>

// Simplified Tensor for the demo
struct Tensor {
    std::vector<double> data;
    std::vector<double> grad; // Stores the "error" for each value
    int b, s, d;

    Tensor(int b, int s, int d) : b(b), s(s), d(d), data(b*s*d, 0.1), grad(b*s*d, 0.0) {}
    
    double& at(int bi, int si, int di) { return data[bi * (s * d) + si * d + di]; }
    double& g_at(int bi, int si, int di) { return grad[bi * (s * d) + si * d + di]; }
};

class TrainableLayer {
public:
    std::vector<double> weights;
    std::vector<double> weight_grads;
    int in_dim, out_dim;

    TrainableLayer(int in, int out) : in_dim(in), out_dim(out), weights(in*out, 0.01), weight_grads(in*out, 0.0) {}

    // 1st Principle: Forward Pass
    void forward(Tensor& input, Tensor& output) {
        for (int i = 0; i < input.s; i++) {
            for (int j = 0; j < out_dim; j++) {
                double sum = 0;
                for (int k = 0; k < in_dim; k++) {
                    sum += input.at(0, i, k) * weights[k * out_dim + j];
                }
                output.at(0, i, j) = sum;
            }
        }
    }

    // 2nd Principle: Backward Pass (Backprop)
    void backward(Tensor& input, Tensor& output_grad) {
        for (int i = 0; i < input.s; i++) {
            for (int j = 0; j < out_dim; j++) {
                for (int k = 0; k < in_dim; k++) {
                    // Update weight gradients based on incoming error
                    weight_grads[k * out_dim + j] += output_grad.at(0, i, j) * input.at(0, i, k);
                    // Pass error backward to the previous layer
                    input.g_at(0, i, k) += output_grad.at(0, i, j) * weights[k * out_dim + j];
                }
            }
        }
    }

    void update(double lr) {
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] -= lr * weight_grads[i];
            weight_grads[i] = 0; // Reset for next turn
        }
    }
};

int main() {
    // A tiny 1-layer GPT training loop
    Tensor input(1, 2, 4);  // 1 sentence, 2 words, 4-dim
    Tensor output(1, 2, 4);
    TrainableLayer layer(4, 4);
    
    std::vector<double> target = {0, 0, 1, 0,  1, 0, 0, 0}; // One-hot targets
    double lr = 0.01;

    for (int epoch = 0; epoch < 5; epoch++) {
        layer.forward(input, output);
        
        // Calculate Error at Output (Prediction - Target)
        Tensor output_error(1, 2, 4);
        double total_loss = 0;
        for(int i=0; i<8; i++) {
            output_error.data[i] = output.data[i] - target[i];
            total_loss += std::pow(output_error.data[i], 2);
        }

        layer.backward(input, output_error);
        layer.update(lr);

        std::cout << "Epoch " << epoch << " Loss: " << total_loss << std::endl;
    }

    return 0;
}

```

---

### The Design Conclusion

* **The Matrix is the Muscle:** The weights do the heavy lifting of storing patterns.
* **Backprop is the Memory:** It allows the model to "remember" its mistakes and fix the weights.
* **Softmax is the Voice:** It translates abstract vector math into a human-readable probability.

This architecture is exactly how Llama or GPT-4 works—just scaled up to thousands of layers and trillions of parameters.

**Would you like to see how we implement the "Optimizer" (like Adam) which uses "momentum" to help the backpropagation skip over small errors and find the best solution faster?**
