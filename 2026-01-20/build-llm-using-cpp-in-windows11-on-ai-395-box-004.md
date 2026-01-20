To calculate the parameters of a **7B (7 Billion)** model, we follow the "gold standard" architecture (like Llama 2 or Llama 3). We assume a **Dimension () of 4096** and **32 Layers ()**.

Everything in the model is a matrix () or a vector (). Here is how the "neurons" add up to 7 billion parameters.

---

### 1. The Embedding Layer (The Dictionary)

The model needs to store a vector for every word in its vocabulary ().

* **Formula:** 
* **Calculation:** 

---

### 2. The Attention Block (The Relationship Processor)

In each layer, we have four main weight matrices:  (to create signals) and  (to combine them).

* **Formula:** 
* **Calculation:**  (per layer)

---

### 3. The MLP Block (The Knowledge Bank)

This is where the most "neurons" live. In modern models (using SwiGLU), we use **three** matrices:  and . We expand the dimension to  instead of the traditional  for better performance.

* **Formula:** 
* **Calculation:**  (per layer)

---

### 4. The Grand Total (Summing the Layers)

Now we multiply the layers by the number of floors in our "factory" ():

1. **One Layer Total:** 
2. **All 32 Layers:** 
3. **Add Embedding:** 
4. **Final Touches:** Add LayerNorms, position encodings, and the final output layer...

* **Result:** 

---

### 5. Summary Table: Where do the "Weights" go?

| Component | Matrix Math | Param Count (Approx) | Role |
| --- | --- | --- | --- |
| **Embedding** |  | 131,000,000 | Storing "Word Identity" |
| **Attention** |  | 2,150,000,000 | Storing "Context/Logic" |
| **MLP** |  | 4,320,000,000 | Storing "World Knowledge" |
| **Total** |  | **~6.6 - 7.0 Billion** |  |

---

### 6. Where is the Bias ()?

You asked about bias. In many modern 7B models (like Llama), **bias is set to zero (False)** in almost all these matrices to save memory and speed up computation.

* **Calculation:** If  is False, we just do .
* **Impact:** This saves about **0.01%** of the parameters, but in a 7B model, that’s millions of numbers!

### Final Insight: The  Rule

Notice that if you double the **Dimension ()** from 4096 to 8192, the model size doesn't double—it **quadruples** (). This is why making models "wider" is much more expensive than making them "deeper" (more layers).

**Would you like to know how "Quantization" lets us squeeze these 7 Billion parameters (usually 14GB of data) into a small 4GB file so they can run on a phone?**
