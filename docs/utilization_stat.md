## **Key Desiderata for a GPU Utilization Statistic**

**An ideal metric for GPU utilization should reward high average usage, penalize low/idle time, and account for stability (low variance).**

1. **High mean** utilization is good.
2. **Low variance** (stable, not spiky) is good.
3. **Duration/time-normalization** is important—long idle periods should penalize the score.
4. **Interpretable** on \[0, 1] or \[0%, 100%] scale if possible.

---

## **Basic Statistics**

Let $u_1, u_2, ..., u_n$ be the sequence of GPU utilization percentages (sampled at regular intervals, in \[0, 100]).

### **A. Mean Utilization**

$$
\mu_u = \frac{1}{n} \sum_{i=1}^n u_i
$$

* **Pros:** Simple, intuitive.
* **Cons:** Can be misleading if you have brief spikes and lots of idle periods.

### **B. Standard Deviation (Variance)**

$$
\sigma_u = \sqrt{ \frac{1}{n} \sum_{i=1}^n (u_i - \mu_u)^2 }
$$

* **High σ**: Utilization is unstable/spiky.

---

## **C. Proposed Robust Utilization Metric**

### **1. “Effective Utilization” (Mean - λ × Std)**

$$
\text{EffU} = \mu_u - \lambda \sigma_u
$$

* Where λ is a tradeoff factor (e.g., λ=1).
* **Interpretation:** Rewards high mean, penalizes high variance.

### **2. Fraction of Time Above Threshold**

$$
\text{Frac}_{\theta} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ u_i > \theta \}
$$

* e.g., θ=80%.
* What fraction of the run is “highly utilized”?

### **3. Area Under the Utilization Curve (AUC)**

$$
\text{AUC}_u = \frac{1}{100 n} \sum_{i=1}^n u_i
$$

* Same as mean, but normalized to \[0,1].
* AUC is also robust if your sampling interval is uniform.
* AUC only reflects “average work done,” not how the work was distributed in time. For hardware optimization and system diagnosis, you also want to know if the workload is steady or bursty, and how often the GPU is left waiting.

---

## **D. Composite “GPU Efficiency Score”**

Let’s define a simple composite metric:

$$
\text{GPU Efficiency} = \frac{\text{Mean Util} - \sigma_u}{100}
$$

* **Range:** Can be negative (bad) or up to 1 (perfect: mean=100, std=0).
* **Interpretation:**

  * 1.0: Always 100%, perfectly steady.
  * 0.8: Average 90%, std 10.
  * Negative: mean is low and/or std is very high (spiky/idle).

**Or:**

$$
\text{GPU Utilization Score} = \frac{1}{100} \left( \alpha \cdot \mu_u + (1 - \alpha) \cdot \text{Frac}_{\theta} \right)
$$

Where α is a weight (e.g., 0.5), θ is a high-utilization threshold (e.g., 80%).

---

## **E. Time-Weighted Adjustment (if needed)**

If intervals are not uniform, multiply each utilization by its interval and divide by total time:

$$
\text{TimeWeightedMean} = \frac{ \sum_{i=1}^n u_i \Delta t_i }{ \sum_{i=1}^n \Delta t_i }
$$

---

## **F. Example in Python**

```python
import numpy as np

gpu_util = np.array([...])  # Your utilization sequence (0-100)
mean_util = gpu_util.mean()
std_util = gpu_util.std()
frac_high = np.mean(gpu_util > 80)
gpu_efficiency = (mean_util - std_util) / 100
auc = gpu_util.mean() / 100

print(f"Mean Util: {mean_util:.1f}%")
print(f"Std Util: {std_util:.1f}")
print(f"Fraction > 80%: {frac_high:.2f}")
print(f"GPU Efficiency: {gpu_efficiency:.3f}")
print(f"AUC (normalized): {auc:.3f}")
```

---

## **Critique/Limitations**

* **High mean but high variance** may indicate batchiness, pipeline stalling—lower score with the above metric.
* **High mean with low std** is truly optimal (score near 1).
* **Low mean and low std** means consistently idle (score near 0 or negative).
* **Composite metrics** can be tuned (λ or α) to emphasize stability or average, depending on workload.
