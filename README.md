# Reinforcement Learning for Inventory Management Optimization

## Project Overview

This project explores the application of Deep Reinforcement Learning (RL) to optimize inventory management in a retail setting. The primary objective is to minimize cumulative inventory costs—including holding, stockout, and uncertainty penalties—by learning optimal ordering policies from historical sales data. The approach leverages a Deep Q-Network (DQN) framework, enhanced with uncertainty quantification using Dirichlet probability density functions (PDFs) and multinomial opinions. This enables the agent to make more robust and adaptive decisions in the face of real-world demand uncertainty and partial observability[1][2].

Key contributions include:
- Modeling inventory management as a Markov Decision Process (MDP) with sequential decision-making.
- Implementing and comparing multiple DQN-based models with various hyperparameters.
- Integrating uncertainty metrics (vacuity, dissonance, entropy) into the reward function for improved decision-making under uncertainty.
- Conducting sensitivity analyses on critical environment parameters.

For a detailed description of the methodology and results, see the [project paper](Kim_CS5914_P2_Report.pdf).

---

## Dataset Source

- **Dataset:** [Retail Sales Data (Kaggle)](https://www.kaggle.com/datasets/berkayalan/retail-sales-data/data)
- **Description:** Historical sales data from a Turkish retail company, spanning 2017–2019. The dataset includes daily records for multiple products and stores, with columns such as `store_id`, `product_id`, `date`, `sales`, `revenue`, `stock`, `price`, and promotion details[1][2].
- **Preprocessing:**
  - Dropped unnecessary columns (store and promotion info) to focus on core inventory features.
  - Removed rows with missing values (3–6% of the data, mainly in late 2019).
  - Aggregated data by product and date to track daily sales and stock.
  - Split into training (70%, up to 2019-06-30), validation (15%, 2019-07-01 to 2019-08-31), and test sets (15%, from 2019-09-01)[1][2].

---

## Model Architecture and Approach

**State Representation:**  
Each state is a vector `[stock, sales, price]` for each product on a given day.

**Action Space:**  
Order quantities for each product, bounded by `max_order`.

**Reward Function:**  
A composite function balancing sales revenue, stockout penalty, overstock penalty, holding costs, and uncertainty penalties derived from vacuity, dissonance, and entropy metrics[1][2].

**Algorithm:**  
- **Deep Q-Network (DQN):**  
  - Two hidden layers (initially 128 units, later reduced to 64 with dropout to address overfitting).
  - Epsilon-greedy exploration with decaying epsilon.
  - Experience replay and Adam optimizer.
  - Uncertainty-aware action selection via Dirichlet PDF and multinomial opinion transformation.

**Uncertainty Quantification:**  
- **Vacuity:** Lack of evidence for any action.
- **Dissonance:** Conflict among action beliefs.
- **Entropy:** Overall uncertainty in action probabilities.
- These metrics are incorporated into the reward to penalize uncertain or indecisive policies[1][2].

---

## Results

### Model Comparison

Four DQN models were trained with different hyperparameters:

| Model | Learning Rate | Epsilon Decay | Gamma | Validation Reward |
|-------|---------------|--------------|-------|------------------|
| 1     | 0.001         | 0.995        | 0.99  | 122.11           |
| 2     | 0.005         | 0.998        | 0.95  | 142.15           |
| 3     | 0.0005        | 0.990        | 0.90  | 132.20           |
| 4     | 0.002         | 0.997        | 0.98  | **142.47**        |

- **Best Model:** Model 4, with a validation reward of 142.47, achieved the best balance between exploration and exploitation and demonstrated the most stable convergence[1].

### Effectiveness and Efficiency

- **Effectiveness:** Measured by total reward on validation data (e.g., Model 4: 102.12).
- **Efficiency:** Mean loss during training (e.g., Model 4: 1384.90)[1][2].

### Uncertainty Metrics

- **Vacuity** remained around 0.67, indicating persistent uncertainty.
- **Entropy** hovered near 1.0, suggesting high overall uncertainty.
- **Dissonance** was moderate (~0.3), reflecting some internal conflict in action selection.
- The model struggled to reduce uncertainty, highlighting the challenge of learning confident policies in highly stochastic environments[1].

### Sensitivity Analysis

#### Max Order

- Tested values: 30, 50, 100.
- **Best performance** at max_order = 30; higher values led to overstock and increased holding costs, reducing overall reward.

#### Max Stock

- Tested values: 500, 1000, 2000.
- Results varied by model: some performed better with higher max stock, others with lower, indicating that optimal stock limits are sensitive to model hyperparameters and should be carefully tuned for each deployment scenario[1][2].

---

## Result Visualizations

> **[Insert result plots here: validation rewards, training rewards, loss curves, uncertainty metrics, sensitivity analysis, etc.]**  
> _Add figures (e.g., Fig. 2–10 from the report) to illustrate model performance and sensitivity analysis._

---

## Project Paper

For a comprehensive explanation of the methodology, code, and findings, see the [project paper (Kim_CS5914_P2_Report.pdf)](Kim_CS5914_P2_Report.pdf).

---

## How to Run

1. Clone this repository.
2. Download the [retail sales dataset](https://www.kaggle.com/datasets/berkayalan/retail-sales-data/data) and place it in the appropriate directory.
3. Install dependencies (`requirements.txt`).
4. Run `main.py` to preprocess data, train models, and generate results.

---

## References

- [Retail Sales Data (Kaggle)](https://www.kaggle.com/datasets/berkayalan/retail-sales-data/data)
- [Project Paper: Kim_CS5914_P2_Report.pdf](Kim_CS5914_P2_Report.pdf)

---

## TODO

- Add result plots and figures.
- Upload final trained models.
- Extend to multi-product and multi-store scenarios.

---

*For questions or contributions, please open an issue or pull request.*

---

### slrlModels2
<img src="https://github.com/user-attachments/assets/91469233-758e-4482-b331-cf7cd68a122f" width="50%">
<img src="https://github.com/user-attachments/assets/44b88555-0bfc-4812-8d93-9dd233a4e5b6" width="50%">
<img src="https://github.com/user-attachments/assets/0d2d1158-3c3b-4cb3-bfbd-0c99c71e9536" width="50%">
<img src="https://github.com/user-attachments/assets/cd9bb0b8-6cc6-4b65-8e23-0529cbb448d7" width="50%">

### Sensitive Analysis
<img src="https://github.com/user-attachments/assets/7b0268d2-5740-4eaf-916c-d1b22a45e216" width="50%">
<img src="https://github.com/user-attachments/assets/afaf64ec-f33e-4ed1-a607-b381e8589789" width="50%">
<img src="https://github.com/user-attachments/assets/eb1a1598-8aa5-4772-a67c-f4c47738104b" width="50%">
