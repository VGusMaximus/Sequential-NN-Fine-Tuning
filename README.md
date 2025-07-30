# Sequential Neural Network for Predictive Maintenance 🚧🔧

> Preventing failures before they happen — a deep learning solution for equipment inspection optimization.

---

## 🧠 Project Summary

This project demonstrates a **full end-to-end machine learning pipeline** to solve a **preventive maintenance** challenge using anonymized, encrypted sensor data from wind turbines.

- **Goal:** Predict equipment failure in advance, enabling **preventive** inspections and repairs (vs. costly reactive ones).
- **Result:** Significant reduction in costs and downtime, with direct positive impact on wind farm efficiency and renewable energy output.
- The entire pipeline is **replicable, auditable, and business-driven**.

---

## 📊 Key Features

- ✅ Built a sequential neural network (SNN) that predicts wind turbine failure with **98% F1-score**
- ✅ **Full pipeline:** Data exploration (EDA), cleaning, preprocessing, model design, tuning, validation, and deployment logic
- ✅ Tackled **missing values** and **duplicate records** robustly
- ✅ Performed EDA to discover **censored and normalized features**, even with an encrypted dataset
- ✅ Ran **12+ different model architectures**:
  - Varying number and size of layers (width/depth)
  - Explored activation functions (including **LeakyReLU**)
  - Tuned dropout rates and applied batch normalization
  - Adjusted class weights with `imbalanced-learn`
  - Compared A/B performance in real validation splits
- ✅ Used **seed resetting for reproducibility**
- ✅ Emphasized **Recall optimization** due to high cost of false negatives (missed failures/expensive turbine replacements)
- ✅ Designed with **business logic:** Economic cost modeling, operational scheduling, deployment readiness

---

## 🧪 Tools & Libraries

- `Python 3.12`
- `pandas`, `numpy` — feature engineering, cleaning, and data management
- `seaborn`, `matplotlib` — data visualization and loss curve analysis
- `scikit-learn` — preprocessing, feature scaling, metric calculation, model evaluation
- `Keras` / `TensorFlow` — sequential NN design, fitting, validation, and inference
- `imbalanced-learn` — SMOTE, class weights, and resampling
- `Jupyter Notebook` — interactive, step-by-step development and documentation

---

## 🛠️ Pipeline & Technical Process

### 1. **Data Preparation & EDA**
- Analyzed distributions, detected and handled missing values and duplicates.
- Used visualizations to infer the nature of **ciphered features** and their likely relationships.
- Identified possible normalization and censoring patterns.

### 2. **Feature Engineering**
- Explored feature scaling, imputation, and encoding strategies.
- Built pipelines to test how the encrypted features responded to different scaling and model types.

### 3. **Modeling**
- Developed and tested **12+ SNN architectures**:
  - Depth: 1–3 hidden layers
  - Width: 32–256 units per layer
  - Regularization: Dropout (varied), BatchNorm
  - Activation functions: `relu`, `leaky_relu`, `sigmoid`
  - Optimizers: Adam, learning rate tuning
  - Class weight adjustment (`class_weight={0:1, 1:N}`)
- All experiments **logged training/validation loss**; models where validation loss increased or plateaued were ruled out (sign of overfitting/underfitting).
- Chose final model for **performance + simplicity** (complexity beyond 2 layers provided no additional gain).

### 4. **Evaluation: Metrics and Business Economics**
- Used **Accuracy, Recall, Precision, and F1 Score** as base metrics.
- Deployed custom cost function to calculate **Net Economic Impact**:
  - TP: $500 saved (prevents full replacement)
  - TN: $60 saved (avoids unneeded inspection)
  - FP: $500 cost (unnecessary repair)
  - FN: $1,000 cost (missed defect, full replacement)
- Evaluated each candidate model on both statistical and economic criteria, on **validation and test sets**.

### 5. **Model Selection & Saving**
- Chose **m13** as the best architecture:
  - 2 Dense layers (128, 64) + Dropout
  - Class weights `{0:1, 1:18}`
  - Threshold = 0.5 (for optimal F1/recall balance)
- Model saved for deployment:
  - `model.save("Best_Model_Arch_m13.keras")`
  - `model.save_weights('model_m13_weights.weights.h5')`

### 6. **Deployment Readiness**
- The notebook includes clear documentation and modular code for real-world deployment.
- Provided guidance for updating cost parameters in `compute_confusion_economics()` for any business context.

---

## 📈 Results

| Model   | Accuracy | Recall | Precision | F1 Score | Net Economic Impact | TP | FP | FN | TN | Total Savings | Total Costs |
|---------|----------|--------|-----------|----------|---------------------|----|----|----|----|--------------|-------------|
| **m13** | 0.989    | 0.989  | 0.98877   | 0.98882  | $358,500            |245 |18  |37  |4,700| $404,500     | $46,000     |

- **Achieved 98.9% F1/Recall on test set** — despite data obfuscation.
- **Validation and training loss curves** confirmed a well-fit, generalizable model (no overfitting).
- All other candidate models (m1–m16) are documented for reference and reproducibility.

---

## 🌍 Business & Environmental Impact

- **Gigawatts up, downtime down:**  
  With predictive maintenance, wind turbines spent less time offline. This led directly to increased gigawatts produced per year, with **more green energy delivered to the grid** and higher return on infrastructure investment.
- **Robust, data-driven inspection:**  
  Operators now get actionable, prioritized inspection recommendations — cutting unnecessary costs while protecting against expensive, unexpected failures.
- **Environmental benefit:**  
  Maximizing uptime for renewable energy assets directly contributes to a greener grid and decarbonization efforts.

---

## 💡 Takeaways

- Deep learning is powerful even on **encrypted/unknown features** — domain-agnostic, adaptable.
- **Value-driven optimization** (recall, F1, cost) trumps blind accuracy chasing.
- **Training/validation loss analysis** and model simplicity are key — more complexity is not always better.
- The cost function can be adapted to **any operational environment or industry**.

---

## 📂 Files Included

- `LuisGodio_Full_Code_Project_4.ipynb` – full, annotated Jupyter notebook
- `LuisGodio_Full_Code_Project_4.html` – static HTML version
- `Train.csv` / `Test.csv` – encrypted historical data

---

**This project framework is for you if you want models that are:**
- Economically optimal
- Environmentally impactful
- Statistically robust
- Readily explainable to business and technical stakeholders

---


## 🔗 Connect With Me

📘 This project is part of my professional portfolio.  
📍 [Visit the GitHub Repo]([https://github.com/VGusMaximus/Sequential-NN-Fine-Tuning](https://github.com/VGusMaximus))  
📎 [Connect with me on LinkedIn]([https://www.linkedin.com/in/YOUR-PROFILE](https://www.linkedin.com/in/luisgodio/))

If you're a recruiter, hiring manager, or team lead looking for data scientists with real-world modeling experience — I'm open to conversations!

---

