# Dual-Primal_HingeLoss_SVM_FromScratch

This project implements a **Support Vector Machine (SVM) from scratch** using the **hinge loss function**, supporting two training formulations:

- **Primal SVM (Linear)** using gradient-based optimization
- **Dual SVM (Kernelized with RBF)** also trained manually, without using QP solvers

The goal is to fully understand and build both types of SVMs using only fundamental tools like NumPy — no black-box SVM functions or quadratic programming libraries were used.

---

## 🧠 Motivation

Support Vector Machines are powerful classifiers with a solid mathematical foundation. This project aims to:

- Explore how SVMs work under both **primal** and **dual** formulations
- Understand and apply the **hinge loss function**
- Implement RBF kernel logic manually in the dual form
- Evaluate the models on real-world data

---

## 📊 Dataset

**Banknote Authentication Dataset**  
Features:
- Variance
- Skewness
- Curtosis
- Entropy  
Target:
- Class (0 = Forged, 1 = Authentic)

Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

---

## 🔧 Implementation Highlights

- Manual gradient-based optimization for primal SVM (linear kernel)
- Custom dual formulation with RBF kernel (no QP library used)
- From-scratch hinge loss and margin computation
- Visualization using PCA and 3D plots
- Comparison with `sklearn.svm.SVC` for benchmarking

---

## 🔍 Hinge Loss

The hinge loss function used is:

\[
L = \sum_{i=1}^n \max(0, 1 - y_i(w^T x_i + b)) + \lambda ||w||^2
\]

For dual SVM with an RBF kernel:

\[
K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
\]

---

## 📈 Results & Evaluation

- Model accuracy on test data
- Loss curve progression
- Visual boundary comparisons
- PCA-based feature projection

---

## 🛠 Requirements

```bash
pip install numpy pandas matplotlib scikit-learn seaborn plotly
