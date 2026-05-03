# Synthetic Tabular Data Generation for Privacy-Preserving Machine Learning

A deep learning project demonstrating privacy-preserving synthetic tabular data generation using Conditional Tabular GANs (CTGAN).

---

# Project Overview

This project explores the use of CTGAN to generate realistic synthetic census data while preserving privacy and maintaining machine learning utility.

The model was trained and evaluated using the UCI Adult Census Income dataset and tested across three major dimensions:

* Fidelity
* Utility
* Privacy

The project also includes deployment through a Gradio web interface running on Google Colab.

---

# Objectives

The main objectives of this project are:

* Generate realistic synthetic tabular data
* Preserve statistical properties of the original dataset
* Maintain machine learning performance on synthetic data
* Reduce privacy leakage and memorization risks
* Deploy an interactive synthetic data generation interface

---

# Dataset

## UCI Adult Census Income Dataset

Dataset Characteristics:

* 48,842 rows
* 15 features
* Numerical and categorical columns
* Binary income classification target

Features include:

* Age
* Workclass
* Education
* Occupation
* Marital Status
* Hours Per Week
* Capital Gain
* Capital Loss
* Native Country
* Income Label

---

# Technologies Used

* Python
* CTGAN
* SDV (Synthetic Data Vault)
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Gradio
* Google Colab

---

# Model Architecture

The project uses Conditional Tabular GAN (CTGAN), which extends traditional GANs for tabular datasets.

Key techniques:

* Conditional Training
* Mode-Specific Normalization
* WGAN-GP Loss Function
* Variational Gaussian Mixture Encoding

---

# Hyperparameter Configurations

| Configuration | Epochs | Embedding Dimension |
| ------------- | ------ | ------------------- |
| Baseline      | 100    | 128                 |
| Capacity      | 100    | 256                 |
| Long          | 200    | 256                 |
| Stable        | 200    | 128                 |

---

# Best Model Results

## Long Configuration

| Metric                        | Result |
| ----------------------------- | ------ |
| SDV Quality Score             | 82.4%  |
| TSTR Accuracy                 | 82.1%  |
| Utility Retention             | 96.3%  |
| Macro F1 Score                | 0.731  |
| AUC-ROC                       | 0.881  |
| Membership Inference Accuracy | 53.1%  |

---

# Key Findings

* CTGAN successfully generated high-quality synthetic tabular data.
* Embedding dimension had a stronger impact than training duration.
* Synthetic data retained strong machine learning utility.
* Privacy evaluation showed no meaningful memorization of training records.
* Zero-inflated columns such as capital-gain and capital-loss remained difficult to model accurately.

---

# Privacy Evaluation

The project implemented a Membership Inference Attack (MIA) to test privacy risks.

Results:

* MIA Accuracy: 53.1%
* Safe Threshold: Below 55%

This indicates that the model did not significantly memorize real training records.

---

# Deployment

The final model was deployed using Gradio inside Google Colab.

Features:

* Configuration selector
* Synthetic row generation
* SDV Quality Report option
* CSV download support

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-github-repository-link>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Open the Notebook

Open the notebook in Google Colab or Jupyter Notebook.

## 4. Enable GPU

Google Colab:
Runtime → Change Runtime Type → T4 GPU

## 5. Run All Cells

```bash
Ctrl + F9
```

## 6. Launch Gradio

After execution, a public `.gradio.live` link will be generated.

---

# Project Structure

```bash
├── data/
├── notebooks/
├── models/
├── outputs/
├── figures/
├── app/
├── README.md
└── requirements.txt
```

---

# Future Improvements

* Integrate DP-CTGAN for formal differential privacy
* Improve handling of zero-inflated columns
* Hyperparameter optimization using Optuna
* Deploy using Hugging Face Spaces
* Test on healthcare datasets such as MIMIC-III

---

# Author

MD Rakib Hossain

University of Roehampton
School of Arts, Humanities and Social Science
London, United Kingdom

---

# Academic Integrity Statement

AI-assisted tools were used only for improving clarity, organization, and understanding of technical concepts. All coding, experiments, evaluations, and analytical conclusions were independently completed by the author.

---

# License

This project is for academic and educational purposes only.
