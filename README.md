# Synthetic Tabular Data Generation for Privacy-Preserving ML

**Author:** MD Rakib Hossain | **Student ID:** A00057300  
**Module:** Deep Learning Applications, University of Roehampton  
**Dataset:** [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult)  
**Live Demo:** [Google Colab](https://colab.research.google.com/drive/14WhUk9y-gxgHtt5lrufMhm71bK862kVr?usp=sharing)

---

## What This Project Does

This project trains an AI model that can generate **fake but realistic census data**. The fake data looks and behaves like real data, but it does not contain information about any real person. This is called **synthetic data**, and it is useful when you need to share data for research without breaking privacy laws like GDPR.

The model used is called **CTGAN** (Conditional Tabular GAN), which is a type of deep learning model designed specifically for tables with a mix of numbers and categories.

---

## The Problem It Solves

Sharing personal data is becoming harder under modern privacy laws. But researchers and developers still need data to build things. Synthetic data is one of the best solutions: you train a model on real data, then share only what the model generates. The generated records carry the same statistical patterns as the original without exposing real individuals.

---

## Project Structure

```
.
├── A00057300_Md_Rakib_Hossain_Milestone_4_Final.ipynb   # Main notebook (all milestones)
├── README.md                                             # This file
├── requirements.txt                                      # Python dependencies
└── docs/
    └── MD_Rakib_Hossain_A00057300_Milestone4_Final.docx # Full IEEE-format report
```

---

## Milestones

| Milestone | What Was Done |
|-----------|---------------|
| **2** | Loaded and cleaned the dataset. Defined and trained four CTGAN configurations. |
| **3** | Tested the best model on held-out data using five quality metrics. Deployed a web interface using Gradio. |
| **4** | Added formal SDV quality scoring, privacy evaluation (Membership Inference Attack), cross-partition stability test, and an enhanced web interface. |

---

## The Model

CTGAN is trained on the **UCI Adult Census Income** dataset, which has 48,842 rows describing people from the 1994 US Census. After cleaning, 45,222 rows remain. The dataset has 14 features (age, occupation, education, etc.) and one target label (whether someone earns more than $50,000 a year).

Four configurations were trained and compared:

| Config | Epochs | Embedding Size | SDV Quality | TSTR Accuracy |
|--------|--------|----------------|-------------|---------------|
| Baseline | 100 | 128 (default) | 78.9% | 79.4% |
| Capacity | 100 | 256 | 80.6% | 81.0% |
| **Long (Best)** | **200** | **256** | **82.4%** | **81.8%** |
| Stable | 200 | 128 | 79.8% | 80.7% |

**Winner: Long configuration** -- 200 epochs, embedding dimension 256.

---

## Key Results

| What We Measured | Result |
|------------------|--------|
| SDV Quality Score | 82.4% |
| TSTR Accuracy (trained on synthetic, tested on real) | 81.8% |
| Real-data baseline accuracy | 85.3% |
| Utility Retention | 95.9% |
| Macro F1 Score | 0.700 |
| AUC-ROC | 0.881 |
| Membership Inference Attack accuracy | 53.1% (safe -- below 55% threshold) |
| Income class deviation | 7.5 percentage points |

In plain terms: a classifier trained **only on fake data** achieves 95.9% of the performance of one trained on real data. The privacy test shows the model is not memorising real records.

---

## How to Run It

### Option 1 -- Google Colab (Recommended, no setup needed)

1. Open the [Colab link](https://colab.research.google.com/drive/14WhUk9y-gxgHtt5lrufMhm71bK862kVr?usp=sharing)
2. Go to **Runtime > Change runtime type > T4 GPU**
3. Press **Ctrl+F9** to run all cells
4. Wait for training to finish (about 2-3 hours total for all four models)
5. Copy the `.gradio.live` URL that appears at the end and open it in your browser

### Option 2 -- Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/ctgan-synthetic-data.git
cd ctgan-synthetic-data

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook A00057300_Md_Rakib_Hossain_Milestone_4_Final.ipynb
```

---

## Using the Web Interface

Once the notebook is running, a Gradio interface launches automatically:

1. **Pick a model** from the dropdown (Long is the best one)
2. **Choose how many rows** to generate using the slider (10 to 5,000)
3. **Optional:** tick "Run SDV Quality Report" to get a live fidelity score (adds about 10 seconds)
4. Click **Generate**
5. Browse the 20-row preview or **download the full CSV**

---

## Requirements

```
ucimlrepo
pandas
sdv
sdmetrics
scikit-learn
scipy
gradio
numpy
matplotlib
seaborn
```

Install everything with:

```bash
pip install ucimlrepo pandas sdv sdmetrics scikit-learn scipy gradio numpy matplotlib seaborn
```

---

## What the Model Cannot Do

- **It has no formal privacy guarantee.** The Membership Inference Attack test shows the model is safe under a basic black-box attack, but a stronger attacker with access to the model weights could still extract some information.
- **Capital-gain and capital-loss are not well reproduced.** About 92% of people in the dataset have exactly zero for both of these. The model struggles with this kind of zero-heavy distribution (mean error: 35.6% for capital-gain).
- **The 76/24 income class ratio is not perfectly preserved.** The synthetic data over-represents the lower income class by about 7.5 percentage points, which slightly hurts minority-class prediction.

---

## Future Work

1. Add **differential privacy** (DP-CTGAN) for formal privacy guarantees
2. Handle zero-inflated columns with a two-part model (zero vs. positive tail)
3. Automate hyperparameter search with **Optuna**
4. Test on a healthcare dataset like MIMIC-III
5. Deploy permanently on Hugging Face Spaces

---

## Dataset Citation

Kohavi, R. (1996). Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree hybrid. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*. UCI Machine Learning Repository, ID 2.

## Model Citation

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *Advances in Neural Information Processing Systems*, 32.

---

## Acknowledgements

- SDV library by the MIT Data to AI Lab
- UCI Machine Learning Repository for the Adult Census Income dataset
- University of Roehampton, Deep Learning Applications module (CMP020L016)
