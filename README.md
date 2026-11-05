# Adult-to-Neonatal Brain Lesion Segmentation Using Domain Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning research framework investigating **Unsupervised Domain Adaptation (UDA)** and **Representation Learning** to bridge the severe domain gap between **adult acute ischemic stroke MRI** and **neonatal Hypoxic-Ischemic Encephalopathy (HIE) MRI**.

---

## Table of Contents
1. [Abstract & Overview](#1-abstract--project-overview)
2. [Problem Statement & Motivation](#2-problem-statement--clinical-motivation)
3. [Research Objectives](#3-research-objectives)
4. [Datasets](#4-datasets)
5. [Methodology & Processing Pipeline](#5-methodology--processing-pipeline)
6. [Network Architectures](#6-network-architectures)
   - [Attention U-Net](#61-attention-u-net)
   - [Domain-Adversarial Neural Network (DANN)](#62-domain-adversarial-neural-network-dann)
7. [The Modality Inversion Breakthrough](#7-the-modality-inversion-breakthrough)
8. [Experiment Roadmap & Status](#8-experiment-roadmap--status)
9. [Quantitative Results & Benchmark Comparisons](#9-quantitative-results--benchmark-comparisons)
   - [Cohort-Wide Benchmark](#91-cohort-wide-benchmark-n4-held-out-cohort)
   - [Granular Per-Subject Evaluation](#92-granular-per-subject-breakdown)
   - [MGHNICU_001 Small Punctate Lesion Autopsy](#93-mghnicu_001-small-punctate-lesion-autopsy)
10. [Technology Stack](#10-technology-stack)
11. [Repository Architecture & Directory Structure](#11-repository-architecture--directory-structure)
12. [Installation & Setup](#12-installation--setup)
13. [Interactive Web Frontend (Streamlit)](#13-interactive-web-frontend-streamlit)
14. [Evaluation & Reproduction Instructions](#14-evaluation--reproduction-instructions)
15. [Limitations](#15-limitations)
16. [Future Work](#16-future-work)
17. [References & Acknowledgements](#17-references--acknowledgements)

---

## 1. Abstract & Project Overview

Automated segmentation of ischemic brain lesions from Magnetic Resonance Imaging (MRI) is vital for rapid clinical intervention and outcome prognosis. While large annotated benchmarks exist for adult acute ischemic stroke (e.g., ISLES 2022), neonatal neuroimaging datasets (e.g., BONBID-HIE 2023) are constrained by limited cohort sizes and complex morphological phenotypes.

Direct zero-shot transfer of adult models to neonatal brains fails dramatically (initial Dice $\approx 0.1451$) due to significant domain shifts:
1. **Modality Mismatch:** Adult stroke protocols rely on Diffusion-Weighted Imaging (DWI, where infarctions are hyperintense), Apparent Diffusion Coefficient (ADC, hypointense), and FLAIR (hyperintense). In contrast, neonatal protocols frequently supply only ADC and age-standardized Z-score ADC ($Z$-ADC), lacking true DWI and FLAIR sequences.
2. **Anatomical Scale Discrepancy:** Neonatal brain volume is approximately one-third to one-half that of an adult, resulting in severe receptive-field mismatch and excessive zero-padding.
3. **Pathophysiological Heterogeneity:** Adult stroke manifests predominantly as focal, territorial infarcts with severe diffusion restriction, whereas neonatal HIE presents as multifocal, diffuse, or punctate subcortical injuries with subtle contrast shifts.

This repository implements a systematic investigation into **modality representation mapping**, **domain-adversarial learning (DANN)**, **spatial field-of-view scale normalization**, and **native two-channel modeling** to solve cross-cohort lesion segmentation.

---

## 2. Problem Statement & Clinical Motivation

* **High Neonatal Morbidity:** Hypoxic-Ischemic Encephalopathy occurs in 1 to 3 per 1,000 live births in high-income countries and up to 26 per 1,000 in low-resource settings, leading to cerebral palsy, epilepsy, and cognitive impairment.
* **Scarcity of Expert Annotations:** Delineating millimeter-scale neonatal brain lesions across anisotropic 3D volumes is labor-intensive and requires specialized pediatric neuroradiologists.
* **The Domain Transfer Dilemma:** Deep neural networks trained on adult stroke MRI cannot be naively deployed on neonates. Overcoming this barrier without requiring extensive manual neonatal annotations is the core challenge addressed by this project.

---

## 3. Research Objectives

1. **Quantify Zero-Shot Domain Transfer Degradation:** Establish baseline transfer metrics from adult ISLES 2022 models to neonatal BONBID-HIE validation scans.
2. **Solve Modality Inversion:** Mathematically map standardized neonatal diffusion channels to align with the perceptual expectations of adult-trained convolution kernels.
3. **Adversarial Domain Alignment:** Implement a Domain-Adversarial Neural Network (DANN) with a Gradient Reversal Layer (GRL) to align bottleneck feature representations across adult and neonatal domains.
4. **Field-of-View & Scale Normalization (E4):** Formulate automated brain bounding-box cropping and spatial resizing to resolve anatomical brain size discrepancies.
5. **Native Multi-Channel Architecture (E2):** Engineer and train a dedicated two-channel $[ADC, Z\text{-}ADC]$ Attention U-Net on neonatal cohorts to establish the definitive supervised target-domain benchmark.

---

## 4. Datasets

### Source Domain: ISLES 2022 (Adult Acute Ischemic Stroke)
* **Cohort Size:** 250 adult acute ischemic stroke patients.
* **Input Modalities:** 3 co-registered sequences:
  - DWI (Diffusion-Weighted Imaging)
  - ADC (Apparent Diffusion Coefficient)
  - FLAIR (Fluid-Attenuated Inversion Recovery)
* **Annotations:** Expert binary ground-truth segmentation masks of acute ischemic infarct core.
* **Standard Resolution:** Resampled to isotropic $2.0 \times 2.0 \times 2.0\text{ mm}$ spacing.

### Target Domain: BONBID-HIE 2023 (Neonatal Hypoxic-Ischemic Encephalopathy)
* **Cohort Size:** 85 training subjects (`BONBID2023_Train`) and 4 official held-out validation subjects (`BONBID2023_Val`: `MGHNICU_001`, `MGHNICU_002`, `MGHNICU_302`, `MGHNICU_436`).
* **Input Modalities:** 2 standardized diffusion sequences:
  - `1ADC_ss`: Skull-stripped Apparent Diffusion Coefficient ($\times 10^{-4}\text{ mm}^2/\text{s}$).
  - `2Z_ADC`: Age-matched voxel-wise standardized Z-score diffusion map.
* **Annotations:** Expert binary ground-truth lesion labels (`3LABEL`).
* **Format:** MetaImage (`.mha`) and NIfTI (`.nii.gz`).

---

## 5. Methodology & Processing Pipeline

```
Adult Source (ISLES 2022)          Neonatal Target (BONBID-HIE)
 [DWI, ADC, FLAIR]                     [ADC, Z-ADC]
         |                                   |
         v                                   v
Isotropic 2.0mm Resampling          Isotropic 2.0mm Resampling
         |                                   |
Z-Score Normalization               Z-Score Normalization
         |                                   |
         |                         +---------+---------+
         |                         |                   |
         |                   [Inversion D1]       [Native E2]
         |                   [-Z, ADC, 0]         [ADC, Z-ADC]
         |                         |                   |
         v                         v                   v
+------------------+     +------------------+  +------------------+
| Adult Attention  |     | Domain Adaptation|  | Dedicated 2-Ch   |
| Baseline U-Net   |     | DANN (GRL + Disc)|  | Attention U-Net  |
+------------------+     +------------------+  +------------------+
         |                         |                   |
         +-------------------------+-------------------+
                                   |
                                   v
             Zero-Leakage Held-Out Validation Evaluation
             (Voxel, 3D 26-Connectivity Lesion, Volumetric)
```

1. **Neuroimaging Preprocessing:**
   - Automated spatial re-orientation and affine validation.
   - Voxel-spacing standardization to $2.0 \times 2.0 \times 2.0\text{ mm}$ isotropic grid via continuous B-spline/linear interpolation for intensity volumes and nearest-neighbor for masks.
   - Intracranial brain mask derivation from ADC using background thresholding ($\epsilon > 10.0$ or normalized $> -1.5$).
   - Intensity normalization using robust volume-level z-scoring.
2. **Slice Extraction & Class Balancing:**
   - 2D axial slices sampled with a 70% lesion bias to counter severe foreground-background voxel imbalance ($< 0.5\%$ lesion tissue).

---

## 6. Network Architectures

### 6.1 Attention U-Net
The core segmentation engine is a 2D Attention U-Net incorporating soft additive attention gates on skip connections:
$$\mathbf{q}_{att} = \sigma_2\left(\psi^T \text{ReLU}\left(\mathbf{W}_x^T \mathbf{x}_l + \mathbf{W}_g^T \mathbf{g} + \mathbf{b}_g\right) + b_\psi\right)$$
$$\hat{\mathbf{x}}_l = \mathbf{x}_l \odot \mathbf{q}_{att}$$
* **Encoder:** 4 downsampling stages with residual double-convolution blocks (32, 64, 128, 256, 512 channels) and $2 \times 2$ MaxPool.
* **Decoder:** Transposed convolution upsampling with spatial attention gating prior to feature concatenation.
* **Loss:** `CombinedLoss` balancing Soft Dice Loss and Binary Cross-Entropy:
  $$\mathcal{L}_{seg} = 0.5 \cdot \mathcal{L}_{Dice} + 0.5 \cdot \mathcal{L}_{BCE}$$

### 6.2 Domain-Adversarial Neural Network (DANN)
To extract domain-invariant features across adult and neonatal brains, a domain discriminator $\mathcal{D}$ operates at the encoder bottleneck:
* **Gradient Reversal Layer (GRL):** Forward pass acts as identity; backward pass scales and inverts gradients by $-\lambda_p$:
  $$\mathcal{R}(x) = x, \quad \frac{d\mathcal{R}}{dx} = -\lambda_p \mathbf{I}$$
  where $\lambda_p = \frac{2}{1 + \exp(-10 \cdot p)} - 1$ dynamically scales with training progress $p \in [0, 1]$.
* **Min-Max Objective:**
  $$\mathcal{L}(\theta_f, \theta_y, \theta_d) = \mathcal{L}_{seg}(\theta_f, \theta_y) - \lambda_p \mathcal{L}_{domain}(\theta_f, \theta_d)$$

---

## 7. The Modality Inversion Breakthrough

In adult stroke, restricted diffusion produces **hyperintense (bright) signal on DWI** and **hypointense (dark) signal on ADC**. Early zero-shot transfer naively passed `[ADC, Z-ADC, Z-ADC]` into the 3-channel adult network, yielding a disastrous **Mean Dice of 0.1451**. The network expected hyperintense infarctions in Channel 0, but received dark ADC values.

**The Solution (Representation D1):**
By taking the negative of the standardized diffusion score, $-\text{Z-ADC}$, restricted diffusion is mathematically inverted into a strongly positive, hyperintense feature map that mimics DWI:
$$\text{Input D1} = \begin{bmatrix} -\text{Z-ADC} \\ \text{ADC} \\ 0 \end{bmatrix}$$
* **Result:** Zero-shot Adult Baseline performance surged from **$0.1451 \to \mathbf{0.4088}$ Mean Dice** without updating a single model weight.

---

## 8. Experiment Roadmap & Status

| Experiment ID | Description | Input Representation | Status | Key Outcome |
| :---: | :--- | :---: | :---: | :--- |
| **E1** | Reference Benchmark (Full-Volume Pad $128^2$) | `[-Z-ADC, ADC, 0]` | **Completed** | Baseline Dice: $0.4088$, DANN Dice: $0.3062$ |
| **E4-A** | Bounding-Box Cropping (No Resizing) | `[-Z-ADC, ADC, 0]` | **Completed** | Identical to E1 (zero-padding empty background is a null op) |
| **E4-B** | Bounding-Box Cropping with Resizing ($128^2$) | `[-Z-ADC, ADC, 0]` | **Completed** | Magnifies brain $\sim 1.7\times$; DANN Dice on `MGHNICU_436` jumped $+56.2\%$ |
| **E2** | Native 2-Channel Attention U-Net | `[ADC, Z-ADC]` | **Completed** | **New Benchmark:** Mean Dice $\mathbf{0.4448}$, Recall $\mathbf{0.5222}$, HD95 $\mathbf{6.9\text{ mm}}$ |
| **E3** | Learnable Modality Adapter ($2 \to 3$ channels) | Learnable Conv Block | *Planned* | Dynamic pseudo-FLAIR synthesis for frozen adult backbones |
| **E5** | Isotropic vs Native Anisotropic Geometry | Multi-resolution | *Planned* | Investigates high in-plane vs thick slice sampling |

---

## 9. Quantitative Results & Benchmark Comparisons

### 9.1 Cohort-Wide Benchmark ($N=4$ Held-Out Cohort)

All models evaluated strictly on the 4 held-out validation subjects (`MGHNICU_001`, `MGHNICU_002`, `MGHNICU_302`, `MGHNICU_436`).

| Model Architecture | Experiment | Mean Dice $\pm$ Std | Mean IoU | Mean Prec | Mean Rec | Mean Spec | Mean HD95 (mm) | Mean Lesion F1 | Mean Rel Vol Err |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** | `E1_Reference` | $0.4088 \pm 0.2549$ | $0.2869$ | **$0.6533$** | $0.3018$ | **$0.9999$** | $13.0$ | **$0.3950$** | **$0.66$** |
| **Baseline U-Net** | `E4B_Crop_Resize` | $0.4177 \pm 0.2545$ | $0.2938$ | $0.6305$ | $0.3156$ | $0.9998$ | $13.4$ | $0.3598$ | **$0.62$** |
| **DANN U-Net** | `E1_Reference` | $0.3062 \pm 0.2206$ | $0.2006$ | $0.6918$ | $0.2040$ | $\mathbf{1.0000}$ | $19.0$ | $0.3183$ | $0.78$ |
| **DANN U-Net** | `E4B_Crop_Resize` | $0.3252 \pm 0.2060$ | $0.2113$ | $0.6801$ | $0.2163$ | $0.9999$ | $18.5$ | $0.3363$ | $0.76$ |
| **E2 Native U-Net** | **`E2_Native_Reference`** | **$0.4448 \pm 0.3012$** | **$0.3343$** | $0.4368$ | **$0.4987$** | $0.9974$ | **$6.9$** | $0.1566$ | $0.68$ |
| **E2 Native U-Net** | **`E2_Native_Crop_Resize`** | **$0.4403 \pm 0.3106$** | **$0.3348$** | $0.4288$ | **$0.5222$** | $0.9971$ | **$7.3$** | $0.1920$ | $0.88$ |

### 9.2 Granular Per-Subject Breakdown

| Model / Configuration | Subject | Clinical Phenotype | Ground Truth Vol | Dice | Precision | Recall | HD95 (mm) | Rel Vol Err |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline E1** | `MGHNICU_001` | Subtle punctate foci (12) | $2,736\text{ mm}^3$ | $0.0000$ | $0.0000$ | $0.0000$ | N/A | $1.00$ |
| | `MGHNICU_002` | Massive hemispheric infarct | $195,840\text{ mm}^3$ | $0.5712$ | **$0.9992$** | $0.3999$ | $8.25$ | $0.60$ |
| | `MGHNICU_302` | Focal cortical lesion | $17,344\text{ mm}^3$ | **$0.6661$** | $0.8872$ | $0.5332$ | $4.69$ | $0.40$ |
| | `MGHNICU_436` | Disseminated multifocal | $13,200\text{ mm}^3$ | $0.3979$ | $0.7267$ | $0.2739$ | $26.11$ | $0.62$ |
| **E2 Native (Reference)**| `MGHNICU_001` | Subtle punctate foci (12) | $2,736\text{ mm}^3$ | $0.0000$ | $0.0000$ | $0.0000$ | N/A | $1.00$ |
| | `MGHNICU_002` | Massive hemispheric infarct | $195,840\text{ mm}^3$ | **$0.8145$** | $0.9102$ | **$0.7370$** | **$2.83$** | **$0.19$** |
| | `MGHNICU_302` | Focal cortical lesion | $17,344\text{ mm}^3$ | $0.5952$ | $0.5780$ | **$0.6135$** | $6.32$ | **$0.06$** |
| | `MGHNICU_436` | Disseminated multifocal | $13,200\text{ mm}^3$ | $0.3694$ | $0.2590$ | **$0.6442$** | **$11.66$** | $1.49$ |
| **E2 Native (Crop-Resize)**| `MGHNICU_001` | Subtle punctate foci (12) | $2,736\text{ mm}^3$ | $0.0000$ | $0.0000$ | $0.0000$ | N/A | $1.00$ |
| | `MGHNICU_002` | Massive hemispheric infarct | $195,840\text{ mm}^3$ | **$0.8354$** | $0.9220$ | **$0.7637$** | **$2.83$** | **$0.17$** |
| | `MGHNICU_302` | Focal cortical lesion | $17,344\text{ mm}^3$ | $0.5952$ | $0.5780$ | **$0.6135$** | $6.32$ | **$0.06$** |
| | `MGHNICU_436` | Disseminated multifocal | $13,200\text{ mm}^3$ | $0.3306$ | $0.2153$ | **$0.7115$** | $12.81$ | $2.30$ |

### 9.3 MGHNICU_001 Small Punctate Lesion Autopsy

Subject `MGHNICU_001` features 12 punctate lesion foci ($342\text{ voxels} = 2,736\text{ mm}^3$) with mild diffusion restriction ($Z_{\min} = -3.75$).

* **In Adult 3-Channel Models (E1 & E4-B):** Peak predicted probability inside the lesion was $1.29 \times 10^{-6}$ (logits down to $-17.1$). The lesions were completely silent due to the empty 3rd channel.
* **In Native 2-Channel Model (E2 Crop-Resize):** Peak predicted probability surged to **$0.1698$ (16.98%)**—a relative activation boost of **over 5 orders of magnitude ($>100,000\times$)**.
* **Clinical Insight:** While fixed binarization at $\tau = 0.5$ yields $0.0000$ Dice, the feature representations are no longer dead; an adaptive decision boundary ($\tau \approx 0.15$) localizes these mild diffusion restrictions.

---

## 10. Technology Stack

* **Deep Learning Framework:** PyTorch 2.1+ (autograd, AMP support)
* **Medical Image Computing:** NiBabel (NIfTI I/O), SimpleITK (MetaImage `.mha` handling), SciPy (ndimage, distance transforms, connected components)
* **Web Frontend:** Streamlit, Plotly, HTML5/CSS3
* **Scientific Computing & Data:** NumPy, Pandas, Scikit-learn, OpenPyXL
* **Visualization:** Matplotlib, Seaborn

---

## 11. Repository Architecture & Directory Structure

```
ischemic_stroke_lesions/
├── configs/
│   └── config.yaml                     # Central hyperparameters, paths, loss weights
├── frontend/
│   ├── app.py                          # Streamlit application entry point
│   ├── components/                     # Modular UI cards, charts, visualizers
│   ├── pages/                          # Multi-page views (Dashboard, Prediction, Comparison)
│   ├── styles/                         # Custom dark-navy medical CSS theme
│   └── utils/                          # Frontend model and image inference loaders
├── src/
│   ├── model.py                        # Configurable Attention U-Net architecture
│   ├── dann_model.py                   # DANN architecture with GRL and domain discriminator
│   ├── data_utils.py                   # Unified NIfTI/MHA loaders, resamplers, augmentors
│   ├── losses.py                       # Combined Dice + BCE / Focal loss implementations
│   ├── metrics.py                      # Voxel and lesion evaluation metrics
│   ├── trainer.py                      # Baseline supervised trainer
│   ├── train_baseline.py               # Phase 1 adult baseline training script
│   ├── train_dann.py                   # Phase 4 DANN domain adaptation training script
│   ├── train_evaluate_e2.py            # Phase 2 E2 native 2-channel training & evaluation
│   ├── run_phase2_e1_e4.py             # Phase 2 E1 vs E4-A vs E4-B evaluation engine
│   ├── evaluate_neonatal_zeroshot.py   # Zero-shot evaluation script
│   ├── scientific_evaluation.py        # Independent scientific verification & HD95
│   └── inspect_data.py                 # Dataset verification & integrity checker
├── tests/
│   ├── test_dann_architecture.py       # Unit tests for DANN tensor shapes & GRL
│   ├── test_frontend.py                # Frontend component smoke tests
│   ├── test_load_neonatal.py           # MetaImage loading & shape verification
│   └── test_load_patient.py            # NIfTI patient loading tests
├── outputs/
│   └── phase2/
│       ├── E1_E4_REPORT.md             # Comprehensive scientific report for E1 vs E4
│       ├── logs/                       # Comparative metrics CSVs
│       ├── visualizations/             # High-resolution diagnostic multi-slice overlays
│       └── E2/
│           ├── E2_COMPARISON_REPORT.md # Comprehensive scientific report for E2
│           ├── logs/                   # E2 metrics CSV, training history, JSON logs
│           └── visualizations/         # E2 diagnostic slice overlays & training curves
├── .gitignore                          # Strict isolation of datasets and checkpoints
├── README.md                           # Master scientific repository documentation
└── requirements.txt                    # Python runtime dependencies
```

---

## 12. Installation & Setup

### Prerequisites
* Python 3.10+ (64-bit recommended)
* Git

### Step 1: Clone Repository
```bash
git clone https://github.com/adhithyan05/ischemic_stroke_lesions.git
cd ischemic_stroke_lesions
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 13. Interactive Web Frontend (Streamlit)

A dedicated, multi-page web application provides interactive slice-by-slice medical visualization, multi-modal overlays, and comparative metric analytics.

### Launch Command
```bash
streamlit run frontend/app.py --server.port 8501
```
Open your browser and navigate to `http://localhost:8501`.

### Modules Included:
1. **Executive Dashboard:** High-level project KPIs, domain adaptation gaps, and summary radar charts.
2. **Interactive Prediction:** Live slice viewer with interactive opacity sliders for ADC, Z-ADC, ground truth, and model prediction overlays.
3. **Model Comparison:** Side-by-side diagnostic viewer comparing Baseline Attention U-Net vs. DANN vs. Native E2.
4. **Dataset Explorer:** Cohort distribution inspection across ISLES 2022 and BONBID-HIE 2023.
5. **System Documentation:** Full architecture and loss formulations.

> **Note:** Model checkpoints (`.pth`) and raw datasets (`.nii`, `.mha`) are not stored in Git due to privacy and file size constraints. Place checkpoints in `outputs/checkpoints/` before running live inference.

---

## 14. Evaluation & Reproduction Instructions

### Run Phase 2 Step 1 Benchmark (E1 vs E4-A vs E4-B)
```bash
python src/run_phase2_e1_e4.py
```
Outputs: `outputs/phase2/logs/e1_e4_comparison.csv` and multi-panel figures in `outputs/phase2/visualizations/`.

### Run Phase 2 Experiment E2 (Native 2-Channel Attention U-Net)
```bash
python src/train_evaluate_e2.py
```
Trains the native 2-channel U-Net on `BONBID2023_Train` (held-out subjects strictly excluded) and executes the complete 4-subject evaluation.

### Run Verification Test Suite
```bash
pytest tests/
```

---

## 15. Limitations

1. **Subtle Lesion Sensitivity at Fixed $\tau = 0.5$:** Very small punctate injuries ($< 30\text{ voxels/slice}$) with mild diffusion restriction ($Z > -4.0$) struggle to cross the global $0.5$ threshold when trained alongside massive infarctions.
2. **2D Axial Formulation:** 2D slice processing discards inter-slice 3D spatial continuity, which is critical for resolving isolated millimeter-scale punctate foci.
3. **Hardware Execution:** Training benchmarks were executed on CPU; GPU acceleration with larger batch sizes and 3D spatial convolutions could further refine feature extraction.

---

## 16. Future Work

* **Experiment E3 (Learnable Modality Adapter):** Implementing a lightweight convolutional block to synthesize pseudo-FLAIR representations from ADC and Z-ADC.
* **Native 2-Channel DANN:** Aligning 2-channel adult representations (DWI/ADC) directly with native neonatal $[ADC, Z\text{-}ADC]$.
* **Adaptive Lesion-Size Thresholding:** Implementing scale-aware or focal Tversky losses tailored for punctate neonatal encephalopathy.

---

## 17. References & Acknowledgements

1. **ISLES 2022:** Hernandez Petzsche, M. R., et al. "ISLES 2022: A multi-center magnetic resonance imaging dataset for stroke lesion segmentation." *Scientific Data* 9.1 (2022): 762.
2. **BONBID-HIE 2023:** BONBID-HIE Challenge: Brain Lesion Segmentation in Neonatal Hypoxic-Ischemic Encephalopathy. *MICCAI 2023*.
3. **Attention U-Net:** Oktay, O., et al. "Attention U-Net: Learning Where to Look for the Pancreas." *MIDL 2018*.
4. **Domain-Adversarial Training (DANN):** Ganin, Y., et al. "Domain-adversarial training of neural networks." *JMLR* 17.1 (2016): 2096-2030.

---
*Developed as part of Advanced Deep Learning Research for Medical Image Segmentation.*
