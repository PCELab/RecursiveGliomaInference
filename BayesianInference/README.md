# BayesianInference

## Overview

This folder contains Bayesian inference scripts for the **mechanically coupled reaction–diffusion (Mech-RD)** tumor model using **hyperelastic mechanics**.

The main script is:

* `Rat-3_hyper.py`

Which performs full inference for a specific rat dataset.

---

## 📁 Contents

* `Rat-3_hyper.py` → Main inference scripts
* `mech_hyper.py` → Hyperelastic mechanics model
* `td_mechHyp.py` → Time-dependent PDE formulation
* `utils.py` → Helper functions

---

## 📁 Required Folder Structure

The code must be executed with the following structure:

```text
parent_directory/
│
├── MRI_data/        (dataset for corresponding rat, i.e; W05_data)
├── hippylib/        (required library)
└── RD+Hyper/
    ├── Rat-3_hyper.py
    ├── mech_hyper.py
    ├── td_mechHyp.py
    └── utils.py
```

---

## 📌 Notes

* No modification is required to run the scripts with default settings
* Ensure correct MRI dataset is available for the rat
* All auxiliary files must remain in the same folder

---

## Output

This script generates:

* Inferred parameter fields
* Tumor evolution over time
* Files for visualization and analysis

