# Acute Stroke Unit Discrete Event Simulation (DES) - Replication of Monks et al. (2016)

This repository contains a simplified Python-based recreation of the Discrete Event Simulation (DES) model described in:

> Monks T, Worthington D, Allen M, Pitt M, Stein K, James M. A modelling tool for capacity planning in acute and community stroke services. *BMC Health Services Research*. 2016;16:530. [https://doi.org/10.1186/s12913-016-1789-4](https://doi.org/10.1186/s12913-016-1789-4)

The original model was developed using SIMUL8. This project recreates its core logic and experiments using Python, SimPy and Streamlit offering open and reproducible code with results.

---
## Authors
Kayleigh Haydock and Guled Abdullahi


## Project Structure

```
.
├── binder/                                 # Reproducible conda environment
│   └── environment.yml
├── images/                                 # Diagrams and visual outputs
│   ├── asu_model.png
│   ├── replicated_model.png
│   ├── iteration_1.png
│   └── dataframe_op.png
├── iterations/                             # Iterative development notebooks and Streamlit app builds
│   ├── 1_iteration.ipynb
│   ├── 2_iteration.ipynb
│   ├── 3_iteration.ipynb
│   ├── 4_iteration.ipynb
│   ├── 5_iteration.ipynb
│   ├── 6_iteration.ipynb
│   ├── my_streamlit_app.py
│   ├── my_streamlit_app_2.py
│   └── my_streamlit_app_3.py
├── papers/                                 # Original research article and appendix
│   ├── Monks_et_al.pdf
│   └── Monks_et_al_appendix.docx
├── result/                                 # Simulation outputs and summary CSVs
│   ├── default_scenario.csv
│   ├── exclusion.csv
│   ├── increase_scenario.csv
│   └── comparing_delay_probability.png
├── 7_iteration.ipynb                       # Final consolidated model notebook
├── final_report.ipynb                      # Final report write-up
├── distribution.py                         # Parameter sampling logic
├── stroke_appendix_data.ipynb              # Research papers data inputs
├── LICENSE                                 # Project license
├── .gitignore                              # Files to ignore in GitHub
└── README.md                               # You're here!
```

---
## How to Run

### Option 1: Local Setup

1. Clone the repo:

```bash
git clone https://github.com/KaysHaydock
```
```bash
cd acute_stroke_des_simulation
```

2. Create a virtual environment and install dependencies:

```bash
conda env create -f binder/environment.yml
```
```bash
conda activate sim_env
```

3. Open the notebook named (7_iteration):

```bash
jupyter-lab
```

4. (Optional once finalised app) Launch the Streamlit app:

```bash
streamlit run iterations/my_streamlit_app_3.py
```

---

## Features

- Core DES logic of ASU patient arrivals, LOS and discharge routing
- Warm-up analysis and multiple replications for statistical validity
- Scenario analysis (base case, 5% increase, patient exclusion)
- Visual outputs including delay probabilities vs bed capacity
- Streamlit app for interactive model exploration (still in draft progress)

---

## Key Visuals

**Model Comparison**

![ASU Original](images/asu_model.png)
![Replicated Model](images/replicated_model.png)

**Output Example**

![Simulation Output](images/comparison_delay_probability.png)

---

## Referenced Material

- `papers/Monks_et_al.pdf` — Original publication
- `papers/Monks_et_al_appendix.docx` — Supplementary data used for parameters

---

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

## Acknowledgements

- Original DES model by Monks et al. (2016)
- Supported by Health Data Science and Operations Research GitHub resources
- Streamlit for enabling rapid UI deployment (which is still in progress)

---
## Contact

For questions, please raise an issue or contact one of the project contributors