Directory Structure:

DARWIN/
├── README.md (update existing)
├── LICENSE (MIT - already exists)
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   ├── API_reference.md
│   ├── examples.md
│   └── mathematical_foundations.md
├── src/
│   └── sfh/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── coherence_model.py
│       │   ├── fertility_model.py
│       │   └── optimization.py
│       ├── dynamics/
│       │   ├── __init__.py
│       │   ├── symbiotic_evolution.py
│       │   ├── population_dynamics.py
│       │   └── ecological_networks.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── meta_analysis.py
│       │   ├── statistical_tests.py
│       │   └── data_processing.py
│       └── visualization/
│           ├── __init__.py
│           ├── phase_plots.py
│           ├── network_viz.py
│           └── evolutionary_trees.py
├── examples/
│   ├── basic_coherence_fertility.py
│   ├── symbiotic_dynamics_demo.py
│   ├── mitochondrial_evolution.py
│   └── bacterial_cooperation.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_dynamics.py
│   ├── test_analysis.py
│   └── test_visualization.py
├── data/
│   ├── empirical/
│   │   ├── endosymbiosis_data.csv
│   │   ├── bacterial_cooperation.csv
│   │   └── metabolic_networks.csv
│   └── simulated/
│       ├── coherence_fertility_timeseries.csv
│       └── symbiotic_evolution_results.csv
├── notebooks/
│   ├── 01_SFH_Introduction.ipynb
│   ├── 02_Coherence_Fertility_Analysis.ipynb
│   ├── 03_Symbiotic_Evolution_Models.ipynb
│   ├── 04_Empirical_Validation.ipynb
│   └── 05_Meta_Analysis_Reproduction.ipynb
└── scripts/
    ├── run_simulations.py
    ├── generate_figures.py
    └── reproduce_paper_results.py

