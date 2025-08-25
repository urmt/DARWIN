# DARWIN: Computational Models for SFH Evolution Theory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/urmt/DARWIN)

**DARWIN** is a comprehensive computational framework designed to model and analyze the **Sentience-Field Hypothesis (SFH)** evolutionary theory. This framework provides mathematical models, simulations, and analytical tools to explore the emergence of coherence, fertility, and cooperative behaviors in biological systems through the lens of sentience-field dynamics.

## 🧬 Overview

The Sentience-Field Hypothesis proposes that biological evolution is guided not just by random mutations and natural selection, but by emergent sentience fields that influence evolutionary trajectories toward greater coherence and fertility. DARWIN provides the computational infrastructure to model, simulate, and analyze these complex dynamics.

### Key Concepts

- **Sentience Fields**: Emergent informational structures that guide evolutionary processes
- **Coherence**: Measure of systematic organization and functional integration
- **Fertility**: Capacity for sustainable reproduction and growth
- **Symbiotic Evolution**: Cooperative evolutionary dynamics driven by sentience-field interactions

## 🚀 Features

### Core Computational Models
- **Field Dynamics Simulation**: Mathematical modeling of sentience-field emergence and evolution
- **Coherence-Fertility Optimization**: Algorithms for modeling the optimization of biological coherence and fertility
- **Multi-Scale Analysis**: Tools for analyzing evolutionary dynamics at cellular, organism, and ecosystem levels
- **Symbiotic Relationship Modeling**: Frameworks for simulating cooperative evolutionary strategies

### Simulation Capabilities
- **Population Dynamics**: Large-scale evolutionary simulations with sentience-field influences
- **Mutation-Selection Balance**: Models incorporating field-guided mutation patterns
- **Ecosystem Interactions**: Multi-species simulations with emergent cooperation
- **Temporal Evolution**: Long-term evolutionary trajectory analysis

### Visualization & Analysis
- **Real-time Visualization**: Interactive plots of evolutionary dynamics
- **Statistical Analysis**: Comprehensive statistical tools for model validation
- **Data Export**: Results exportable to multiple formats (CSV, JSON, HDF5)
- **Comparative Analysis**: Tools for comparing different evolutionary scenarios

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Standard Installation
```bash
pip install sfh-darwin
```

### Development Installation
```bash
git clone https://github.com/urmt/DARWIN.git
cd DARWIN
pip install -e .[dev]
```

### With Optional Dependencies
```bash
# For Jupyter notebook examples
pip install sfh-darwin[notebooks]

# For documentation building
pip install sfh-darwin[docs]

# For development tools
pip install sfh-darwin[dev]

# All extras
pip install sfh-darwin[notebooks,docs,dev]
```

## 🔧 Quick Start

### Web Interface (Recommended for Scientists)

**No programming required!** Use the interactive web interface:

1. **Download** the repository or just the `web_interface/darwin_interface.html` file
2. **Open** `darwin_interface.html` in any web browser (Chrome, Firefox, Safari, Edge)
3. **Experiment** with parameters using intuitive controls
4. **Run simulations** and see real-time results
5. **Export data** for further analysis

The web interface provides:
- 🎛️ Parameter adjustment without code editing
- 📊 Real-time visualization of evolution dynamics  
- 📈 Statistical analysis and trend detection
- 💾 Data export to CSV format
- 📚 Built-in theoretical background

### Python API (For Advanced Users)

```python
import sfh_darwin as sfh

# Initialize a basic SFH simulation
simulation = sfh.SFHSimulation(
    population_size=1000,
    mutation_rate=0.001,
    selection_strength=0.1,
    sentience_field_strength=0.05
)

# Run the simulation
results = simulation.run(generations=1000)

# Analyze coherence and fertility evolution
coherence_trajectory = results.get_coherence_trajectory()
fertility_trajectory = results.get_fertility_trajectory()

# Visualize results
sfh.plot_evolutionary_trajectory(results)
```

### Command Line Interface

The framework includes command-line tools for common operations:

```bash
# Run comprehensive simulations
sfh-simulate --config config/default.yaml --output results/

# Run basic demonstration
sfh-demo
```

## 📁 Project Structure

```
DARWIN/
├── src/sfh_darwin/
│   ├── core/                 # Core mathematical models
│   │   ├── fields.py         # Sentience field dynamics
│   │   ├── coherence.py      # Coherence measurement and optimization
│   │   ├── fertility.py      # Fertility modeling
│   │   └── evolution.py      # Evolutionary dynamics
│   ├── simulations/          # Simulation engines
│   │   ├── population.py     # Population-level simulations
│   │   ├── ecosystem.py      # Ecosystem-wide dynamics
│   │   └── temporal.py       # Long-term evolution
│   ├── analysis/             # Analysis and statistics
│   │   ├── metrics.py        # Key performance metrics
│   │   ├── statistics.py     # Statistical analysis tools
│   │   └── validation.py     # Model validation
│   ├── visualization/        # Plotting and visualization
│   │   ├── plots.py          # Standard plots
│   │   ├── interactive.py    # Interactive visualizations
│   │   └── export.py         # Data export utilities
│   └── utils/                # Utility functions
├── examples/                 # Example scripts and notebooks
├── docs/                     # Documentation
├── tests/                    # Unit and integration tests
├── scripts/                  # Command-line scripts
└── config/                   # Configuration files
```

## 📊 Examples

### Example 1: Basic Coherence-Fertility Analysis

```python
from sfh_darwin import CoherenceFertilityModel
import matplotlib.pyplot as plt

# Create a simple C-F model
model = CoherenceFertilityModel(
    initial_coherence=0.3,
    initial_fertility=0.5,
    field_coupling=0.1
)

# Evolve the system
time_points = model.evolve(time_steps=1000)

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(time_points, model.coherence_history)
plt.title('Coherence Evolution')
plt.xlabel('Time')
plt.ylabel('Coherence')

plt.subplot(1, 2, 2)
plt.plot(time_points, model.fertility_history)
plt.title('Fertility Evolution')
plt.xlabel('Time')
plt.ylabel('Fertility')
plt.tight_layout()
plt.show()
```

### Example 2: Multi-Species Ecosystem Simulation

```python
from sfh_darwin import EcosystemSimulation

# Define species with different characteristics
species_config = [
    {'name': 'cooperative', 'cooperation_tendency': 0.8},
    {'name': 'competitive', 'cooperation_tendency': 0.2},
    {'name': 'neutral', 'cooperation_tendency': 0.5}
]

# Create ecosystem
ecosystem = EcosystemSimulation(
    species_config=species_config,
    environment_complexity=0.7,
    sentience_field_enabled=True
)

# Run simulation
results = ecosystem.run(time_steps=5000)

# Analyze emergence of cooperation
cooperation_levels = results.analyze_cooperation_emergence()
ecosystem.plot_species_dynamics(results)
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sfh_darwin

# Run specific test categories
pytest tests/test_core.py
pytest tests/test_simulations.py
```

## 📚 Documentation

### Currently Available:
- **Web Interface**: Interactive tutorial built into `web_interface/darwin_interface.html`
- **Theory Tab**: SFH concepts and mathematical framework (in web interface)
- **Code Examples**: Sample usage in this README

### Planned Documentation (Coming Soon):
- **API Reference**: Detailed API documentation  
- **Theory Guide**: Comprehensive mathematical foundations of SFH
- **Tutorials**: Step-by-step research guides
- **Examples**: Jupyter notebooks with biological case studies

> 📝 **Note**: Full documentation is under development. For now, use the web interface for tutorials and the theory tab for SFH background.

## 🤝 Contributing

We welcome contributions to the DARWIN framework! 

### Development Setup
```bash
git clone https://github.com/urmt/DARWIN.git
cd DARWIN
pip install -e .[dev]
```

### Current Priority Areas:
1. **Core Python modules** - Implement the mathematical models
2. **Documentation** - Create comprehensive guides and tutorials  
3. **Examples** - Develop real-world biological case studies
4. **Testing** - Build test suite for validation

### Code Style
We use:
- **Black** for code formatting
- **flake8** for linting
- **isort** for import sorting

Format code before submitting:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

> 📧 **Get Involved**: Contact the project team to discuss contributions and current development priorities.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Scientific Background

### Theoretical Foundation

The Sentience-Field Hypothesis builds upon established evolutionary theory while proposing novel mechanisms for the emergence of biological complexity. Key theoretical components include:

1. **Field Theory**: Borrowed from physics, applied to biological information systems
2. **Emergent Complexity**: Mathematical models of how simple rules create complex behaviors
3. **Cooperative Evolution**: Extensions of game theory to multi-level selection
4. **Information Theory**: Quantitative measures of biological information and organization

### Publications & References

*[This section would include relevant publications, papers, and theoretical references supporting the SFH framework]*

## 🚧 Roadmap

### Version 1.1 (Planned)
- [ ] GPU acceleration for large-scale simulations
- [ ] Machine learning integration for parameter optimization
- [ ] Extended visualization capabilities
- [ ] Web-based simulation interface

### Version 1.2 (Future)
- [ ] Distributed computing support
- [ ] Real biological data integration
- [ ] Advanced statistical analysis tools
- [ ] Mobile app for basic simulations

## 🆘 Support & Community

- **Issues**: [GitHub Issues](https://github.com/urmt/DARWIN/issues) - Report bugs or request features
- **Web Interface**: Built-in help and theory guide in `darwin_interface.html`
- **Email**: Contact the project team for research collaborations

> 🚀 **Quick Help**: The web interface includes a Theory tab with SFH explanations and built-in tutorials.

## 📞 Contact

**Mark Traver**  
Project Lead & Creator  
[Email](mailto:mark@hotmail.com) | [GitHub](https://github.com/urmt)

## 🙏 Acknowledgments

Special thanks to the scientific community working on evolutionary theory, complex systems, and computational biology. This work builds upon decades of research in these fields.

---

*DARWIN Framework - Advancing our understanding of evolution through computational modeling of sentience-field dynamics.*
