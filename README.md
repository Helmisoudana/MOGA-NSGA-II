# Electric Vehicle Multi-Objective Optimization using NSGA-II

This project implements a **Multi-Objective Genetic Algorithm (NSGA-II)** to optimize an electric vehicle (EV) design based on multiple objectives such as cost, weight, maximum speed, range, and energy consumption. The project includes a **Streamlit-based interface** for interactive exploration and visualization of the Pareto front.

---

## Features

- Evaluate EV designs with multiple objectives:
  - **Weight** (minimize)
  - **Cost** (minimize)
  - **Maximum Speed** (maximize)
  - **Range** (maximize)
  - **Energy Consumption** (minimize)
- Multi-objective optimization using **NSGA-II** (via [DEAP](https://deap.readthedocs.io/en/master/))
- Interactive visualization with **Streamlit**
- Visualization of Pareto front and top designs
- Supports custom initial population or random generation
- Tracks history of optimization

---

## Project Structure

```bash
.
├── ev_models_simplified.py # EV physical model and objective functions
├── optimization_core_simplified.py # NSGA-II setup and optimization logic
├── main_app_simplified.py # Streamlit application
├── requirements.txt # Python dependencies
└── README.md

```


---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ev-nsga2-optimization.git
cd ev-nsga2-optimization
```

2.**Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

```
3.**Install dependencies**
```bash
pip install -r requirements.txt
```
requirements.txt example:
```bash
numpy
deap
streamlit
matplotlib
pandas

```
## Running the Project

1. **Run the Streamlit App**
```bash
streamlit run main_app_simplified.py
```

This will open a web interface in your browser.

- You can interactively:

    - Test individual designs

    - Run NSGA-II optimization

    - Visualize Pareto front

    - Explore top solutions

2. **Run Optimization Script Only (Optional)**

```bash
python optimization_core_simplified.py
```

This will execute the NSGA-II algorithm in the console and output the final Pareto front.