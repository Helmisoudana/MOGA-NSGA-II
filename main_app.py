"""
main_app.py

Streamlit application orchestrating the NSGA-II EV multi-objective optimization.
Modern UI layout with sidebar settings, manual inputs, CSV upload, control buttons,
live Pareto visualization (Plotly), data table, and 3D placeholder.

Run:
    streamlit run main_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, List

# Local modules
from ev_models import calculate_objectives
from optimization_core import run_nsga2, DEFAULT_BOUNDS, DEFAULT_VAR_NAMES
from visualization_3d import render_3d_car_animation

import plotly.express as px

# ----------------------------
# Page config and helpers
# ----------------------------
st.set_page_config(page_title="EV Multi-Objective Optimization (NSGA-II)", layout="wide", initial_sidebar_state="expanded")

VAR_NAMES = DEFAULT_VAR_NAMES
BOUNDS = {name: b for name, b in zip(VAR_NAMES, DEFAULT_BOUNDS)}


def default_design_dict():
    return {
        "batterie_kWh": 60.0,
        "puissance_moteur_kW": 150.0,
        "masse_châssis_kg": 700.0,
        "Cd": 0.24,
        "surface_frontale_m2": 2.2,
        "facteur_roue": 1.0,
    }


def design_to_series(design: Dict[str, float]) -> pd.Series:
    return pd.Series({k: float(v) for k, v in design.items()})


# ----------------------------
# Sidebar: NSGA-II settings
# ----------------------------
st.sidebar.title("NSGA-II Settings")
pop_size = st.sidebar.number_input("Population size", min_value=10, max_value=500, value=80, step=10)
generations = st.sidebar.number_input("Generations", min_value=1, max_value=2000, value=120, step=10)
cxpb = st.sidebar.slider("Crossover rate (cxpb)", 0.0, 1.0, 0.9, 0.01)
mutpb = st.sidebar.slider("Mutation rate (mutpb)", 0.0, 1.0, 0.2, 0.01)
history_interval = st.sidebar.number_input("History snapshot interval (generations)", min_value=1, max_value=generations, value=max(1, generations // 10))

if "reset" not in st.session_state:
    st.session_state.reset = False

if st.sidebar.button("Réinitialiser l'état"):
    # reseat session state keys used by app
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Aide")
st.sidebar.info("Cette application illustre un pipeline NSGA-II pour un problème EV académique. "
                "Les formules sont simplifiées pour démonstration et pédagogie.")

# ----------------------------
# Main layout
# ----------------------------
st.title("Optimisation Multi-Objectifs pour la Conception d'un Véhicule Électrique (NSGA-II)")
st.markdown("**Objectifs:** Minimiser Poids, Coût, Consommation ; Maximiser Vitesse et Autonomie. "
            "L'interface permet de tester des designs manuels, charger un jeu initial via CSV, "
            "et lancer NSGA-II.")

col1, col2 = st.columns([2, 1])

# Left column: Inputs and Controls
with col1:
    with st.expander("Inputs & Initial Population", expanded=True):
        tabs = st.tabs(["Manuel (tester un design)", "Charger CSV (initial pop)"])
        # Manual input tab
        with tabs[0]:
            st.markdown("**Saisir un design EV (variable continue)**")
            manual = {}
            c1, c2 = st.columns(2)
            with c1:
                manual["batterie_kWh"] = st.slider("Batterie (kWh)", min_value=float(BOUNDS["batterie_kWh"][0]), max_value=float(BOUNDS["batterie_kWh"][1]), value=default_design_dict()["batterie_kWh"])
                manual["puissance_moteur_kW"] = st.slider("Puissance moteur (kW)", min_value=float(BOUNDS["puissance_moteur_kW"][0]), max_value=float(BOUNDS["puissance_moteur_kW"][1]), value=default_design_dict()["puissance_moteur_kW"])
                manual["masse_châssis_kg"] = st.slider("Masse châssis (kg)", min_value=float(BOUNDS["masse_châssis_kg"][0]), max_value=float(BOUNDS["masse_châssis_kg"][1]), value=default_design_dict()["masse_châssis_kg"])
            with c2:
                manual["Cd"] = st.slider("Cd (coefficient aérodynamique)", min_value=float(BOUNDS["Cd"][0]), max_value=float(BOUNDS["Cd"][1]), value=default_design_dict()["Cd"], step=0.01)
                manual["surface_frontale_m2"] = st.slider("Surface frontale (m²)", min_value=float(BOUNDS["surface_frontale_m2"][0]), max_value=float(BOUNDS["surface_frontale_m2"][1]), value=default_design_dict()["surface_frontale_m2"])
                manual["facteur_roue"] = st.slider("Facteur roue", min_value=float(BOUNDS["facteur_roue"][0]), max_value=float(BOUNDS["facteur_roue"][1]), value=default_design_dict()["facteur_roue"], step=0.01)

            if st.button("Évaluer le design manuel"):
                objs = calculate_objectives(manual)
                st.success("Évaluation terminée.")
                st.json(objs)

        # CSV upload tab
        with tabs[1]:
            st.markdown("**Charger un CSV contenant des designs initiaux**")
            st.markdown("CSV must contain columns: " + ", ".join(VAR_NAMES))
            uploaded = st.file_uploader("Téléverser CSV", type=["csv"])
            initial_population = None
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    missing = [c for c in VAR_NAMES if c not in df.columns]
                    if missing:
                        st.error(f"Colonnes manquantes dans CSV: {missing}")
                    else:
                        # Take up to pop_size rows
                        df = df[VAR_NAMES].astype(float)
                        st.success(f"Chargé {len(df)} designs depuis le CSV.")
                        st.dataframe(df.head(20))
                        initial_population = df.to_dict(orient="records")
                except Exception as e:
                    st.error(f"Erreur lecture CSV: {e}")

# Right column: Controls & Status
with col2:
    st.markdown("### Contrôles")
    run_button = st.button("▶️ Lancer l'Optimisation")
    stop_button = st.button("⏸️ Arrêter / Pause (simulé)")
    st.markdown("---")
    st.markdown("### Statut")
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    gen_placeholder = st.empty()
    best_placeholder = st.empty()

# Visualization area full width
st.markdown("---")
viz_col1, viz_col2 = st.columns([1.2, 1.0])

pareto_fig_placeholder = viz_col1.empty()
table_placeholder = viz_col2.empty()

# 3D animation area
three_col1, three_col2 = st.columns([1, 2])
three_container = three_col1.empty()


# ----------------------------
# Optimization orchestration
# ----------------------------
# Hold "running" state in session_state
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

# read initial population from uploaded if present
if 'uploaded' in locals() and uploaded is not None and 'initial_population' in locals() and initial_population:
    initial_pop = initial_population
else:
    initial_pop = None

# Callback to be used by optimizer
def ui_callback(gen: int, current_pareto: List[Dict]) -> bool:
    """
    Called by the optimizer every generation. Updates UI components.
    Return True to stop early.
    """
    # update UI values
    st.session_state.latest_generation = gen
    # current_pareto contains dicts with 'design' and 'objectives' lists
    # convert to readable table for display
    pareto_readable = []
    for item in current_pareto:
        design = item["design"]
        # compute readable objectives
        objs = _evaluate_design_for_ui(design)
        pareto_readable.append({**design, **objs})
    # show top 10
    dfp = pd.DataFrame(pareto_readable)
    if not dfp.empty:
        dfp_display = dfp.sort_values(by="autonomie_km", ascending=False).head(15)
        table_placeholder.dataframe(dfp_display)

        # Plot (Cost vs Range) interactive
        try:
            fig = px.scatter(dfp, x="cout_usd", y="autonomie_km", color="vitesse_max_kmh",
                             size="batterie_kWh",
                             hover_data=VAR_NAMES + ["consommation_kwh_100km", "poids_total"])
            pareto_fig_placeholder.plotly_chart(fig, use_container_width=True)
        except Exception:
            pareto_fig_placeholder.write("Impossible d'afficher la figure pour le moment.")

    # status text
    status_placeholder.markdown(f"**Génération:** {gen}")
    gen_placeholder.markdown(f"⏳ Génération en cours: {gen}")
    # update best placeholder with a best-so-far (by domination heuristic)
    if pareto_readable:
        best = max(pareto_readable, key=lambda r: r["autonomie_km"])
        best_placeholder.markdown(f"**Meilleur (par autonomie):** Autonomie {best['autonomie_km']:.1f} km — Coût ${best['cout_usd']:.0f}")
    # simulate a visual 3D update
    # compute a simple progress fraction
    progress = (gen % max(1, generations)) / float(max(1, generations))
    render_3d_car_animation(three_container, pareto_readable[0] if pareto_readable else {}, progress)

    # check stop flag set via UI
    if st.session_state.get("stop_requested", False):
        return True
    return False


# Helper to compute readable objectives (used for UI building of table)
def _evaluate_design_for_ui(design: Dict[str, float]) -> Dict[str, float]:
    objs = calculate_objectives(design)
    return {
        "poids_total": objs["poids_total"],
        "cout_usd": objs["cout_usd"],
        "vitesse_max_kmh": objs["vitesse_max_kmh"],
        "autonomie_km": objs["autonomie_km"],
        "consommation_kwh_100km": objs["consommation_kwh_100km"],
    }


# Run when button pressed
if run_button:
    # reset stop flag
    st.session_state.stop_requested = False
    st.session_state.is_running = True
    status_placeholder.info("Démarrage de l'optimisation...")
    st.session_state.latest_generation = 0

    # Build config
    config = {
        "population_size": pop_size,
        "generations": generations,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "var_names": VAR_NAMES,
        "bounds": [(float(BOUNDS[n][0]), float(BOUNDS[n][1])) for n in VAR_NAMES],
        "history_interval": history_interval,
    }

    # prepare initial population: if CSV provided, use it; else None
    init_pop = initial_pop

    # Run optimization (blocking loop that updates UI via callback)
    start_time = time.time()
    result = run_nsga2(config, initial_population=init_pop, callback=ui_callback)
    end_time = time.time()

    st.session_state.is_running = False

    # Show final results
    st.success(f"Optimisation terminée en {end_time - start_time:.1f} s (ou arrêtée).")
    pareto_solutions = result.get("pareto_solutions", [])
    if pareto_solutions:
        # Build dataframe
        rows = []
        for s in pareto_solutions:
            row = {**s["design"], **s["objectives"]}
            rows.append(row)
        df_final = pd.DataFrame(rows)
        st.markdown("### Front de Pareto final (solutions non-dominées)")
        st.dataframe(df_final.sort_values(by="autonomie_km", ascending=False).reset_index(drop=True))

        # Plot final
        fig_final = px.scatter(df_final, x="cout_usd", y="autonomie_km",
                               color="vitesse_max_kmh",
                               size="batterie_kWh",
                               hover_data=VAR_NAMES + ["consommation_kwh_100km", "poids_total"])
        pareto_fig_placeholder.plotly_chart(fig_final, use_container_width=True)
    else:
        st.warning("Aucune solution pareto trouvée.")

# Stop button handling (set flag)
if stop_button:
    st.session_state.stop_requested = True
    st.session_state.is_running = False
    status_placeholder.warning("Arrêt demandé par l'utilisateur (simulé).")

# If not running, show a static sample using manual input for demonstration
if not st.session_state.is_running:
    st.markdown("### Aperçu (exemple de front de Pareto généré aléatoirement pour démonstration)")
    # quick random sample demonstration
    demo_rows = []
    for _ in range(40):
        sample_design = {n: float(np.random.uniform(low=BOUNDS[n][0], high=BOUNDS[n][1])) for n in VAR_NAMES}
        objs = _evaluate_design_for_ui(sample_design)
        demo_rows.append({**sample_design, **objs})
    df_demo = pd.DataFrame(demo_rows)
    # show demo plot
    fig_demo = px.scatter(df_demo, x="cout_usd", y="autonomie_km", color="vitesse_max_kmh",
                          size="batterie_kWh", hover_data=VAR_NAMES + ["consommation_kwh_100km"])
    pareto_fig_placeholder.plotly_chart(fig_demo, use_container_width=True)
    table_placeholder.dataframe(df_demo.sort_values(by="autonomie_km", ascending=False).head(10))

st.markdown("---")
st.caption("Conçu pour un projet académique — Formules simplifiées. Pour une utilisation en production, "
           "affiner les modèles, valider les constantes, et utiliser un solveur NSGA-II robuste (DEAP / Platypus / pymoo).")
