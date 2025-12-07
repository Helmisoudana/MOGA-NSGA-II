"""
main_app_simplified.py

Streamlit application for NSGA-II EV multi-objective optimization.
Simplified version with only 3 design variables.

Run:
    streamlit run main_app_simplified.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, List

# Local modules
from ev_models_simplified import calculate_objectives
from optimization_core_simplified import run_nsga2, DEFAULT_BOUNDS, DEFAULT_VAR_NAMES

import plotly.express as px

# ----------------------------
# Page config and helpers
# ----------------------------
st.set_page_config(page_title="EV Optimization Simplifié (NSGA-II)", layout="wide", initial_sidebar_state="expanded")

VAR_NAMES = DEFAULT_VAR_NAMES
BOUNDS = {name: b for name, b in zip(VAR_NAMES, DEFAULT_BOUNDS)}


def default_design_dict():
    return {
        "batterie_kWh": 60.0,
        "puissance_moteur_kW": 150.0,
        "masse_chassis_kg": 700.0,
    }


def design_to_series(design: Dict[str, float]) -> pd.Series:
    return pd.Series({k: float(v) for k, v in design.items()})


# ----------------------------
# Sidebar: NSGA-II settings
# ----------------------------
st.sidebar.title("NSGA-II Settings")
pop_size = st.sidebar.number_input("Population size", min_value=10, max_value=500, value=80, step=10)
generations = st.sidebar.number_input("Generations", min_value=1, max_value=2000, value=100, step=10)
cxpb = st.sidebar.slider("Crossover rate (cxpb)", 0.0, 1.0, 0.9, 0.01)
mutpb = st.sidebar.slider("Mutation rate (mutpb)", 0.0, 1.0, 0.2, 0.01)
history_interval = st.sidebar.number_input("History snapshot interval", min_value=1, max_value=generations, value=max(1, generations // 10))

if "reset" not in st.session_state:
    st.session_state.reset = False

if st.sidebar.button("Réinitialiser l'état"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Paramètres fixes")
st.sidebar.markdown("""
- **Coefficient de traînée (Cd):** 0.25
- **Surface frontale:** 2.2 m²
- **Coefficient de roulement:** 0.010
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Aide")
st.sidebar.info("Version simplifiée avec 3 variables: Batterie, Puissance moteur, Masse châssis.")

# ----------------------------
# Main layout
# ----------------------------
st.title("Optimisation Multi-Objectifs pour Véhicule Électrique (Version Simplifiée)")
st.markdown("**Objectifs:** Minimiser Poids, Coût, Consommation ; Maximiser Vitesse et Autonomie. "
            "**Variables:** Batterie (kWh), Puissance moteur (kW), Masse châssis (kg).")

col1, col2 = st.columns([2, 1])

# Left column: Inputs and Controls
with col1:
    with st.expander("Inputs & Initial Population", expanded=True):
        tabs = st.tabs(["Manuel (tester un design)", "Charger CSV (initial pop)"])
        
        # Manual input tab
        with tabs[0]:
            st.markdown("**Saisir un design EV**")
            manual = {}
            c1, c2 = st.columns(2)
            with c1:
                manual["batterie_kWh"] = st.slider("Batterie (kWh)", 
                                                  min_value=float(BOUNDS["batterie_kWh"][0]), 
                                                  max_value=float(BOUNDS["batterie_kWh"][1]), 
                                                  value=default_design_dict()["batterie_kWh"],
                                                  step=1.0)
                manual["puissance_moteur_kW"] = st.slider("Puissance moteur (kW)", 
                                                         min_value=float(BOUNDS["puissance_moteur_kW"][0]), 
                                                         max_value=float(BOUNDS["puissance_moteur_kW"][1]), 
                                                         value=default_design_dict()["puissance_moteur_kW"],
                                                         step=5.0)
            with c2:
                manual["masse_chassis_kg"] = st.slider("Masse châssis (kg)", 
                                                      min_value=float(BOUNDS["masse_chassis_kg"][0]), 
                                                      max_value=float(BOUNDS["masse_chassis_kg"][1]), 
                                                      value=default_design_dict()["masse_chassis_kg"],
                                                      step=10.0)

            if st.button("Évaluer le design manuel"):
                objs = calculate_objectives(manual)
                st.success("Évaluation terminée.")
                
                # Display results in a nice format
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Poids total", f"{objs['poids_total']:.1f} kg")
                    st.metric("Coût total", f"${objs['cout_usd']:,.0f}")
                    st.metric("Consommation", f"{objs['consommation_kwh_100km']:.1f} kWh/100km")
                with col_b:
                    st.metric("Vitesse max", f"{objs['vitesse_max_kmh']:.1f} km/h")
                    st.metric("Autonomie", f"{objs['autonomie_km']:.1f} km")
        
        # CSV upload tab
        with tabs[1]:
            st.markdown("**Charger un CSV contenant des designs initiaux**")
            st.markdown("CSV doit contenir les colonnes: " + ", ".join(VAR_NAMES))
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
    run_button = st.button("▶️ Lancer l'Optimisation", type="primary")
    stop_button = st.button("⏸️ Arrêter / Pause")
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

# Additional info area
st.markdown("---")
info_col1, info_col2 = st.columns(2)

# ----------------------------
# Optimization orchestration
# ----------------------------
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
    
    # show top 10 in table
    if pareto_readable:
        dfp = pd.DataFrame(pareto_readable)
        dfp_display = dfp.sort_values(by="autonomie_km", ascending=False).head(15)
        table_placeholder.dataframe(dfp_display)

        # Plot (Cost vs Range) interactive
        try:
            fig = px.scatter(dfp, x="cout_usd", y="autonomie_km", 
                           color="vitesse_max_kmh",
                           size="batterie_kWh",
                           hover_data=VAR_NAMES + ["consommation_kwh_100km", "poids_total"],
                           title=f"Front de Pareto - Génération {gen}")
            pareto_fig_placeholder.plotly_chart(fig, use_container_width=True)
        except Exception:
            pareto_fig_placeholder.write("Mise à jour du graphique...")

    # status text
    progress = gen / generations
    status_placeholder.markdown(f"**Génération:** {gen} / {generations}")
    progress_placeholder.progress(progress)
    gen_placeholder.markdown(f"⏳ Progression: {gen}/{generations}")
    
    # update best placeholder
    if pareto_readable:
        # Find best by a weighted score (autonomy + speed - cost)
        def score(row):
            return (row["autonomie_km"] / 500.0 + row["vitesse_max_kmh"] / 200.0 - row["cout_usd"] / 100000.0)
        
        df_scored = pd.DataFrame(pareto_readable)
        df_scored["score"] = df_scored.apply(score, axis=1)
        best_row = df_scored.loc[df_scored["score"].idxmax()]
        
        best_placeholder.markdown(f"""
        **Meilleur design actuel:**
        - Autonomie: **{best_row['autonomie_km']:.1f} km**
        - Vitesse max: **{best_row['vitesse_max_kmh']:.1f} km/h**
        - Coût: **${best_row['cout_usd']:,.0f}**
        - Batterie: **{best_row['batterie_kWh']:.1f} kWh**
        """)

    # check stop flag
    if st.session_state.get("stop_requested", False):
        return True
    return False


# Helper to compute readable objectives
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

    # prepare initial population
    init_pop = initial_pop

    # Show initial message
    with info_col1:
        st.info(f"Optimisation lancée avec:\n- Population: {pop_size}\n- Générations: {generations}")

    # Run optimization
    start_time = time.time()
    result = run_nsga2(config, initial_population=init_pop, callback=ui_callback)
    end_time = time.time()

    st.session_state.is_running = False

    # Show final results
    st.success(f"Optimisation terminée en {end_time - start_time:.1f} secondes!")
    pareto_solutions = result.get("pareto_solutions", [])
    
    if pareto_solutions:
        # Build dataframe
        rows = []
        for s in pareto_solutions:
            row = {**s["design"], **s["objectives"]}
            rows.append(row)
        df_final = pd.DataFrame(rows)
        
        st.markdown("### Front de Pareto final (solutions non-dominées)")
        
        # Display in two columns: metrics and data
        final_col1, final_col2 = st.columns(2)
        
        with final_col1:
            st.metric("Nombre de solutions Pareto", len(df_final))
            st.metric("Meilleure autonomie", f"{df_final['autonomie_km'].max():.1f} km")
            st.metric("Plus bas coût", f"${df_final['cout_usd'].min():,.0f}")
        
        with final_col2:
            st.dataframe(df_final.sort_values(by="autonomie_km", ascending=False).reset_index(drop=True).head(10))

        # Plot final Pareto front
        fig_final = px.scatter(df_final, x="cout_usd", y="autonomie_km",
                             color="vitesse_max_kmh",
                             size="batterie_kWh",
                             hover_data=VAR_NAMES + ["consommation_kwh_100km", "poids_total"],
                             title="Front de Pareto Final")
        pareto_fig_placeholder.plotly_chart(fig_final, use_container_width=True)
        
        # Add download button for results
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger résultats CSV",
            data=csv,
            file_name="pareto_front_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("Aucune solution Pareto trouvée.")

# Stop button handling
if stop_button:
    st.session_state.stop_requested = True
    st.session_state.is_running = False
    status_placeholder.warning("Arrêt demandé par l'utilisateur.")

# If not running, show a static sample using manual input for demonstration
if not st.session_state.is_running and not run_button:
    with info_col1:
        st.markdown("### 📊 Relations entre variables")
        
        # Generate sample data for visualization
        sample_data = []
        for batt in np.linspace(40, 100, 5):
            for power in np.linspace(80, 250, 5):
                for mass in np.linspace(600, 1000, 5):
                    design = {
                        "batterie_kWh": float(batt),
                        "puissance_moteur_kW": float(power),
                        "masse_chassis_kg": float(mass)
                    }
                    objs = calculate_objectives(design)
                    sample_data.append({
                        **design,
                        **objs,
                        "ratio_power_weight": power / (mass/1000)
                    })
        
        df_sample = pd.DataFrame(sample_data)
        
        # Show relationships
        fig1 = px.scatter(df_sample, x="batterie_kWh", y="autonomie_km", 
                         color="cout_usd", size="puissance_moteur_kW",
                         title="Batterie vs Autonomie")
        st.plotly_chart(fig1, use_container_width=True)
    
    with info_col2:
        st.markdown("### ⚙️ Paramètres recommandés")
        st.markdown("""
        Pour de bonnes performances:
        
        **Batterie:** 60-80 kWh  
        → Bon équilibre autonomie/coût
        
        **Puissance moteur:** 150-200 kW  
        → Vitesse adéquate (180-220 km/h)
        
        **Masse châssis:** 700-900 kg  
        → Équilibre rigidité/poids
        """)
        
        # Show correlation matrix
        corr_data = df_sample[['batterie_kWh', 'puissance_moteur_kW', 'masse_chassis_kg', 
                               'autonomie_km', 'cout_usd', 'vitesse_max_kmh']].corr()
        
        fig2 = px.imshow(corr_data, text_auto=True, aspect="auto",
                        title="Corrélations entre variables")
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Version simplifiée - 3 variables de conception. Modèle physique académique pour démonstration.")
