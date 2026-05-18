# app.py — MAT423E Project 1
# Interactive SEIR simulator — streamlit run app.py

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from methods import euler, rk4, rkf45, adams_bashforth_moulton, nsfd_seir
from seir_model import SEIRParams, make_seir_rhs, epidemic_summary


# Sayfa konfigürasyonu
st.set_page_config(
    page_title="SEIR Epidemic Simulator — MAT423E",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colors — consistent across all plots
COLORS = {
    "S": "#1f77b4",    # blue   — susceptible
    "E": "#ff7f0e",    # orange — exposed
    "I": "#d62728",    # red    — infected
    "R": "#2ca02c",    # green  — recovered
    "Euler": "#8c564b",
    "RK4":   "#2ca02c",
    "ABM4":  "#1f77b4",
    "RKF45": "#d62728",
    "NSFD":  "#9467bd",  # purple — Mickens NSFD
}

METHOD_FUNCS = {
    "Euler": euler,
    "RK4": rk4,
    "ABM4": adams_bashforth_moulton,
    # RKF45 and NSFD have different signatures — handled separately
}


# Cached solver — skips recomputation for identical parameters
@st.cache_data(show_spinner=False)
def solve_seir(method: str, beta: float, sigma: float, gamma: float, nu: float,
               N: float, I0: float, t_end: float, h: float,
               rkf45_tol: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    params = SEIRParams(
        beta=beta, sigma=sigma, gamma=gamma, nu=nu, N=N,
        S0=N - I0, E0=0.0, I0=I0, R0_init=0.0,
    )
    f = make_seir_rhs(params)
    y0 = params.y0

    t0 = time.perf_counter()
    if method == "RKF45":
        t, y = rkf45(f, 0.0, t_end, y0, tol=rkf45_tol,
                     h_init=h, h_min=1e-6, h_max=max(1.0, 5 * h))
        n = len(t) - 1
        n_fev = 6 * n
    elif method == "NSFD":
        # NSFD: SEIR-specific; takes model params directly instead of f(t,y)
        t, y = nsfd_seir(beta, sigma, gamma, nu, N, 0.0, t_end, y0, h)
        n = len(t) - 1
        n_fev = n  # 1 I^n evaluation per step (analytic update, no f calls)
    else:
        t, y = METHOD_FUNCS[method](f, 0.0, t_end, y0, h)
        n = len(t) - 1
        n_fev = {"Euler": 1, "RK4": 4}.get(method, 0) * n
        if method == "ABM4":
            n_fev = 4 * min(3, n) + 2 * max(0, n - 3) + 1
    dt = time.perf_counter() - t0
    return t, y, dt, n_fev


# Sidebar — parameter controls
def render_sidebar() -> dict:
    st.sidebar.title("🦠 SEIR Parameters")
    st.sidebar.caption("MAT423E Project 1 — Epidemic & Vaccination")

    # ---- Quick presets ----------------------------------------------------
    st.sidebar.subheader("⚡ Quick Presets")
    preset = st.sidebar.selectbox(
        "Predefined scenarios:",
        ["(Manual)", "COVID-19-like (unvaccinated)", "Vaccination campaign",
         "Isolation / distancing", "Low transmissibility (flu)", "Very high R₀ (measles)"],
        index=0,
    )

    # Preset → values
    presets = {
        "COVID-19-like (unvaccinated)": dict(beta=0.5, sigma=1/5, gamma=1/7, nu=0.0),
        "Vaccination campaign":         dict(beta=0.5, sigma=1/5, gamma=1/7, nu=0.01),
        "Isolation / distancing":       dict(beta=0.25, sigma=1/5, gamma=1/7, nu=0.0),
        "Low transmissibility (flu)":   dict(beta=0.3, sigma=1/2, gamma=1/4, nu=0.0),
        "Very high R₀ (measles)":       dict(beta=1.5, sigma=1/10, gamma=1/7, nu=0.0),
    }
    defaults = presets.get(preset, dict(beta=0.5, sigma=1/5, gamma=1/7, nu=0.0))

    # ---- Epidemic parameters ---------------------------------------------
    st.sidebar.subheader("Epidemic Parameters")
    beta = st.sidebar.slider("β — Transmission rate (1/day)",
                              0.0, 2.0, defaults["beta"], 0.01,
                              help="Rate at which a susceptible becomes infected per unit time")
    sigma = st.sidebar.slider("σ — Incubation exit rate (1/day)",
                               0.05, 1.0, defaults["sigma"], 0.01,
                               help="1/σ = mean incubation period (days)")
    gamma = st.sidebar.slider("γ — Recovery rate (1/day)",
                               0.05, 1.0, defaults["gamma"], 0.01,
                               help="1/γ = mean infectious period (days)")
    nu = st.sidebar.slider("ν — Vaccination rate (1/day)",
                            0.0, 0.05, defaults["nu"], 0.001,
                            help="Daily fraction of susceptibles vaccinated (0 = no campaign)")

    # Derived quantities — inform the user
    R0 = beta / gamma if gamma > 0 else float("inf")
    incub = 1 / sigma if sigma > 0 else float("inf")
    infect = 1 / gamma if gamma > 0 else float("inf")

    c1, c2 = st.sidebar.columns(2)
    c1.metric("R₀ = β/γ", f"{R0:.2f}")
    c2.metric("Incubation", f"{incub:.1f} days")
    st.sidebar.metric("Infectious period", f"{infect:.1f} days")

    # ---- Population -------------------------------------------------------
    st.sidebar.subheader("Population")
    N = st.sidebar.number_input("N — Total population",
                                 min_value=1000, max_value=100_000_000,
                                 value=1_000_000, step=10_000)
    I0 = st.sidebar.number_input("I₀ — Initial infected",
                                  min_value=1, max_value=10000, value=10, step=1)

    # ---- Numerical solution settings -------------------------------------
    st.sidebar.subheader("Numerical Solution")
    t_end = st.sidebar.slider("Simulation duration (days)", 30, 365, 160, 10)
    h = st.sidebar.select_slider("Step size h (days)",
                                  options=[1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01],
                                  value=0.1)
    rkf45_tol = st.sidebar.select_slider(
        "RKF45 tolerance",
        options=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
    )

    # Info: interpretation based on R₀
    if R0 > 1:
        threshold = 1 - 1 / R0
        st.sidebar.info(
            f"📊 **R₀ = {R0:.2f} > 1**: Epidemic will spread.\n\n"
            f"Herd immunity threshold: **{threshold:.1%}** "
            f"(this fraction of the population must be immune)."
        )
    else:
        st.sidebar.success(
            f"✅ **R₀ = {R0:.2f} ≤ 1**: Epidemic will die out.\n\n"
            f"Under the current parameter set, the disease cannot "
            f"sustain spread in the population."
        )

    return dict(beta=beta, sigma=sigma, gamma=gamma, nu=nu,
                N=float(N), I0=float(I0),
                t_end=float(t_end), h=h, rkf45_tol=rkf45_tol, R0=R0)


# Plot helpers
def plot_seir_curves(t, y, N, title, show_compartments=("S", "E", "I", "R")):
    """Plot SEIR curves in a single panel with optional compartment selection."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = ["S (susceptible)", "E (exposed)", "I (infected)", "R (recovered)"]
    keys = ["S", "E", "I", "R"]
    for i, (key, label) in enumerate(zip(keys, labels)):
        if key in show_compartments:
            ax.plot(t, y[:, i], color=COLORS[key], lw=2.0, label=label)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of individuals")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    return fig


def plot_methods_overlay(results: dict, compartment_idx: int, comp_name: str, t_end: float):
    """Overlay the same compartment for all methods."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, r in results.items():
        ax.plot(r["t"], r["y"][:, compartment_idx],
                color=COLORS[name], lw=1.6, alpha=0.85, label=name)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(f"{comp_name} (individuals)")
    ax.set_title(f"Methods comparison — {comp_name}", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, t_end)
    return fig


# Tab 1 — Epidemic Curve
def tab_epidemic_curve(params: dict):
    st.subheader("📈 Epidemic Curve")
    st.caption("Adjust parameters in the left panel — the chart updates instantly.")

    method = st.radio(
        "Method to display:",
        list(METHOD_FUNCS.keys()) + ["RKF45", "NSFD"],
        index=1, horizontal=True,
        help="RK4 recommended for quick exploration. See Tab 2 for method comparison.",
    )

    show = st.multiselect(
        "Compartments to show:",
        ["S", "E", "I", "R"], default=["S", "E", "I", "R"],
    )

    t, y, runtime, n_fev = solve_seir(
        method, params["beta"], params["sigma"], params["gamma"],
        params["nu"], params["N"], params["I0"],
        params["t_end"], params["h"], params["rkf45_tol"],
    )
    summ = epidemic_summary(t, y, params["N"])

    # Top banner — key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak infected", f"{summ['peak_I']:,.0f}",
              f"{summ['peak_I'] / params['N']:.1%} of population")
    m2.metric("Peak time", f"{summ['peak_time']:.1f} days")
    m3.metric("Total attack rate", f"{summ['attack_rate']:.1%}",
              "all ever infected / N")
    m4.metric("Unimmunized", f"{summ['final_S_frac']:.1%}",
              "of population")

    fig = plot_seir_curves(
        t, y, params["N"],
        title=f"SEIR solution — {method} (h={params['h']}, "
              f"R₀={params['R0']:.2f}, ν={params['nu']}/day)",
        show_compartments=tuple(show),
    )
    st.pyplot(fig, clear_figure=True)

    # Bottom: hands-on activity
    with st.expander("🎯 Hands-on activity: predict the peak"):
        st.markdown(
            f"""
            **Activity flow (5 min in class):**

            1. With the current parameters the peak is **{summ['peak_I']:,.0f}**
               individuals (day {summ['peak_time']:.0f}).
            2. Ask: if I double β, how does the peak change?
               Predict first, then move the slider.
            3. Discuss: when ν increases, does the peak arrive **earlier**
               or **later**? Why?
            4. Bonus: herd immunity threshold is {(1 - 1/params['R0']):.1%}
               (meaningful only for R₀ > 1) — how large must ν be for
               the attack rate to approach this threshold?
            """
        )

    # Performance / numerical health
    with st.expander("ℹ️ Numerical health of the solution"):
        totals = y.sum(axis=1)
        conserv = float(np.max(np.abs(totals - params["N"])) / params["N"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Steps", f"{len(t) - 1:,}")
        c2.metric("Function evaluations", f"{n_fev:,}")
        c3.metric("Conservation relative error", f"{conserv:.1e}",
                  help="|S+E+I+R - N| / N — closer to zero is better")
        st.caption(f"Solve time: **{runtime*1000:.1f} ms**")


# Tab 2 — Method Comparison
def tab_method_comparison(params: dict):
    st.subheader("⚖️ Method Comparison")
    st.caption(
        "The same SEIR system is solved with all methods. Fixed-step methods share "
        "the same h; RKF45 is adaptive and selects its own steps based on the tolerance."
    )

    results = {}
    for method in ["Euler", "RK4", "ABM4", "RKF45", "NSFD"]:
        t, y, runtime, n_fev = solve_seir(
            method, params["beta"], params["sigma"], params["gamma"],
            params["nu"], params["N"], params["I0"],
            params["t_end"], params["h"], params["rkf45_tol"],
        )
        results[method] = {
            "t": t, "y": y, "runtime": runtime, "n_fev": n_fev,
            "n_steps": len(t) - 1,
            "summary": epidemic_summary(t, y, params["N"]),
        }

    comp_choice = st.selectbox(
        "Compartment to compare:", ["I (infected)", "S (susceptible)",
                                     "E (exposed)", "R (recovered)"], index=0,
    )
    comp_idx = {"S (susceptible)": 0, "E (exposed)": 1,
                "I (infected)": 2, "R (recovered)": 3}[comp_choice]
    fig = plot_methods_overlay(results, comp_idx, comp_choice[0], params["t_end"])
    st.pyplot(fig, clear_figure=True)

    # Tablo
    rows = []
    ref_peak = results["RKF45"]["summary"]["peak_I"]
    for name, r in results.items():
        s = r["summary"]
        diff_peak = abs(s["peak_I"] - ref_peak)
        rows.append({
            "Method": name,
            "Steps": r["n_steps"],
            "Function evaluations": r["n_fev"],
            "Time (ms)": f"{r['runtime']*1000:.2f}",
            "Peak I (individuals)": f"{s['peak_I']:,.0f}",
            "Peak day": f"{s['peak_time']:.2f}",
            "Peak diff vs RKF45": f"{diff_peak:,.0f}",
            "Attack rate %": f"{s['attack_rate']*100:.2f}",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **What to look for in the table:**

        - **Function evaluations** ≈ actual computational cost. RK4: 4 per step,
          ABM4 (main loop): 2, Euler: 1, RKF45: 6 per accepted step.
        - **Peak diff vs RKF45** ≈ practical error. RKF45 with tight tolerance
          acts as a reference; Euler should have the largest deviation.
        - **Steps** differs for RKF45 — it is adaptive.
        """
    )


# Main flow
def main():
    st.title("🦠 SEIR Epidemic Spread Simulator")
    st.markdown(
        "**MAT423E — Numerical Solutions of ODEs · Project 1: Epidemic & Vaccination** · "
        "This app solves the SEIR + vaccination model with 5 numerical methods."
    )

    params = render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "📈 Epidemic Curve",
        "⚖️ Method Comparison",
        "📚 Model & Methods",
    ])

    with tab1:
        tab_epidemic_curve(params)
    with tab2:
        tab_method_comparison(params)
    with tab3:
        st.markdown(
            r"""
            ### Model: SEIR + Vaccination

            Closed population (N fixed), homogeneous mixing assumption:

            $$
            \begin{aligned}
            \frac{dS}{dt} &= -\beta\,\frac{S\,I}{N} - \nu\,S \\
            \frac{dE}{dt} &=  \beta\,\frac{S\,I}{N} - \sigma\,E \\
            \frac{dI}{dt} &=  \sigma\,E - \gamma\,I \\
            \frac{dR}{dt} &=  \gamma\,I + \nu\,S
            \end{aligned}
            $$

            **Parameters:**

            - β: effective transmission rate (1/day)
            - σ: 1 / incubation period (1/day)
            - γ: 1 / infectious period (1/day)
            - ν: daily vaccination rate (1/day)

            **Basic reproduction number:** $R_0 = \beta / \gamma$.
            If R₀ > 1 the epidemic spreads. Herd immunity threshold: $1 - 1/R_0$.

            ### Numerical Methods

            | Method | Type | Order | Step/call | Note |
            |--------|------|-------|-----------|------|
            | Euler | Single-step | 1 | 1 fev | Reference / baseline |
            | RK4   | Single-step | 4 | 4 fev | Classic high accuracy |
            | RKF45 | Single-step, adaptive | 4(5) | 6 fev/step | Error-controlled |
            | ABM4  | Multi-step P-C | 4 | 2 fev/main loop | Cheap for long runs |
            | NSFD  | Single-step, semi-implicit | 1 | 0 fev (analytic) | Positivity + conservation guarantee |

            All methods are implemented **from scratch** in `methods.py`.
            scipy.solve_ivp is used only in `validation.py` as a **reference solution**
            (project rule).

            **NSFD (Nonstandard Finite Difference — Mickens):**

            SEIR-specific semi-implicit scheme. An analytically solvable
            linear update at each step:

            $$
            \begin{aligned}
            S^{n+1} &= \frac{S^n}{1 + \varphi(\beta I^n/N + \nu)} \\
            E^{n+1} &= \frac{E^n + \varphi\,\beta\,S^{n+1}\,I^n/N}{1 + \varphi\,\sigma} \\
            I^{n+1} &= \frac{I^n + \varphi\,\sigma\,E^{n+1}}{1 + \varphi\,\gamma} \\
            R^{n+1} &= N - S^{n+1} - E^{n+1} - I^{n+1}
            \end{aligned}
            $$

            where $\varphi = h$ (Mickens denominator function, 1st-order choice).
            The last equation enforces the conservation identity ($S+E+I+R = N$)
            **exactly** at every step. The denominators $(1 + \varphi\,\cdot) > 1$
            **analytically prevent** negative solutions for any $h > 0$.

            ### Hands-on Flow

            1. In **Tab 1**, choose a starting parameter set.
            2. Change β → watch the R₀ banner → predict the peak.
            3. Increase ν → observe how vaccination suppresses the peak.
            4. In **Tab 2**, compare Euler and RK4 → discuss the importance
               of numerical order.

            """
        )


main()