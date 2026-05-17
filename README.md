# MAT423E — Modelling Epidemic Spread (Project 1)

An interactive Streamlit application that solves and compares the SEIR epidemic model using Euler, RK4, RKF45, and Adams-Bashforth-Moulton methods.

## Model

Closed-population SEIR model with vaccination:

```
dS/dt = -β·S·I/N - ν·S
dE/dt =  β·S·I/N - σ·E
dI/dt =  σ·E - γ·I
dR/dt =  γ·I + ν·S
```

| Parameter | Description |
|-----------|-------------|
| β (beta)  | Effective transmission rate [1/day] |
| σ (sigma) | 1 / incubation period [1/day] |
| γ (gamma) | Recovery rate [1/day] |
| ν (nu)    | Vaccination rate [1/day] |

## File Structure

| File | Description |
|------|-------------|
| `app.py` | Interactive Streamlit UI |
| `seir_model.py` | SEIR right-hand side function and parameters |
| `methods.py` | Euler, RK4, RKF45, Adams-Bashforth-Moulton implementations |
| `validation.py` | Error analysis against scipy reference solution |
| `driver.py` | Driver for CLI and programmatic use |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Streamlit app:**
```bash
streamlit run app.py
```

**CLI:**
```bash
python driver.py
```

**Validation / error analysis:**
```bash
python validation.py
```
