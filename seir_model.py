from dataclasses import dataclass
from typing import Callable
import numpy as np


# ---------------------------------------------------------------------------
# Parametre paketi
# ---------------------------------------------------------------------------
@dataclass
class SEIRParams:
    # Model parametreleri ve başlangıç koşulları
    beta: float = 0.5          # bulaşma oranı [1/gün]
    sigma: float = 1.0 / 5.0   # kuluçka çıkış oranı (5 gün kuluçka)
    gamma: float = 1.0 / 7.0   # iyileşme oranı (7 gün bulaştırıcılık)
    nu: float = 0.0            # aşılama oranı [1/gün] (default: kampanya yok)
    N: float = 1_000_000.0     # toplam popülasyon

    # Başlangıç koşulu (S0, E0, I0, R0). Toplamı N olmalı.
    S0: float = 999_990.0
    E0: float = 0.0
    I0: float = 10.0           # 10 başlangıç vakası
    R0_init: float = 0.0       # "R0" parametre adıyla çakışmasın diye _init

    @property
    def R0_basic(self) -> float:
        return self.beta / self.gamma

    @property
    def herd_immunity_threshold(self) -> float:
        # p_c = 1 - 1/R₀, yalnızca R₀ > 1 için anlamlı
        if self.R0_basic <= 1.0:
            return 0.0
        return 1.0 - 1.0 / self.R0_basic

    @property
    def y0(self) -> np.ndarray:
        return np.array([self.S0, self.E0, self.I0, self.R0_init], dtype=float)

    def summary(self) -> str:
        return (
            f"SEIR parametreleri:\n"
            f"  β = {self.beta:.4f}   (bulaşma oranı, 1/gün)\n"
            f"  σ = {self.sigma:.4f}   (kuluçka çıkış, 1/{1/self.sigma:.1f} gün)\n"
            f"  γ = {self.gamma:.4f}   (iyileşme, 1/{1/self.gamma:.1f} gün)\n"
            f"  ν = {self.nu:.4f}   (aşılama oranı, 1/gün)\n"
            f"  N = {self.N:,.0f}\n"
            f"  R₀ = β/γ = {self.R0_basic:.3f}\n"
            f"  Aşılama eşiği p_c = {self.herd_immunity_threshold:.3f}\n"
            f"  Başlangıç: S={self.S0:,.0f}, E={self.E0:,.0f}, "
            f"I={self.I0:,.0f}, R={self.R0_init:,.0f}"
        )


def make_seir_rhs(params: SEIRParams) -> Callable[[float, np.ndarray], np.ndarray]:
    # Closure: parametreleri yakala, (t, y) -> dy/dt imzalı fonksiyon döndür
    beta = params.beta
    sigma = params.sigma
    gamma = params.gamma
    nu = params.nu
    N = params.N

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        S, E, I, R = y[0], y[1], y[2], y[3]
        new_infections = beta * S * I / N
        vaccinated = nu * S

        dS = -new_infections - vaccinated
        dE =  new_infections - sigma * E
        dI =  sigma * E - gamma * I
        dR =  gamma * I + vaccinated

        return np.array([dS, dE, dI, dR])

    return rhs


def conservation_error(y: np.ndarray, N: float) -> float:
    # max |S+E+I+R - N| / N — sıfıra yakınsa yöntem korunum sağlıyor
    totals = y.sum(axis=1)
    return float(np.max(np.abs(totals - N)) / N)


def epidemic_summary(t: np.ndarray, y: np.ndarray, N: float) -> dict:
    I = y[:, 2]
    R = y[:, 3]
    S = y[:, 0]

    peak_idx = int(np.argmax(I))

    return {
        "peak_I": float(I[peak_idx]),
        "peak_time": float(t[peak_idx]),
        "attack_rate": float(R[-1] / N),
        "final_S_frac": float(S[-1] / N),
        "conserv_err": conservation_error(y, N),
    }
