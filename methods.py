# methods.py — MAT423E Proje 1
# Euler, RK4, RKF45 ve Adams-Bashforth-Moulton yöntemlerinin implementasyonu.
# Tüm yöntemler genel amaçlı — herhangi bir IVP sistemine uygulanabilir.

from typing import Callable, Tuple
import numpy as np


# 1. Forward Euler — 1. mertebe, global hata O(h)
def euler(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    y0: np.ndarray,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_steps = int(np.ceil((tf - t0) / h))
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)

    y = np.zeros((n_steps + 1, y0.size))
    y[0] = y0

    for n in range(n_steps):
        y[n + 1] = y[n] + h * f(t[n], y[n])

    return t, y


# 2. Klasik RK4 — 4. mertebe, global hata O(h^4)
def rk4(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    y0: np.ndarray,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_steps = int(np.ceil((tf - t0) / h))
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)

    y = np.zeros((n_steps + 1, y0.size))
    y[0] = y0

    for n in range(n_steps):
        tn, yn = t[n], y[n]
        k1 = f(tn,           yn)
        k2 = f(tn + h / 2.0, yn + (h / 2.0) * k1)
        k3 = f(tn + h / 2.0, yn + (h / 2.0) * k2)
        k4 = f(tn + h,       yn + h * k3)
        y[n + 1] = yn + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t, y


# 3. RKF45 — adaptif adım, Fehlberg katsayıları (Burden & Faires, Böl. 5.5)
# 4. ve 5. mertebe çözümler aynı 6 k değerini paylaşır; farkları hata tahmini.

# Düğüm katsayıları
_C2 = 1 / 4
_C3 = 3 / 8
_C4 = 12 / 13
_C5 = 1.0
_C6 = 1 / 2

# Aşama katsayıları
_A21 = 1 / 4

_A31 = 3 / 32
_A32 = 9 / 32

_A41 = 1932 / 2197
_A42 = -7200 / 2197
_A43 = 7296 / 2197

_A51 = 439 / 216
_A52 = -8.0
_A53 = 3680 / 513
_A54 = -845 / 4104

_A61 = -8 / 27
_A62 = 2.0
_A63 = -3544 / 2565
_A64 = 1859 / 4104
_A65 = -11 / 40

# 4. mertebe ağırlıkları
_B1_4 = 25 / 216
_B3_4 = 1408 / 2565
_B4_4 = 2197 / 4104
_B5_4 = -1 / 5

# 5. mertebe ağırlıkları
_B1_5 = 16 / 135
_B3_5 = 6656 / 12825
_B4_5 = 28561 / 56430
_B5_5 = -9 / 50
_B6_5 = 2 / 55


def rkf45(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    y0: np.ndarray,
    tol: float = 1e-6,
    h_init: float = 0.1,
    h_min: float = 1e-6,
    h_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # h_new = 0.84 * (tol/err)^(1/4) * h — adım uyarlama; çıktı t dizisi düzensizdir
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    t_list = [t0]
    y_list = [y0.copy()]

    t = t0
    y = y0.copy()
    h = h_init

    while t < tf:
        # Son adımı tam tf'e götür
        if t + h > tf:
            h = tf - t

        k1 = f(t,            y)
        k2 = f(t + _C2 * h,  y + h * (_A21 * k1))
        k3 = f(t + _C3 * h,  y + h * (_A31 * k1 + _A32 * k2))
        k4 = f(t + _C4 * h,  y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3))
        k5 = f(t + _C5 * h,  y + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4))
        k6 = f(t + _C6 * h,  y + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5))

        # 4. ve 5. mertebe tahminler
        y4 = y + h * (_B1_4 * k1 + _B3_4 * k3 + _B4_4 * k4 + _B5_4 * k5)
        y5 = y + h * (_B1_5 * k1 + _B3_5 * k3 + _B4_5 * k4 + _B5_5 * k5 + _B6_5 * k6)

        # Yerel hata tahmini (vektör için sonsuz norm)
        err = np.linalg.norm(y5 - y4, ord=np.inf)

        # Adım kabul / red kararı
        if err <= tol or h <= h_min:
            # Adımı kabul et — daha yüksek mertebeli çözümü kullan (local extrapolation)
            t += h
            y = y5
            t_list.append(t)
            y_list.append(y.copy())

        # Yeni adım önerisi
        if err == 0.0:
            h = h_max
        else:
            s = 0.84 * (tol / err) ** 0.25
            h = max(h_min, min(h_max, h * s))

    return np.array(t_list), np.array(y_list)


# 4. Adams-Bashforth-Moulton 4. mertebe — Predictor-Corrector
# İlk 3 adım RK4 ile başlatılır (self-starting değil). Adım başına 2 fev.
def adams_bashforth_moulton(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    y0: np.ndarray,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_steps = int(np.ceil((tf - t0) / h))
    t = np.linspace(t0, t0 + n_steps * h, n_steps + 1)

    y = np.zeros((n_steps + 1, y0.size))
    y[0] = y0

    # f-değerleri arabelleği
    F = np.zeros((n_steps + 1, y0.size))
    F[0] = f(t[0], y[0])

    # İlk 3 adımı RK4 ile başlat (AB4 için 4 geçmiş noktaya ihtiyaç var)
    start_steps = min(3, n_steps)
    for n in range(start_steps):
        tn, yn = t[n], y[n]
        k1 = f(tn,           yn)
        k2 = f(tn + h / 2.0, yn + (h / 2.0) * k1)
        k3 = f(tn + h / 2.0, yn + (h / 2.0) * k2)
        k4 = f(tn + h,       yn + h * k3)
        y[n + 1] = yn + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        F[n + 1] = f(t[n + 1], y[n + 1])

    # AB4 predictor + AM3 corrector (P-E-C-E)
    for n in range(3, n_steps):
        # Predictor (Adams-Bashforth 4. mertebe)
        y_pred = y[n] + (h / 24.0) * (
            55.0 * F[n] - 59.0 * F[n - 1] + 37.0 * F[n - 2] - 9.0 * F[n - 3]
        )

        f_pred = f(t[n + 1], y_pred)

        # Corrector (Adams-Moulton 3. mertebe)
        y[n + 1] = y[n] + (h / 24.0) * (
            9.0 * f_pred + 19.0 * F[n] - 5.0 * F[n - 1] + F[n - 2]
        )

        F[n + 1] = f(t[n + 1], y[n + 1])

    return t, y


# İsim → fonksiyon eşleştirmesi
METHODS = {
    "euler": euler,
    "rk4": rk4,
    "rkf45": rkf45,
    "abm": adams_bashforth_moulton,
}

