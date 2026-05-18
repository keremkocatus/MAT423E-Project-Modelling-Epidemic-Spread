import argparse
import time
from typing import Optional
import matplotlib.pyplot as plt
from methods import euler, rk4, rkf45, adams_bashforth_moulton
from seir_model import SEIRParams, make_seir_rhs, epidemic_summary


def run_all_methods(
    params: SEIRParams,
    t_end: float = 160.0,
    h: float = 0.1,
    rkf45_tol: float = 1e-6,
    methods: Optional[list[str]] = None,
) -> dict:
    
    if methods is None:
        methods = ["euler", "rk4", "rkf45", "abm"]

    f = make_seir_rhs(params)
    y0 = params.y0
    results: dict = {}

    if "euler" in methods:
        t0 = time.perf_counter()
        t, y = euler(f, 0.0, t_end, y0, h)
        dt = time.perf_counter() - t0
        results["Euler"] = {
            "t": t, "y": y,
            "runtime": dt,
            "n_steps": len(t) - 1,
            "n_fevals": len(t) - 1,             # adım başına 1 değerlendirme
            "summary": epidemic_summary(t, y, params.N),
        }

    if "rk4" in methods:
        t0 = time.perf_counter()
        t, y = rk4(f, 0.0, t_end, y0, h)
        dt = time.perf_counter() - t0
        results["RK4"] = {
            "t": t, "y": y,
            "runtime": dt,
            "n_steps": len(t) - 1,
            "n_fevals": 4 * (len(t) - 1),        # adım başına 4 değerlendirme
            "summary": epidemic_summary(t, y, params.N),
        }

    if "abm" in methods:
        t0 = time.perf_counter()
        t, y = adams_bashforth_moulton(f, 0.0, t_end, y0, h)
        dt = time.perf_counter() - t0
        # ABM: ilk 3 adım RK4 (12 fev) + sonraki adımlar başına 2 fev (predictor + corrector)
        n = len(t) - 1
        n_fev = 4 * min(3, n) + 2 * max(0, n - 3) + 1  # +1 başlangıç f(t0,y0)
        results["ABM4"] = {
            "t": t, "y": y,
            "runtime": dt,
            "n_steps": n,
            "n_fevals": n_fev,
            "summary": epidemic_summary(t, y, params.N),
        }

    if "rkf45" in methods:
        t0 = time.perf_counter()
        t, y = rkf45(f, 0.0, t_end, y0, tol=rkf45_tol,
                     h_init=h, h_min=1e-6, h_max=max(1.0, 5 * h))
        dt = time.perf_counter() - t0
        # adım başına 6 fonksiyon değerlendirmesi (red edilenler sayılmıyor)
        results["RKF45"] = {
            "t": t, "y": y,
            "runtime": dt,
            "n_steps": len(t) - 1,
            "n_fevals": 6 * (len(t) - 1),
            "summary": epidemic_summary(t, y, params.N),
        }

    return results


def print_results_table(results: dict) -> None:
    print(f"\n{'Yöntem':<8s} | {'Adım':>6s} | {'#fev':>7s} | "
          f"{'Süre (ms)':>10s} | {'Zirve I':>10s} | {'Zirve gün':>10s} | "
          f"{'Saldırı %':>10s} | {'Korunum hata':>13s}")
    print("-" * 95)
    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<8s} | {r['n_steps']:>6d} | {r['n_fevals']:>7d} | "
              f"{r['runtime']*1000:>10.2f} | {s['peak_I']:>10,.0f} | "
              f"{s['peak_time']:>10.2f} | {s['attack_rate']*100:>10.2f} | "
              f"{s['conserv_err']:>13.2e}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SEIR + aşılama modeli — 4 sayısal yöntemle çözücü.")
    p.add_argument("--beta",  type=float, default=0.5,    help="Bulaşma oranı (1/gün)")
    p.add_argument("--sigma", type=float, default=1/5,    help="Kuluçka çıkış oranı (1/gün)")
    p.add_argument("--gamma", type=float, default=1/7,    help="İyileşme oranı (1/gün)")
    p.add_argument("--nu",    type=float, default=0.0,    help="Aşılama oranı (1/gün)")
    p.add_argument("--N",     type=float, default=1e6,    help="Toplam popülasyon")
    p.add_argument("--I0",    type=float, default=10.0,   help="Başlangıç enfekte sayısı")
    p.add_argument("--t-end", type=float, default=160.0,  help="Simülasyon süresi (gün)")
    p.add_argument("--h",     type=float, default=0.1,    help="Sabit adım büyüklüğü (gün)")
    p.add_argument("--tol",   type=float, default=1e-6,   help="RKF45 toleransı")
    p.add_argument("--plot",  action="store_true", help="Sonuçları matplotlib ile çiz")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    params = SEIRParams(
        beta=args.beta, sigma=args.sigma, gamma=args.gamma, nu=args.nu, N=args.N,
        S0=args.N - args.I0, E0=0.0, I0=args.I0, R0_init=0.0,
    )

    print(params.summary())

    results = run_all_methods(params, t_end=args.t_end, h=args.h, rkf45_tol=args.tol)
    print_results_table(results)

    if args.plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        compartments = ["S (duyarlı)", "E (maruz)", "I (enfekte)", "R (iyileşmiş)"]
        for i, ax in enumerate(axes.ravel()):
            for name, r in results.items():
                ax.plot(r["t"], r["y"][:, i], label=name, lw=1.5, alpha=0.85)
            ax.set_ylabel(compartments[i])
            ax.grid(alpha=0.3)
            if i >= 2:
                ax.set_xlabel("Zaman (gün)")
        axes[0, 0].legend(loc="best", fontsize=9)
        fig.suptitle(
            f"SEIR çözümü — 4 yöntem karşılaştırması "
            f"(β={args.beta}, γ={args.gamma:.3f}, σ={args.sigma:.3f}, "
            f"ν={args.nu}, R₀={params.R0_basic:.2f})",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = "seir_comparison.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"\nGrafik kaydedildi: {out_path}")

