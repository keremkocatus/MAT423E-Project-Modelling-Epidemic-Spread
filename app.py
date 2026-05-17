# app.py — MAT423E Proje 1
# İnteraktif SEIR simülatörü — streamlit run app.py

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from methods import euler, rk4, rkf45, adams_bashforth_moulton
from seir_model import SEIRParams, make_seir_rhs, epidemic_summary


# Sayfa konfigürasyonu
st.set_page_config(
    page_title="SEIR Salgın Simülatörü — MAT423E",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Renkler — tüm grafiklerde tutarlı bir kimlik
COLORS = {
    "S": "#1f77b4",    # mavi — duyarlı
    "E": "#ff7f0e",    # turuncu — maruz
    "I": "#d62728",    # kırmızı — enfekte
    "R": "#2ca02c",    # yeşil — iyileşmiş
    "Euler": "#8c564b",
    "RK4":   "#2ca02c",
    "ABM4":  "#1f77b4",
    "RKF45": "#d62728",
}

METHOD_FUNCS = {
    "Euler": euler,
    "RK4": rk4,
    "ABM4": adams_bashforth_moulton,
    # RKF45 farklı imza — ayrı bir kolda çağırıyoruz
}


# Cache'li çözücü — aynı parametrelerle ikinci kez koşmasın
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
    else:
        t, y = METHOD_FUNCS[method](f, 0.0, t_end, y0, h)
        n = len(t) - 1
        n_fev = {"Euler": 1, "RK4": 4}.get(method, 0) * n
        if method == "ABM4":
            n_fev = 4 * min(3, n) + 2 * max(0, n - 3) + 1
    dt = time.perf_counter() - t0
    return t, y, dt, n_fev


# Sidebar — parametre kontrolleri
def render_sidebar() -> dict:
    st.sidebar.title("🦠 SEIR Parametreleri")
    st.sidebar.caption("MAT423E Proje 1 — Salgın & Aşılama")

    # ---- Hızlı senaryolar -------------------------------------------------
    st.sidebar.subheader("⚡ Hızlı Senaryolar")
    preset = st.sidebar.selectbox(
        "Önceden tanımlı senaryolar:",
        ["(Manuel ayar)", "COVID-19 benzeri (aşısız)", "Aşılama kampanyası",
         "İzolasyon / mesafe", "Düşük bulaşıcılık (grip)", "Çok yüksek R₀ (kızamık)"],
        index=0,
    )

    # Preset → değerler
    presets = {
        "COVID-19 benzeri (aşısız)":   dict(beta=0.5, sigma=1/5, gamma=1/7, nu=0.0),
        "Aşılama kampanyası":          dict(beta=0.5, sigma=1/5, gamma=1/7, nu=0.01),
        "İzolasyon / mesafe":          dict(beta=0.25, sigma=1/5, gamma=1/7, nu=0.0),
        "Düşük bulaşıcılık (grip)":    dict(beta=0.3, sigma=1/2, gamma=1/4, nu=0.0),
        "Çok yüksek R₀ (kızamık)":    dict(beta=1.5, sigma=1/10, gamma=1/7, nu=0.0),
    }
    defaults = presets.get(preset, dict(beta=0.5, sigma=1/5, gamma=1/7, nu=0.0))

    # ---- Epidemik parametreler -------------------------------------------
    st.sidebar.subheader("Epidemik Parametreler")
    beta = st.sidebar.slider("β — Bulaşma oranı (1/gün)",
                              0.0, 2.0, defaults["beta"], 0.01,
                              help="Birim zamanda bir duyarlının enfekte olma oranı")
    sigma = st.sidebar.slider("σ — Kuluçka çıkış oranı (1/gün)",
                               0.05, 1.0, defaults["sigma"], 0.01,
                               help="1/σ = ortalama kuluçka süresi (gün)")
    gamma = st.sidebar.slider("γ — İyileşme oranı (1/gün)",
                               0.05, 1.0, defaults["gamma"], 0.01,
                               help="1/γ = ortalama bulaştırıcılık süresi (gün)")
    nu = st.sidebar.slider("ν — Aşılama oranı (1/gün)",
                            0.0, 0.05, defaults["nu"], 0.001,
                            help="Günlük aşılanan duyarlı oranı (0 = kampanya yok)")

    # Türetilmiş büyüklükler — kullanıcıya bilgi ver
    R0 = beta / gamma if gamma > 0 else float("inf")
    incub = 1 / sigma if sigma > 0 else float("inf")
    infect = 1 / gamma if gamma > 0 else float("inf")

    c1, c2 = st.sidebar.columns(2)
    c1.metric("R₀ = β/γ", f"{R0:.2f}")
    c2.metric("Kuluçka", f"{incub:.1f} gün")
    st.sidebar.metric("Bulaştırıcılık süresi", f"{infect:.1f} gün")

    # ---- Popülasyon -------------------------------------------------------
    st.sidebar.subheader("Popülasyon")
    N = st.sidebar.number_input("N — Toplam popülasyon",
                                 min_value=1000, max_value=100_000_000,
                                 value=1_000_000, step=10_000)
    I0 = st.sidebar.number_input("I₀ — Başlangıç enfekte sayısı",
                                  min_value=1, max_value=10000, value=10, step=1)

    # ---- Sayısal çözüm ayarları ------------------------------------------
    st.sidebar.subheader("Sayısal Çözüm")
    t_end = st.sidebar.slider("Simülasyon süresi (gün)", 30, 365, 160, 10)
    h = st.sidebar.select_slider("Adım büyüklüğü h (gün)",
                                  options=[1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01],
                                  value=0.1)
    rkf45_tol = st.sidebar.select_slider(
        "RKF45 toleransı",
        options=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
    )

    # Bilgi: R₀ değerine göre yorum
    if R0 > 1:
        threshold = 1 - 1 / R0
        st.sidebar.info(
            f"📊 **R₀ = {R0:.2f} > 1**: Salgın yayılacak.\n\n"
            f"Sürü bağışıklığı eşiği: **{threshold:.1%}** "
            f"(popülasyonun bu kadarı bağışık olmalı)."
        )
    else:
        st.sidebar.success(
            f"✅ **R₀ = {R0:.2f} ≤ 1**: Salgın söner.\n\n"
            f"Mevcut parametre seti altında hastalık popülasyonda "
            f"kalıcı bir yayılım gösteremez."
        )

    return dict(beta=beta, sigma=sigma, gamma=gamma, nu=nu,
                N=float(N), I0=float(I0),
                t_end=float(t_end), h=h, rkf45_tol=rkf45_tol, R0=R0)


# Grafik yardımcıları
def plot_seir_curves(t, y, N, title, show_compartments=("S", "E", "I", "R")):
    """SEIR eğrilerini tek panelde, opsiyonel bölme seçimi."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = ["S (duyarlı)", "E (maruz)", "I (enfekte)", "R (iyileşmiş)"]
    keys = ["S", "E", "I", "R"]
    for i, (key, label) in enumerate(zip(keys, labels)):
        if key in show_compartments:
            ax.plot(t, y[:, i], color=COLORS[key], lw=2.0, label=label)
    ax.set_xlabel("Zaman (gün)")
    ax.set_ylabel("Kişi sayısı")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    return fig


def plot_methods_overlay(results: dict, compartment_idx: int, comp_name: str, t_end: float):
    """Aynı bölmeyi (varsayılan: I) 4 yöntem için üst üste çiz."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, r in results.items():
        ax.plot(r["t"], r["y"][:, compartment_idx],
                color=COLORS[name], lw=1.6, alpha=0.85, label=name)
    ax.set_xlabel("Zaman (gün)")
    ax.set_ylabel(f"{comp_name} (kişi)")
    ax.set_title(f"Dört yöntemin {comp_name} eğrileri", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, t_end)
    return fig


def plot_scenario_compare(t_a, y_a, label_a, t_b, y_b, label_b, comp_idx, comp_name):
    """İki senaryoyu yan yana karşılaştır (I bölmesi için)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t_a, y_a[:, comp_idx], color="#d62728", lw=2.2, label=label_a)
    ax.plot(t_b, y_b[:, comp_idx], color="#2ca02c", lw=2.2, label=label_b)
    ax.fill_between(t_a, y_a[:, comp_idx], 0, color="#d62728", alpha=0.10)
    ax.fill_between(t_b, y_b[:, comp_idx], 0, color="#2ca02c", alpha=0.10)
    ax.set_xlabel("Zaman (gün)")
    ax.set_ylabel(f"{comp_name} (kişi)")
    ax.set_title(f"Senaryo karşılaştırması — {comp_name}", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    return fig


# Tab 1 — Salgın Eğrisi
def tab_epidemic_curve(params: dict):
    st.subheader("📈 Salgın Eğrisi")
    st.caption("Sol panelden parametreleri değiştir — grafik anında güncellenir.")

    method = st.radio(
        "Görüntülenecek yöntem:",
        list(METHOD_FUNCS.keys()) + ["RKF45"],
        index=1, horizontal=True,
        help="Hızlı keşif için RK4 önerilir. Yöntem karşılaştırması için Tab 2.",
    )

    show = st.multiselect(
        "Gösterilecek bölmeler:",
        ["S", "E", "I", "R"], default=["S", "E", "I", "R"],
    )

    t, y, runtime, n_fev = solve_seir(
        method, params["beta"], params["sigma"], params["gamma"],
        params["nu"], params["N"], params["I0"],
        params["t_end"], params["h"], params["rkf45_tol"],
    )
    summ = epidemic_summary(t, y, params["N"])

    # Üst banner — anahtar metrikler
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Zirve enfekte", f"{summ['peak_I']:,.0f}",
              f"{summ['peak_I'] / params['N']:.1%} popülasyonun")
    m2.metric("Zirve zamanı", f"{summ['peak_time']:.1f} gün")
    m3.metric("Toplam saldırı oranı", f"{summ['attack_rate']:.1%}",
              "tüm enfekte olanlar / N")
    m4.metric("Bağışık olmayan", f"{summ['final_S_frac']:.1%}",
              "popülasyonun")

    fig = plot_seir_curves(
        t, y, params["N"],
        title=f"SEIR çözümü — {method} (h={params['h']}, "
              f"R₀={params['R0']:.2f}, ν={params['nu']}/gün)",
        show_compartments=tuple(show),
    )
    st.pyplot(fig, clear_figure=True)

    # Alt: tahmin oyunu
    with st.expander("🎯 Hands-on aktivite önerisi: zirveyi tahmin et"):
        st.markdown(
            f"""
            **Aktivite akışı (sınıfta 5 dakika):**

            1. Mevcut parametrelerle zirve **{summ['peak_I']:,.0f}** kişi
               oldu (gün {summ['peak_time']:.0f}).
            2. Hocaya/öğrenciye sor: β'yı **2 katına** çıkarırsam zirve
               nasıl değişir? Önce tahmin et, sonra slider'ı oynat.
            3. Tartış: aşılama oranı ν'yü artırdığında zirve **erken**
               mi gelir, **geç** mi? Neden?
            4. Bonus: sürü bağışıklığı eşiği {(1 - 1/params['R0']):.1%}
               (yalnızca R₀ > 1 için anlamlı) — ν'yü ne kadar
               artırırsan saldırı oranı bu eşiğe yaklaşır?
            """
        )

    # Performans / sağlık bilgisi
    with st.expander("ℹ️ Çözümün sayısal sağlığı"):
        totals = y.sum(axis=1)
        conserv = float(np.max(np.abs(totals - params["N"])) / params["N"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Adım sayısı", f"{len(t) - 1:,}")
        c2.metric("Fonksiyon değerlendirmesi", f"{n_fev:,}")
        c3.metric("Korunum bağıl hatası", f"{conserv:.1e}",
                  help="|S+E+I+R - N| / N — sıfıra ne kadar yakınsa o kadar iyi")
        st.caption(f"Çözüm süresi: **{runtime*1000:.1f} ms**")


# Tab 2 — Yöntem Karşılaştırması
def tab_method_comparison(params: dict):
    st.subheader("⚖️ Yöntem Karşılaştırması")
    st.caption(
        "Aynı SEIR sistemi 4 yöntemle çözülür. Sabit-adımlı yöntemler için "
        "aynı h kullanılır; RKF45 adaptif olduğu için tolerans ayarına göre "
        "kendi adımlarını seçer."
    )

    results = {}
    for method in ["Euler", "RK4", "ABM4", "RKF45"]:
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
        "Karşılaştırma için bölme:", ["I (enfekte)", "S (duyarlı)",
                                       "E (maruz)", "R (iyileşmiş)"], index=0,
    )
    comp_idx = {"S (duyarlı)": 0, "E (maruz)": 1,
                "I (enfekte)": 2, "R (iyileşmiş)": 3}[comp_choice]
    fig = plot_methods_overlay(results, comp_idx, comp_choice[0], params["t_end"])
    st.pyplot(fig, clear_figure=True)

    # Tablo
    rows = []
    ref_peak = results["RKF45"]["summary"]["peak_I"]
    for name, r in results.items():
        s = r["summary"]
        diff_peak = abs(s["peak_I"] - ref_peak)
        rows.append({
            "Yöntem": name,
            "Adım sayısı": r["n_steps"],
            "Fonksiyon değerlendirme": r["n_fev"],
            "Süre (ms)": f"{r['runtime']*1000:.2f}",
            "Zirve I (kişi)": f"{s['peak_I']:,.0f}",
            "Zirve günü": f"{s['peak_time']:.2f}",
            "RKF45'e göre zirve farkı": f"{diff_peak:,.0f}",
            "Saldırı oranı %": f"{s['attack_rate']*100:.2f}",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Tabloda neye dikkat etmeli:**

        - **Fonksiyon değerlendirmesi** ≈ asıl hesaplama maliyeti. RK4
          adım başına 4, ABM4 (asıl döngüde) 2, Euler 1, RKF45 6 (her
          kabul edilen adım için).
        - **RKF45'e göre fark** ≈ pratik hata. RKF45 sıkı tolerans ile
          referans gibi davranır; Euler'in farkı en büyük olmalıdır.
        - **Adım sayısı** RKF45'te diğerlerinden farklıdır — adaptif
          olduğu için.
        """
    )


# Tab 3 — Senaryo Karşılaştırması
def tab_scenario(params: dict):
    st.subheader("🔬 Senaryo Karşılaştırması")
    st.caption(
        "Mevcut (sol panelde) parametreleri 'Senaryo A' olarak alır; "
        "burada Senaryo B'yi tanımlayıp ikisini karşılaştırırsın."
    )

    st.markdown("#### Senaryo B — değiştirilecek parametreler")
    c1, c2 = st.columns(2)
    with c1:
        beta_b = st.slider("β (Senaryo B)", 0.0, 2.0, max(0.0, params["beta"] * 0.5), 0.01,
                            key="beta_b")
        nu_b = st.slider("ν (Senaryo B)", 0.0, 0.05, max(params["nu"], 0.01), 0.001,
                          key="nu_b")
    with c2:
        sigma_b = st.slider("σ (Senaryo B)", 0.05, 1.0, params["sigma"], 0.01,
                             key="sigma_b")
        gamma_b = st.slider("γ (Senaryo B)", 0.05, 1.0, params["gamma"], 0.01,
                             key="gamma_b")

    method = st.radio("Çözüm yöntemi:", ["RK4", "RKF45", "ABM4", "Euler"],
                      index=0, horizontal=True, key="scenario_method")

    # A senaryosu = sol panel
    t_a, y_a, _, _ = solve_seir(
        method, params["beta"], params["sigma"], params["gamma"],
        params["nu"], params["N"], params["I0"],
        params["t_end"], params["h"], params["rkf45_tol"],
    )
    summ_a = epidemic_summary(t_a, y_a, params["N"])

    # B senaryosu
    t_b, y_b, _, _ = solve_seir(
        method, beta_b, sigma_b, gamma_b, nu_b, params["N"], params["I0"],
        params["t_end"], params["h"], params["rkf45_tol"],
    )
    summ_b = epidemic_summary(t_b, y_b, params["N"])

    R0_a = params["beta"] / params["gamma"]
    R0_b = beta_b / gamma_b

    label_a = (f"A: β={params['beta']:.2f}, ν={params['nu']:.3f}, "
               f"R₀={R0_a:.2f}")
    label_b = (f"B: β={beta_b:.2f}, ν={nu_b:.3f}, R₀={R0_b:.2f}")

    fig = plot_scenario_compare(t_a, y_a, label_a, t_b, y_b, label_b,
                                 comp_idx=2, comp_name="I (enfekte)")
    st.pyplot(fig, clear_figure=True)

    # Yan yana metrik karşılaştırması
    st.markdown("#### Bulgular")
    cols = st.columns(3)
    cols[0].markdown("**Zirve enfekte (kişi)**")
    cols[0].metric("Senaryo A", f"{summ_a['peak_I']:,.0f}")
    cols[0].metric("Senaryo B", f"{summ_b['peak_I']:,.0f}",
                   f"{(summ_b['peak_I'] - summ_a['peak_I'])/summ_a['peak_I']*100:+.1f}%")

    cols[1].markdown("**Zirve zamanı (gün)**")
    cols[1].metric("Senaryo A", f"{summ_a['peak_time']:.1f}")
    cols[1].metric("Senaryo B", f"{summ_b['peak_time']:.1f}",
                   f"{summ_b['peak_time'] - summ_a['peak_time']:+.1f} gün")

    cols[2].markdown("**Toplam saldırı oranı**")
    cols[2].metric("Senaryo A", f"{summ_a['attack_rate']:.1%}")
    cols[2].metric("Senaryo B", f"{summ_b['attack_rate']:.1%}",
                   f"{(summ_b['attack_rate'] - summ_a['attack_rate'])*100:+.1f} pp")

    # Halk sağlığı yorumu
    delta_peak_pct = (summ_b['peak_I'] - summ_a['peak_I']) / summ_a['peak_I'] * 100
    if delta_peak_pct < -10:
        msg = (f"✅ **B müdahalesi zirveyi %{abs(delta_peak_pct):.0f} azaltıyor.** "
               f"Hastane kapasitesi açısından kazanç büyük.")
    elif delta_peak_pct > 10:
        msg = (f"⚠️ **B durumunda zirve %{delta_peak_pct:.0f} artıyor.** "
               f"Önlem alınmazsa daha kötü bir senaryo.")
    else:
        msg = "ℹ️ İki senaryo zirve seviyesinde benzer."
    st.info(msg)


# Ana akış
def main():
    st.title("🦠 SEIR Salgın Yayılımı Simülatörü")
    st.markdown(
        "**MAT423E — Numerical Solutions of ODEs · Proje 1: Salgın & Aşılama** · "
        "Bu uygulama 4 farklı sayısal yöntemle SEIR + aşılama modelini çözer ve "
        "müdahale senaryolarını karşılaştırır."
    )

    params = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Salgın Eğrisi",
        "⚖️ Yöntem Karşılaştırması",
        "🔬 Senaryo (A/B)",
        "📚 Model & Yöntemler",
    ])

    with tab1:
        tab_epidemic_curve(params)
    with tab2:
        tab_method_comparison(params)
    with tab3:
        tab_scenario(params)
    with tab4:
        st.markdown(
            r"""
            ### Model: SEIR + aşılama

            Kapalı popülasyon (N sabit), homojen karışım varsayımı altında:

            $$
            \begin{aligned}
            \frac{dS}{dt} &= -\beta\,\frac{S\,I}{N} - \nu\,S \\
            \frac{dE}{dt} &=  \beta\,\frac{S\,I}{N} - \sigma\,E \\
            \frac{dI}{dt} &=  \sigma\,E - \gamma\,I \\
            \frac{dR}{dt} &=  \gamma\,I + \nu\,S
            \end{aligned}
            $$

            **Parametreler:**

            - β: etkili bulaşma oranı (1/gün)
            - σ: 1 / kuluçka süresi (1/gün)
            - γ: 1 / bulaştırıcılık süresi (1/gün)
            - ν: günlük aşılama oranı (1/gün)

            **Temel üreme sayısı:** $R_0 = \beta / \gamma$.
            R₀ > 1 ise salgın yayılır. Sürü bağışıklığı eşiği $1 - 1/R_0$.

            ### Sayısal Yöntemler

            | Yöntem | Tip | Mertebe | Adım/çağrı | Not |
            |--------|-----|---------|------------|-----|
            | Euler | Tek-adım | 1 | 1 fev | Referans / baseline |
            | RK4   | Tek-adım | 4 | 4 fev | Klasik yüksek doğruluk |
            | RKF45 | Tek-adım, adaptif | 4(5) | 6 fev/kabul | Hata kontrollü |
            | ABM4  | Çok-adım P-C | 4 | 2 fev/asıl döngü | Uzun süre ucuz |

            Tüm yöntemler `methods.py` içinde **sıfırdan** implement edildi.
            scipy.solve_ivp yalnızca `validation.py`'de **referans çözüm**
            olarak kullanılmıştır (proje kuralı).

            ### Hands-on Akış (Üye 4 ile koordine)

            1. **Tab 1**'de bir başlangıç parametresi seç.
            2. β'yı değiştir → R₀ banner'ı izle → zirveyi tahmin et.
            3. ν'yü artır → aşılamanın zirveyi nasıl ezdiğini gör.
            4. **Tab 3**'te A/B karşılaştırması yap → halk sağlığı yorumu oku.
            5. **Tab 2**'de Euler ve RK4 farkını gözlemle → numerik mertebenin
               önemini tartış.

            """
        )


main()