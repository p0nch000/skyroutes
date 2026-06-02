
import time
import io
from bisect import bisect_right
from dataclasses import replace
from functools import lru_cache
from urllib.request import Request, urlopen

import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
import streamlit as st

from main import Config, Simulador, correr_replicas, agregar

# PALETA: fondo oscuro + acentos mas calidos/organicos
BG = "#0E1117"        # fondo
PANEL = "#161A23"     # paneles
TXT = "#E6EAF2"       # texto
MUTED = "#8A93A6"     # texto tenue
LIBRE = "#6FCF97"     # recurso libre (verde)
OCUPADO = "#F2994A"   # recurso ocupado (naranja)
ACENTO = "#56CCF2"    # acento (azul)
COLORES_ESC = ["#56CCF2", "#F2994A", "#6FCF97"]  # base / colapso / solucion
ICON_DRONE_URL = "https://png.pngtree.com/png-vector/20191030/ourmid/pngtree-inspiration-camera-drone-logo-template-design-vector-emblem-design-concept-creative-png-image_1924973.jpg"
ICON_DRONE_FALLBACK_URL = "https://w7.pngwing.com/pngs/240/852/png-transparent-tech-drone-flying-camera-attached-modern-black-drone.png"
ICON_PREP_URL = "https://w7.pngwing.com/pngs/49/117/png-transparent-computer-icons-order-picking-text-task-symbol-thumbnail.png"

def _estilo_ejes(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    ax.grid(alpha=0.15, color=MUTED)

# SIMULADOR CON HISTORIA: registra un snapshot del estado en cada evento.
class SimuladorConHistoria(Simulador):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.snap_t = []            # tiempos de cada snapshot
        self.snap_prep = []         # estaciones ocupadas
        self.snap_drones = []       # drones ocupados
        self.snap_cola_prep = []    # largo cola prep
        self.snap_cola_drones = []  # largo cola drones

    def _snapshot(self):
        self.snap_t.append(self.clock)
        self.snap_prep.append(self.prep_ocupadas)
        self.snap_drones.append(self.drones_ocupados)
        self.snap_cola_prep.append(len(self.cola_prep))
        self.snap_cola_drones.append(len(self.cola_drones))

    def _on_llegada(self):
        super()._on_llegada()
        self._snapshot()

    def _on_fin_prep(self, pedido):
        super()._on_fin_prep(pedido)
        self._snapshot()

    def _on_fin_entrega(self, pedido):
        super()._on_fin_entrega(pedido)
        self._snapshot()

    def estado_en(self, t):
        # Devuelve el estado vigente en el instante t (ultimo snapshot <= t).
        # bisect = busqueda binaria, O(log n), para que la animacion sea fluida.
        i = bisect_right(self.snap_t, t) - 1
        if i < 0:
            return 0, 0, 0, 0
        return (self.snap_prep[i], self.snap_drones[i],
                self.snap_cola_prep[i], self.snap_cola_drones[i])


def config_desde_ui(c_prep, c_drones, lam, prob_falla, t_sim, seed):
    return Config(
        lambda_llegadas=lam,
        c_prep=c_prep,
        c_drones=c_drones,
        prob_falla=prob_falla,
        t_sim=float(t_sim),
        seed=seed,
    )

# Cache: no re-corre las 30 replicas si los parametros no cambiaron
@st.cache_data(show_spinner=False)
def metricas_cacheadas(cfg_dict, n_replicas):
    cfg = Config(**cfg_dict)
    filas = correr_replicas(cfg, n_replicas=n_replicas)
    return agregar(filas)

# GRAFICA: evolucion de ambas colas L(t) para la corrida actual
def fig_evolucion(sim):
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=BG)
    ax.step(sim.snap_t, sim.snap_cola_drones, where="post",
            color=OCUPADO, linewidth=1.4, label="Cola drones")
    ax.step(sim.snap_t, sim.snap_cola_prep, where="post",
            color=LIBRE, linewidth=1.2, label="Cola prep", alpha=0.9)
    ax.axvline(sim.cfg.t_warmup, color=MUTED, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Pedidos en cola")
    ax.set_title("Evolucion de las colas durante la jornada")
    leg = ax.legend(loc="upper left", fontsize=8, facecolor=PANEL, edgecolor=MUTED)
    for txt in leg.get_texts():
        txt.set_color(TXT)
    _estilo_ejes(ax)
    fig.tight_layout()
    return fig

# GRAFICA: barras comparativas de los 3 escenarios con IC95
def fig_escenarios(resultados):
    escenarios = list(resultados.keys())
    paneles = [
        ("rho_drones", "Utilizacion drones", 100, "%"),
        ("W", "Tiempo en sistema", 1, "min"),
        ("Lq_drones", "Cola drones", 1, "ped"),
        ("pct_retrasados", "% retrasados", 1, "%"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), facecolor=BG)
    for ax, (clave, titulo, factor, unidad) in zip(axes, paneles):
        medias = [resultados[e][clave][0] * factor for e in escenarios]
        ics = [resultados[e][clave][1] * factor for e in escenarios]
        x = np.arange(len(escenarios))
        ax.bar(x, medias, yerr=ics, capsize=4, color=COLORES_ESC,
               edgecolor=TXT, linewidth=0.5)
        ax.set_title(titulo, fontsize=10)
        ax.set_ylabel(unidad, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Base", "Demanda", "+Dron"], fontsize=8)
        _estilo_ejes(ax)
    fig.tight_layout()
    return fig


# ANIMACION: flujo operativo claro en 4 etapas.
def _color_estado(frac):
    if frac >= 0.85:
        return "#EB5757"
    if frac >= 0.55:
        return OCUPADO
    return LIBRE

@lru_cache(maxsize=8)
def _load_icon_from_url(url):
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": "https://www.google.com/",
            },
        )
        with urlopen(req, timeout=8) as r:
            data = r.read()
        icon = Image.open(io.BytesIO(data)).convert("RGBA")
        icon.thumbnail((96, 96), Image.Resampling.LANCZOS)
        return np.asarray(icon)
    except Exception:
        return None
def _load_first_available_icon(*urls):
    for url in urls:
        icon = _load_icon_from_url(url)
        if icon is not None:
            return icon
    return None


def _draw_flow_card(ax, x, y, w, h, title, main_text, frac, footer, icon_img=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=PANEL,
            edgecolor=MUTED,
            linewidth=0.8,
        )
    )
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", color=TXT, fontsize=9, fontweight="bold")
    if icon_img is not None:
        icon_artist = AnnotationBbox(
            OffsetImage(icon_img, zoom=0.28),
            (x + 0.34, y + h - 0.35),
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(icon_artist)
    ax.text(x + w / 2, y + h / 2 + 0.10, main_text, ha="center", va="center", color=TXT, fontsize=12, fontweight="bold")

    bar_x = x + 0.20
    bar_y = y + 0.45
    bar_w = w - 0.40
    bar_h = 0.18
    ax.add_patch(
        Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="#202635", edgecolor=MUTED, linewidth=0.5)
    )
    if frac > 0:
        ax.add_patch(
            Rectangle((bar_x, bar_y), bar_w * min(frac, 1.0), bar_h, facecolor=_color_estado(frac), edgecolor="none")
        )
    ax.text(x + w / 2, y + 0.16, footer, ha="center", va="center", color=MUTED, fontsize=8)


def fig_animacion(sim, t):
    prep_busy, drones_busy, q_prep, q_drones = sim.estado_en(t)
    cfg = sim.cfg
    frac_prep = prep_busy / max(cfg.c_prep, 1)
    frac_drones = drones_busy / max(cfg.c_drones, 1)
    frac_q_prep = q_prep / max(cfg.c_prep * 3, 1)
    frac_q_drones = q_drones / max(cfg.c_drones * 3, 1)
    icon_prep = _load_first_available_icon(ICON_PREP_URL)
    icon_drone = _load_first_available_icon(ICON_DRONE_URL, ICON_DRONE_FALLBACK_URL)

    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    ax.text(0.35, 3.88, f"Tiempo: {t:6.1f} min", color=ACENTO, fontsize=12, fontweight="bold")
    ax.text(6.5, 3.88, "Flujo actual del sistema", color=TXT, fontsize=11, ha="center")

    y = 0.7
    w = 2.75
    h = 2.55
    xs = [0.35, 3.55, 6.75, 9.95]

    _draw_flow_card(
        ax,
        xs[0],
        y,
        w,
        h,
        "Cola preparación",
        f"{q_prep} pedidos",
        frac_q_prep,
        f"carga {min(frac_q_prep, 1.0)*100:.0f}%",
        icon_img=icon_prep,
    )
    _draw_flow_card(
        ax,
        xs[1],
        y,
        w,
        h,
        "Preparación",
        f"{prep_busy}/{cfg.c_prep} ocupadas",
        frac_prep,
        f"uso {frac_prep*100:.0f}%",
        icon_img=icon_prep,
    )
    _draw_flow_card(
        ax,
        xs[2],
        y,
        w,
        h,
        "Cola drones",
        f"{q_drones} pedidos",
        frac_q_drones,
        f"carga {min(frac_q_drones, 1.0)*100:.0f}%",
        icon_img=icon_drone,
    )
    _draw_flow_card(
        ax,
        xs[3],
        y,
        w,
        h,
        "Drones",
        f"{drones_busy}/{cfg.c_drones} en vuelo",
        frac_drones,
        f"uso {frac_drones*100:.0f}%",
        icon_img=icon_drone,
    )

    for i in range(3):
        x0 = xs[i] + w + 0.1
        x1 = xs[i + 1] - 0.1
        ax.annotate("", xy=(x1, y + h / 2), xytext=(x0, y + h / 2),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    fig.tight_layout()
    return fig

# INTERFAZ
def main():
    st.set_page_config(page_title="SkyRoute Sim", layout="wide",
                       initial_sidebar_state="expanded")

    st.markdown(f"""
        <style>
        .stApp {{ background-color: {BG}; color: {TXT}; }}
        section[data-testid="stSidebar"] {{ background-color: {PANEL}; }}
        .stTabs [data-baseweb="tab-list"] {{
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 1.15rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            padding: 0.4rem 1.1rem;
            background-color: #202635;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #2C3B53;
            color: {TXT};
            border: 1px solid {ACENTO};
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 0.55rem;
        }}
        h1.app-title {{
            margin: 0 0 1rem 0;
            text-align: center;
            text-transform: lowercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            color: {TXT};
        }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='app-title'>skyroutes logistics</h1>", unsafe_allow_html=True)

    # ---- SIDEBAR: recursos y parametros modificables ----
    with st.sidebar:
        st.header("Recursos")
        c_prep = st.slider("Estaciones de preparacion", 1, 6, 2)
        c_drones = st.slider("Drones", 1, 8, 3)

        st.header("Demanda y fallas")
        lam = st.slider("Tasa de llegada λ (ped/min)", 0.05, 0.50, 0.20, 0.01)
        prob_falla = st.slider("Probabilidad de falla", 0.0, 0.30, 0.08, 0.01)

        st.header("Corrida")
        t_sim = st.select_slider("Jornada (min)", [240, 480, 720, 960], 480)
        n_replicas = st.slider("Replicas (para IC)", 5, 50, 30, 5)
        seed = st.number_input("Seed base", 0, 9999, 42)

    cfg = config_desde_ui(c_prep, c_drones, lam, prob_falla, t_sim, seed)

    # ---- indicador de carga teorica (sanity check en vivo) ----
    e_servicio = (cfg.entrega_a + cfg.entrega_b) / 2 + cfg.prob_falla * cfg.retraso_falla_media
    cap_drones = cfg.c_drones / e_servicio          # pedidos/min que aguantan los drones
    rho_teorico = cfg.lambda_llegadas / cap_drones
    if rho_teorico >= 1:
        st.error(f"ρ drones teorico ≈ {rho_teorico:.2f} → sistema saturado (la cola crece sin parar)")
    elif rho_teorico >= 0.85:
        st.warning(f"ρ drones teorico ≈ {rho_teorico:.2f} → al limite, poco margen")
    else:
        st.success(f"ρ drones teorico ≈ {rho_teorico:.2f} → sistema estable")

    # ---- METRICAS (30 replicas con IC) ----
    res = metricas_cacheadas(cfg.__dict__, n_replicas)
    st.subheader("Metricas (media ± IC 95%)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Utiliz. drones", f"{res['rho_drones'][0]*100:.1f}%", f"± {res['rho_drones'][1]*100:.1f}")
    col2.metric("Tiempo en sistema", f"{res['W'][0]:.1f} min", f"± {res['W'][1]:.1f}")
    col3.metric("Cola drones", f"{res['Lq_drones'][0]:.2f}", f"± {res['Lq_drones'][1]:.2f}")
    col4.metric("% retrasados", f"{res['pct_retrasados'][0]:.1f}%", f"± {res['pct_retrasados'][1]:.1f}")
    col1.metric("Utiliz. prep", f"{res['rho_prep'][0]*100:.1f}%", f"± {res['rho_prep'][1]*100:.1f}")
    col2.metric("Espera en cola", f"{res['Wq'][0]:.2f} min", f"± {res['Wq'][1]:.2f}")
    col3.metric("Throughput", f"{res['throughput'][0]:.1f}/h", f"± {res['throughput'][1]:.1f}")
    col4.metric("% fallas", f"{res['pct_fallas'][0]:.1f}%", f"± {res['pct_fallas'][1]:.1f}")

    # ---- una corrida con historia, para graficas y animacion ----
    sim = SimuladorConHistoria(cfg)
    sim.run()

    tab_evo, tab_anim, tab_esc = st.tabs(["Evolucion", "Animacion", "Escenarios"])

    with tab_evo:
        st.pyplot(fig_evolucion(sim), use_container_width=True)
        st.caption("Una jornada representativa. Linea punteada = fin del warmup. "
                   "Si la cola de drones nunca baja a cero, el sistema esta saturado.")

    with tab_anim:
        st.write("Flujo en 4 etapas con niveles de carga y uso para leer rápido el estado operativo.")
        colb1, colb2 = st.columns([1, 3])
        velocidad = colb2.slider("Velocidad (frames/seg)", 2, 30, 12)
        placeholder = st.empty()
        if colb1.button("Reproducir"):
            # muestreamos el tiempo en pasos fijos (no cada evento) para fluidez
            paso = 2.0  # minutos por frame
            for t in np.arange(0, cfg.t_sim + paso, paso):
                placeholder.pyplot(fig_animacion(sim, t), use_container_width=True)
                time.sleep(1.0 / velocidad)
        else:
            # estado final como vista previa estatica
            placeholder.pyplot(fig_animacion(sim, cfg.t_sim), use_container_width=True)

    with tab_esc:
        st.write("Compara la configuracion base contra mayor demanda y un dron extra.")
        if st.button("Correr 3 escenarios"):
            base = Config()
            escenarios = {
                "Base (3 drones)": base,
                "Mayor demanda (λ=0.30)": replace(base, lambda_llegadas=0.30),
                "Mas drones (4)": replace(base, c_drones=4),
            }
            resultados = {}
            barra = st.progress(0.0)
            for i, (nombre, c) in enumerate(escenarios.items()):
                resultados[nombre] = agregar(correr_replicas(c, n_replicas=n_replicas))
                barra.progress((i + 1) / len(escenarios))
            st.pyplot(fig_escenarios(resultados), use_container_width=True)


if __name__ == "__main__":
    main()