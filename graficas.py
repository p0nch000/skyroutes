from dataclasses import replace
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np

from main import Config, Simulador, correr_replicas, agregar
from simulaciones import definir_escenarios, correr_todos

class SimuladorConHistoria(Simulador):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.hist_t = []
        self.hist_cola = []

    def _registrar(self):
        self.hist_t.append(self.clock)
        self.hist_cola.append(len(self.cola_drones))

    def _on_llegada(self):
        super()._on_llegada()
        self._registrar()

    def _on_fin_prep(self, pedido):
        super()._on_fin_prep(pedido)
        self._registrar()

    def _on_fin_entrega(self, pedido):
        super()._on_fin_entrega(pedido)
        self._registrar()


# ----------------------------------------------------------------------
# GRAFICA 1: Barras comparativas de metricas clave entre escenarios,
# con barras de error = intervalo de confianza al 95%
# ----------------------------------------------------------------------
def grafica_barras(resultados, archivo="grafica1_comparativa.png"):
    escenarios = list(resultados.keys())
    colores = ["#4C72B0", "#C44E52", "#55A868"]  # base / colapso / solucion

    # (clave, titulo del panel, factor para escala, unidad)
    paneles = [
        ("rho_drones",     "Utilizacion de drones", 100, "%"),
        ("W",              "Tiempo total en sistema", 1, "min"),
        ("Lq_drones",      "Cola promedio de drones", 1, "pedidos"),
        ("pct_retrasados", "% pedidos retrasados (W>20)", 1, "%"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (clave, titulo, factor, unidad) in zip(axes, paneles):
        medias = [resultados[e][clave][0] * factor for e in escenarios]
        ics = [resultados[e][clave][1] * factor for e in escenarios]
        x = np.arange(len(escenarios))
        ax.bar(x, medias, yerr=ics, capsize=5, color=colores, edgecolor="black", linewidth=0.6)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_ylabel(unidad)
        ax.set_xticks(x)
        ax.set_xticklabels(["Base", "Mayor\ndemanda", "Mas\ndrones"], fontsize=9)
        # etiqueta de valor encima de cada barra
        for xi, m in zip(x, medias):
            ax.text(xi, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SkyRoute: comparacion de escenarios (30 replicas, IC 95%)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(archivo, dpi=150)
    plt.close(fig)
    return archivo


# ----------------------------------------------------------------------
# GRAFICA 2: evolucion de la cola de drones L(t) durante una jornada,
# una linea por escenario. Muestra el colapso vs la estabilidad.
# ----------------------------------------------------------------------
def grafica_evolucion(archivo="grafica2_evolucion_cola.png"):
    colores = {"Base (3 drones)": "#4C72B0",
               "Mayor demanda (l=0.30)": "#C44E52",
               "Mas drones (4)": "#55A868"}

    fig, ax = plt.subplots(figsize=(11, 5))

    for nombre, cfg in definir_escenarios().items():
        sim = SimuladorConHistoria(cfg)  # misma seed base -> comparacion justa
        sim.run()
        # step: la cola es constante entre eventos, por eso escalon
        ax.step(sim.hist_t, sim.hist_cola, where="post",
                label=nombre, color=colores[nombre], linewidth=1.3)

    ax.axvline(60, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(62, ax.get_ylim()[1]*0.92, "fin warmup", fontsize=8, color="gray")
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Pedidos en cola de drones")
    ax.set_title("Evolucion de la cola de drones durante la jornada",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)
    return archivo


if __name__ == "__main__":
    resultados = correr_todos(n_replicas=30)
    a1 = grafica_barras(resultados)
    a2 = grafica_evolucion()
    print("graficas generadas:")
    print(" ", a1)
    print(" ", a2)