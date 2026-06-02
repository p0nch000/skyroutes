from dataclasses import replace
from main import Config, correr_replicas, agregar


# ESCENARIOS
def definir_escenarios():
    base = Config()
    return {
        "Base (3 drones)":        base,                                  # configuracion actual del caso
        "Mayor demanda (l=0.30)": replace(base, lambda_llegadas=0.30),   # mas pedidos por minuto
        "Mas drones (4)":         replace(base, c_drones=4),             # ataca el cuello de botella
    }


# Corre las 30 replicas por cada escenario y guarda el resumen agregado (media + IC)
def correr_todos(n_replicas=30):
    resultados = {}
    for nombre, cfg in definir_escenarios().items():
        filas = correr_replicas(cfg, n_replicas=n_replicas)  # 30 dias independientes
        resultados[nombre] = agregar(filas)  # promedio +/- IC95 por metrica
    return resultados


# TABLA COMPARATIVA: una fila por metrica, una columna por escenario
def imprimir_tabla(resultados):
    escenarios = list(resultados.keys())

    filas = [
        ("rho_drones",     "Utilizacion drones",   True),
        ("rho_prep",       "Utilizacion prep",     True),
        ("Wq",             "Espera en cola (min)", False),
        ("W",              "Total sistema (min)",  False),
        ("Lq_drones",      "Cola drones (ped)",    False),
        ("throughput",     "Throughput (ped/h)",   False),
        ("completados",    "Completados",          False),
        ("pct_fallas",     "% fallas",             False),
        ("pct_retrasados", "% retrasados (W>20)",  False),
    ]

    ancho_label = 22
    ancho_col = 24

    print(f"{'Metrica':<{ancho_label}}", end="")
    for e in escenarios:
        print(f"{e:>{ancho_col}}", end="")
    print()
    print("-" * (ancho_label + ancho_col * len(escenarios)))

    for clave, etiqueta, es_pct in filas:
        print(f"{etiqueta:<{ancho_label}}", end="")
        for e in escenarios:
            media, ic = resultados[e][clave]
            if es_pct:  
                celda = f"{media*100:.1f}% +/- {ic*100:.1f}"
            else:
                celda = f"{media:.2f} +/- {ic:.2f}"
            print(f"{celda:>{ancho_col}}", end="")
        print()


if __name__ == "__main__":
    resultados = correr_todos(n_replicas=30)
    print("=== TABLA COMPARATIVA DE ESCENARIOS (30 replicas, IC 95%) ===\n")
    imprimir_tabla(resultados)