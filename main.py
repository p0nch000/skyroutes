import heapq
from collections import deque
from dataclasses import dataclass, replace
import numpy as np


# CONFIG: Todos los parametros del caso en un solo lugar
@dataclass
class Config:
    # LLEGADAS
    lambda_llegadas: float = 0.20  # pedidos/min E(Ta) = 1/0.20 = 5 min

    # Etapa 1: Preparacion
    c_prep: int = 2  # Numero de estaciones
    mu_prep: float = 1 / 3  # Tasa por estacion E(Tp) = 1/mu = 3 min

    # Etapa 2: Entrega
    c_drones: int = 3  # Numero de drones
    entrega_a: float = 8.0  # Uniforme(a,b) en min
    entrega_b: float = 14.0

    # Fallas
    prob_falla: float = 0.08  # Bernoulli(p)
    retraso_falla_media: float = 5.0  # Exponencial

    # Corrida
    t_sim: float = 480.0  # Jornada de 8h
    t_warmup: float = 60.0  # Tiempo de calentamiento que NO se mide
    seed: int = 42
    umbral_retraso: float = 20.0  # W > 20 min cuenta como "retrasado"


# PEDIDO: Lleva sus propias marcas de tiempo
@dataclass
class Pedido:
    id: int
    t_llegada: float
    t_inicio_prep: float = None
    t_fin_prep: float = None
    t_inicio_entrega: float = None
    t_fin_entrega: float = None
    tuvo_falla: bool = False


# SIMULADOR
class Simulador:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Estado del motor
        self.clock = 0.0
        self.fel = []  # Future Event List -> (tiempo, seq, tipo, pedido)
        self.seq = 0  # Desempate para heap

        # Estado del sistema
        self.prep_ocupadas = 0  # Estaciones en uso
        self.drones_ocupados = 0  # Drones en uso
        self.cola_prep = deque()  # Pedidos esperando estacion
        self.cola_drones = deque()  # Pedidos esperando dron

        # Contadores
        self.n_creados = 0
        self.n_completados = 0
        self.n_fallas = 0

        # Acumuladores de area (time-weighted)
        self.t_ultimo = 0.0  # Cuando se actualizo area por ultima vez
        self.area_cola_prep = 0.0  # Integral de len(cola_prep) dt
        self.area_cola_drones = 0.0  # Integral de len(cola_drones) dt
        self.area_prep_busy = 0.0  # Integral de prep_ocupadas dt
        self.area_drones_busy = 0.0  # Integral de drones_ocupados dt

        # Registro por pedido
        self.pedidos_terminados = []

        # Control de warmup
        self.warmup_hecho = False  # True una vez que pasamos t_warmup
        self.t_inicio_medicion = 0.0  # Instante real donde empezamos a medir

    # Helpers de aleatorias
    def t_entre_llegadas(self):
        return self.rng.exponential(1 / self.cfg.lambda_llegadas)

    def t_preparacion(self):
        return self.rng.exponential(1 / self.cfg.mu_prep)

    def t_entrega(self):
        base = self.rng.uniform(self.cfg.entrega_a, self.cfg.entrega_b)
        if self.rng.random() < self.cfg.prob_falla:
            base += self.rng.exponential(self.cfg.retraso_falla_media)
            return base, True
        return base, False

    def schedule(self, delay, tipo, pedido=None):
        heapq.heappush(self.fel, (self.clock + delay, self.seq, tipo, pedido))
        self.seq += 1

    def _avanzar_reloj(self, t_nuevo):
        # Acumula area con el estado ACTUAL antes de mover el reloj
        dt = t_nuevo - self.t_ultimo
        self.area_cola_prep += len(self.cola_prep) * dt
        self.area_cola_drones += len(self.cola_drones) * dt
        self.area_prep_busy += self.prep_ocupadas * dt
        self.area_drones_busy += self.drones_ocupados * dt
        self.t_ultimo = t_nuevo

    def _reset_warmup(self):
        # El sistema ya se "lleno", empezamos a medir limpio en t_warmup
        self.area_cola_prep = 0.0
        self.area_cola_drones = 0.0
        self.area_prep_busy = 0.0
        self.area_drones_busy = 0.0
        self.n_completados = 0
        self.n_fallas = 0
        self.pedidos_terminados = []
        self.t_inicio_medicion = self.clock
        self.warmup_hecho = True

    # Loop principal
    def run(self):
        self.schedule(self.t_entre_llegadas(), "llegada")

        while self.fel:
            t, _, tipo, pedido = heapq.heappop(self.fel)
            if t > self.cfg.t_sim:  # Si ya termino la jornada, finalizamos
                break
 
            if not self.warmup_hecho and t >= self.cfg.t_warmup:
                self._avanzar_reloj(self.cfg.t_warmup)
                self.clock = self.cfg.t_warmup
                self._reset_warmup()

            self._avanzar_reloj(t)  # Primero acumular area con el estado viejo
            self.clock = t  # Brinco de reloj

            if tipo == "llegada":
                self._on_llegada()
            elif tipo == "fin_prep":
                self._on_fin_prep(pedido)
            elif tipo == "fin_entrega":
                self._on_fin_entrega(pedido)

        # Cerrar areas hasta el final de la simulacion
        if self.t_ultimo < self.cfg.t_sim:
            self._avanzar_reloj(self.cfg.t_sim)

    # Handlers
    def _on_llegada(self):
        # 1) Crear pedido
        self.n_creados += 1
        p = Pedido(id=self.n_creados, t_llegada=self.clock)

        # 2) Programar siguiente llegada (auto sostenido)
        self.schedule(self.t_entre_llegadas(), "llegada")

        # 3) Meterlo a la cola de preparacion e intentar atenderlo
        self.cola_prep.append(p)
        self._intentar_iniciar_prep()

    def _intentar_iniciar_prep(self):
        # Arranca solo si hay estacion libre y alguien esperando
        if self.prep_ocupadas < self.cfg.c_prep and self.cola_prep:
            p = self.cola_prep.popleft()
            self.prep_ocupadas += 1  # Se ocupa estacion
            p.t_inicio_prep = self.clock
            self.schedule(self.t_preparacion(), "fin_prep", p)

    def _on_fin_prep(self, pedido):
        self.prep_ocupadas -= 1  # Se libera estacion
        pedido.t_fin_prep = self.clock
        self.cola_drones.append(pedido)
        self._intentar_iniciar_entrega()  # Cliente nuevo para drones
        self._intentar_iniciar_prep()  # Estacion libre quiza atienda otro

    def _intentar_iniciar_entrega(self):
        if self.drones_ocupados < self.cfg.c_drones and self.cola_drones:
            p = self.cola_drones.popleft()
            self.drones_ocupados += 1
            p.t_inicio_entrega = self.clock
            dur, hubo_falla = self.t_entrega()
            p.tuvo_falla = hubo_falla
            self.schedule(dur, "fin_entrega", p)

    def _on_fin_entrega(self, pedido):
        self.drones_ocupados -= 1  # Se libera dron
        pedido.t_fin_entrega = self.clock
        self.n_completados += 1
        if pedido.tuvo_falla:
            self.n_fallas += 1
        if self.warmup_hecho:
            self.pedidos_terminados.append(pedido)
        self._intentar_iniciar_entrega()  # Dron libre quiza atienda a otro

    def metricas(self):
        ventana = self.cfg.t_sim - self.t_inicio_medicion

        # Por pedido: Wq y W
        wq = [p.t_inicio_prep - p.t_llegada for p in self.pedidos_terminados]
        w = [p.t_fin_entrega - p.t_llegada for p in self.pedidos_terminados]
        n_retrasados = sum(1 for x in w if x > self.cfg.umbral_retraso)

        # Area / Tiempo = Promedio time-weighted (dividido entre la ventana, no t_sim)
        return {
            "Wq": np.mean(wq) if wq else 0.0,  # Espera en cola
            "W": np.mean(w) if w else 0.0,  # Total en sistema
            "Lq_prep": self.area_cola_prep / ventana,  # Cola prep promedio
            "Lq_drones": self.area_cola_drones / ventana,  # Cola drones promedio
            "rho_prep": self.area_prep_busy / (self.cfg.c_prep * ventana),  # Utilizacion prep
            "rho_drones": self.area_drones_busy / (self.cfg.c_drones * ventana),  # Utilizacion drones
            "completados": self.n_completados,
            "fallas": self.n_fallas,
            "pct_fallas": 100 * self.n_fallas / max(self.n_completados, 1),
            "throughput": self.n_completados / ventana * 60,  # pedidos/h
            "pct_retrasados": 100 * n_retrasados / max(len(w), 1),
        }


# Corre N replicas, cada una con seed distinta (dias independientes)
def correr_replicas(cfg: Config, n_replicas=30):
    filas = []
    for i in range(n_replicas):
        cfg_i = replace(cfg, seed=cfg.seed + i)
        sim = Simulador(cfg_i)
        sim.run()
        filas.append(sim.metricas())
    return filas

def agregar(filas):
    claves = filas[0].keys()
    n = len(filas)
    resumen = {}
    for k in claves:
        vals = np.array([f[k] for f in filas], dtype=float)
        media = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n)  # error estandar de la media
        ic = 1.96 * sem  # 1.96 = z al 95%
        resumen[k] = (media, ic)
    return resumen


if __name__ == "__main__":
    cfg = Config()
    filas = correr_replicas(cfg, n_replicas=30)
    resumen = agregar(filas)

    print(f"=== ESCENARIO BASE  (30 replicas, warmup {cfg.t_warmup:.0f} min) ===\n")
    orden = ["Wq", "W", "Lq_drones", "Lq_prep", "rho_drones", "rho_prep",
             "throughput", "completados", "pct_fallas", "pct_retrasados"]
    for k in orden:
        media, ic = resumen[k]
        if k.startswith("rho"):
            print(f"{k:<16}{media*100:7.2f}%  +/- {ic*100:.2f}")
        else:
            print(f"{k:<16}{media:7.2f}   +/- {ic:.2f}")