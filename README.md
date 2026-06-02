# SkyRoutes
Simulación de operación de entregas con drones y visualización con Streamlit.

## Objetivo del proyecto
Modelar una operación de entregas con dos etapas (preparación y entrega), medir desempeño del sistema (colas, utilización, tiempos) y comparar escenarios de capacidad/demanda.

## Estructura
- `dashboard.py`: dashboard interactivo en Streamlit.
- `main.py`: motor de simulación de eventos discretos.
- `simulaciones.py`: corridas por escenarios y tabla comparativa.
- `graficas.py`: generación de gráficas estáticas en PNG.

## Requisitos
- Python 3.10 o superior.
- `pip` disponible.

## Instalación (recomendada)
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Cómo ejecutar
### 1) Dashboard interactivo
```bash
python3 -m streamlit run dashboard.py
```
Luego abrir el navegador en la URL que muestra Streamlit (normalmente `http://localhost:8501`).

### 2) Simulaciones de escenarios (consola)
```bash
python3 simulaciones.py
```
Imprime una tabla comparativa con medias e intervalos de confianza al 95%.

### 3) Generación de gráficas estáticas
```bash
python3 graficas.py
```
Genera archivos PNG en el directorio del proyecto:
- `grafica1_comparativa.png`
- `grafica2_evolucion_cola.png`

## Nota rápida de uso
Si cierras la terminal o abres una nueva, vuelve a activar el entorno virtual antes de correr el proyecto:
```bash
source .venv/bin/activate
```
