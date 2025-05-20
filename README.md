# Analizador de Redes Transaccionales

Esta aplicación permite visualizar y analizar redes transaccionales a partir de datos en formato CSV. Está diseñada específicamente para equipos de riesgos y compliance que necesitan analizar patrones de transacciones y identificar nodos centrales para revisión detallada.

## Características

- Visualización interactiva de redes transaccionales
- Detección automática de comunidades
- Cálculo de métricas de centralidad
- Análisis detallado de nodos individuales
- Visualización de conexiones entrantes y salientes
- Estadísticas generales de la red
- **Sistema de Scoring de Riesgo**: Calcula un puntaje de riesgo para cada componente basado en indicadores como tamaño, densidad, montos, patrones de transacción y reciprocidad.
- **Filtros Mejorados**: Permiten filtrar por tamaño de componente y score de riesgo en la barra lateral.
- **Panel de Componentes de Alto Riesgo**: Muestra componentes que superan un umbral de riesgo con métricas detalladas.

## Requisitos del archivo CSV

El archivo CSV debe contener las siguientes columnas:
- `origen`: ID del nodo que envía la transacción
- `destino`: ID del nodo que recibe la transacción
- `monto_total`: Monto total transado entre el origen y destino
- `cantidad_envios`: Número de transacciones realizadas entre el origen y destino

## Instalación

1. Clone este repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Cree un entorno virtual (opcional pero recomendado):
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instale las dependencias:
```bash
python3 -m pip install -r requirements.txt
```

## Uso

1. Ejecute la aplicación:
```bash
streamlit run transaction_network_analyzer.py
```

2. En su navegador:
   - Cargue el archivo CSV con los datos de transacciones
   - Use el selector de nodos para analizar nodos específicos
   - Explore la visualización interactiva de la red
   - Revise las métricas y estadísticas

## Métricas Disponibles

### Métricas de Nodo
- Grado de entrada (InDegree)
- Grado de salida (OutDegree)
- Centralidad de grado
- Centralidad de intermediación
- Centralidad de cercanía
- Centralidad de eigenvector
- PageRank
- Total enviado/recibido
- Cantidad de envíos realizados/recibidos

### Métricas de Red
- Número de nodos
- Número de conexiones
- Número de comunidades
- Densidad de la red
- Reciprocidad
- Diámetro de la red

## Interpretación de Métricas

- **Centralidad de grado**: Indica qué tan conectado está un nodo con otros nodos.
- **Centralidad de intermediación**: Mide qué tan importante es un nodo para conectar diferentes partes de la red.
- **Centralidad de cercanía**: Indica qué tan cerca está un nodo de todos los demás nodos de la red.
- **Centralidad de eigenvector**: Mide la influencia de un nodo en la red.
- **PageRank**: Similar a la centralidad de eigenvector, pero considera la dirección de las conexiones.
- **Comunidades**: Grupos de nodos que están más densamente conectados entre sí que con el resto de la red.

## Visualización

- Los nodos están coloreados por nivel de riesgo (escala de colores Viridis)
- El tamaño de los nodos está basado en su PageRank
- Las conexiones muestran el monto y número de transacciones al pasar el mouse
- Al pasar el mouse sobre un nodo se muestra información detallada incluyendo su ID, comunidad, métricas y nivel de riesgo
- El nodo seleccionado se resalta en rojo

## Seguridad

Para usar la versión segura:

1. Configura tu contraseña:
```python
import hashlib
password = "tu_contraseña"
hashed = hashlib.sha256(password.encode()).hexdigest()
print(hashed)
```

2. Coloca el hash en `.streamlit/secrets.toml`

3. Ejecuta la aplicación:
```bash
streamlit run transaction_network_analyzer.py
```

La aplicación ahora:
- Maneja redes grandes eficientemente
- Protege los datos con autenticación
- Registra todas las acciones en un log
- Limpia la memoria automáticamente
- Muestra visualizaciones optimizadas

## Sistema de Scoring de Riesgo

El sistema de scoring de riesgo evalúa cada componente de la red utilizando los siguientes indicadores:
- **Tamaño del Componente**: Mayor riesgo para componentes medianos (10-30 nodos).
- **Densidad del Componente**: Mayor riesgo para componentes densamente conectados.
- **Montos y Frecuencia de Transacciones**: Basado en el monto total y la cantidad de transacciones.
- **Patrones de Transacción**: Detecta ciclos y patrones de "ida y vuelta".
- **Reciprocidad de Montos**: Analiza la similitud en montos de transacciones recíprocas.

Cada indicador contribuye a un score final ponderado que determina el nivel de riesgo del componente.

## Filtros Mejorados

La aplicación permite aplicar filtros para ajustar la visualización de la red:
- **Tamaño Mínimo y Máximo de Componente**: Permite definir el rango de tamaño de los componentes a visualizar.
- **Score Mínimo de Riesgo**: Filtra componentes que superan un umbral de riesgo específico.

## Panel de Componentes de Alto Riesgo

Este panel muestra los componentes que tienen un score de riesgo alto. Para cada componente, se proporciona:
- **Métricas Básicas**: Número de nodos, conexiones, y monto total.
- **Indicadores de Riesgo**: Desglose de los scores de cada indicador.

Los usuarios pueden expandir cada componente para ver detalles adicionales.