# Documentación Técnica: TransactionNetworkAnalyzer

Este documento proporciona una descripción técnica detallada de la clase `TransactionNetworkAnalyzer` y sus métodos, diseñada para el análisis de redes transaccionales con enfoque en la detección de riesgo.

## Índice
1. [Visión General](#visión-general)
2. [Inicialización y Carga de Datos](#inicialización-y-carga-de-datos)
3. [Sistema de Scoring de Riesgo](#sistema-de-scoring-de-riesgo)
4. [Métricas de Nodos](#métricas-de-nodos)
5. [Visualización](#visualización)
6. [Consideraciones Técnicas](#consideraciones-técnicas)

## Visión General

La clase `TransactionNetworkAnalyzer` implementa un conjunto de algoritmos y métodos para analizar redes transaccionales representadas como grafos dirigidos. Está diseñada específicamente para equipos de riesgos y compliance que necesitan identificar patrones sospechosos en transacciones financieras.

## Inicialización y Carga de Datos

### `__init__(self)`

**Función**: Constructor de la clase que inicializa las variables de instancia.

**Detalles**:
- Inicializa un grafo dirigido vacío (`self.G = nx.DiGraph()`)
- Inicializa variables para almacenar:
  - El DataFrame de datos (`self.df = None`)
  - Las comunidades detectadas (`self.communities = None`)
  - Los componentes conectados (`self.components = None`)
  - Los scores de riesgo (`self.risk_scores = {}`)

### `load_data(self, df: pd.DataFrame) -> None`

**Función**: Carga los datos del DataFrame y crea el grafo dirigido.

**Parámetros**:
- `df`: DataFrame de Pandas que debe contener las columnas 'origen', 'destino', 'monto_total' y 'cantidad_envios'.

**Funcionamiento**:
1. Almacena el DataFrame en `self.df`
2. Itera sobre cada fila del DataFrame para crear el grafo:
   - Cada fila representa una transacción entre un nodo origen y un nodo destino
   - Agrega un borde (edge) al grafo con atributos de peso (monto_total) y número de transacciones
3. Calcula las comunidades usando el algoritmo de Louvain (community_louvain.best_partition)
4. Identifica los componentes débilmente conectados del grafo
5. Calcula los scores de riesgo para cada componente

**Requerimientos**:
- El DataFrame debe tener las columnas 'origen', 'destino', 'monto_total' y 'cantidad_envios'
- 'origen' y 'destino' son identificadores de nodos
- 'monto_total' es el monto total transado entre origen y destino
- 'cantidad_envios' es el número de transacciones realizadas

**Complejidad computacional**:
- Tiempo: O(E) donde E es el número de transacciones (filas en el DataFrame)
- Espacio: O(N + E) donde N es el número de nodos y E el número de conexiones

## Sistema de Scoring de Riesgo

### `_calculate_risk_scores(self) -> None`

**Función**: Calcula scores de riesgo para cada componente basado en varios indicadores.

**Funcionamiento**:
1. Itera sobre cada componente conectado del grafo
2. Para cada componente:
   - Crea un subgrafo con los nodos del componente
   - Calcula métricas básicas (número de nodos, conexiones, monto total, transacciones)
   - Calcula 5 indicadores de riesgo:
     - Riesgo por tamaño (`size_score`)
     - Riesgo por densidad (`density_score`)
     - Riesgo por montos (`amount_score`)
     - Riesgo por patrones (`pattern_score`)
     - Riesgo por reciprocidad (`reciprocity_score`)
   - Calcula un score final como promedio ponderado de los indicadores
   - Almacena los resultados en `self.risk_scores`

**Ponderaciones**:
- Tamaño: 15%
- Densidad: 20%
- Montos: 30%
- Patrones: 20%
- Reciprocidad: 15%

**Complejidad computacional**:
- Tiempo: O(C * (N + E)) donde C es el número de componentes, N el número de nodos y E el número de conexiones
- Espacio: O(C) para almacenar los scores de riesgo

### `_calculate_size_risk(self, n_nodes: int) -> float`

**Función**: Calcula el riesgo basado en el tamaño del componente.

**Parámetros**:
- `n_nodes`: Número de nodos en el componente

**Funcionamiento**:
- Asigna mayor riesgo (1.0) a componentes medianos (10-30 nodos)
- Asigna riesgo medio (0.7) a componentes de 5-9 nodos o 31-50 nodos
- Asigna riesgo bajo (0.3) a componentes grandes (>50 nodos)
- Asigna riesgo de 0.4 a componentes pequeños (<5 nodos)

**Lógica**: Los componentes medianos son considerados de mayor riesgo porque son lo suficientemente grandes para ocultar actividades sospechosas pero no tan grandes como para ser redes legítimas comunes.

**Complejidad computacional**:
- Tiempo: O(1)
- Espacio: O(1)

### `_calculate_density_risk(self, subgraph: nx.DiGraph) -> float`

**Función**: Calcula el riesgo basado en la densidad del componente.

**Parámetros**:
- `subgraph`: Subgrafo del componente a analizar

**Funcionamiento**:
- Calcula la densidad del grafo (proporción de conexiones existentes respecto al máximo posible)
- Multiplica la densidad por 2 y limita el resultado a un máximo de 1.0

**Lógica**: Componentes más densos (donde hay muchas conexiones entre los nodos) tienen mayor riesgo, ya que pueden indicar patrones circulares o de lavado.

**Complejidad computacional**:
- Tiempo: O(1) para el cálculo de densidad (implementado eficientemente en NetworkX)
- Espacio: O(1)

### `_calculate_amount_risk(self, total_amount: float, total_transactions: int) -> float`

**Función**: Calcula el riesgo basado en montos y frecuencia de transacciones en relación a toda la red.

**Parámetros**:
- `total_amount`: Monto total transado en el componente
- `total_transactions`: Número total de transacciones en el componente

**Funcionamiento**:
1. Si no hay transacciones, retorna 0.0
2. Calcula los totales de la red completa (monto total y número de transacciones)
3. Calcula la proporción del componente respecto al total de la red
4. Aplica una función no lineal (raíz cuadrada) para dar más peso a proporciones altas
5. Normaliza y limita los valores a un máximo de 1.0
6. Retorna el promedio de los factores de riesgo por monto y por frecuencia

**Lógica**: Componentes que manejan una mayor proporción del flujo total de la red tienen mayor riesgo, ya que representan concentraciones significativas de actividad.

**Complejidad computacional**:
- Tiempo: O(E) donde E es el número total de conexiones en el grafo
- Espacio: O(1)

### `_calculate_pattern_risk(self, subgraph: nx.DiGraph) -> float`

**Función**: Calcula el riesgo basado en patrones de transacción.

**Parámetros**:
- `subgraph`: Subgrafo del componente a analizar

**Funcionamiento**:
1. Detecta ciclos en el grafo (transacciones que forman un ciclo cerrado)
   - Calcula `cycle_risk` como la proporción de ciclos encontrados (máximo en 10 ciclos)
2. Detecta patrones de "ida y vuelta" (transacciones recíprocas entre nodos)
   - Calcula `reciprocal_risk` como la proporción de conexiones recíprocas
3. Retorna el máximo entre ambos factores de riesgo

**Lógica**: Los ciclos y las transacciones recíprocas pueden indicar intentos de ocultar el origen de los fondos.

**Complejidad computacional**:
- Tiempo: O(N + E) para la detección de ciclos y O(E) para la detección de patrones recíprocos
- Espacio: O(N + E) para almacenar los ciclos detectados

### `_calculate_reciprocity_risk(self, subgraph: nx.DiGraph) -> float`

**Función**: Calcula el riesgo basado en la reciprocidad de montos.

**Parámetros**:
- `subgraph`: Subgrafo del componente a analizar

**Funcionamiento**:
1. Para cada par de conexiones recíprocas (A→B y B→A):
   - Compara los montos de ambas transacciones
   - Calcula la diferencia relativa entre los montos
   - Asigna mayor riesgo cuando los montos son similares (1 - diff_ratio)
2. Retorna el promedio de los valores calculados

**Lógica**: Transacciones recíprocas con montos similares pueden indicar intentos de "compensación" o lavado de dinero.

**Complejidad computacional**:
- Tiempo: O(E) donde E es el número de conexiones en el subgrafo
- Espacio: O(E) en el peor caso para almacenar las diferencias de montos

### `get_high_risk_components(self, threshold: float = 0.55) -> List[int]`

**Función**: Retorna los índices de componentes con alto riesgo.

**Parámetros**:
- `threshold`: Umbral de riesgo (por defecto 0.55)

**Funcionamiento**: Filtra los componentes cuyo score de riesgo es mayor o igual al umbral especificado.

**Requerimientos**: Debe haberse ejecutado previamente `_calculate_risk_scores()`.

**Complejidad computacional**:
- Tiempo: O(C) donde C es el número de componentes
- Espacio: O(C) para almacenar los índices de componentes de alto riesgo

## Métricas de Nodos

### `calculate_node_metrics(self, node_id: str) -> Dict`

**Función**: Calcula métricas relevantes para un nodo específico.

**Parámetros**:
- `node_id`: Identificador del nodo a analizar

**Funcionamiento**:
1. Verifica que el nodo exista en el grafo
2. Calcula métricas de centralidad:
   - Grado de entrada y salida
   - Centralidad de grado
   - Centralidad de intermediación
   - Centralidad de cercanía
   - Centralidad de eigenvector
   - Comunidad a la que pertenece
   - PageRank
3. Calcula métricas de transacciones:
   - Total enviado
   - Total recibido
   - Cantidad de envíos realizados
   - Cantidad de envíos recibidos

**Requerimientos**:
- Requiere la biblioteca `scipy` para calcular la centralidad de eigenvector
- El nodo debe existir en el grafo

**Complejidad computacional**:
- Tiempo: O(N + E) para el cálculo de las métricas de centralidad
- Espacio: O(N) para almacenar los resultados de las métricas

### Algoritmo PageRank

**Descripción**: PageRank es un algoritmo desarrollado originalmente por Google para clasificar páginas web en su motor de búsqueda. En el contexto de análisis de redes transaccionales, PageRank proporciona una medida de la importancia o centralidad de un nodo basada en la estructura de la red.

**Funcionamiento**:
1. El algoritmo asigna a cada nodo una puntuación inicial (generalmente 1/N donde N es el número total de nodos).
2. Luego, itera repetidamente actualizando la puntuación de cada nodo según la fórmula:
   ```
   PR(A) = (1-d) + d * (PR(T1)/C(T1) + PR(T2)/C(T2) + ... + PR(Tn)/C(Tn))
   ```
   Donde:
   - PR(A) es el PageRank del nodo A
   - PR(Ti) es el PageRank de los nodos que tienen conexiones hacia A
   - C(Ti) es el número de conexiones salientes del nodo Ti
   - d es un factor de amortiguación (generalmente 0.85)

3. El proceso se repite hasta que los valores convergen (las puntuaciones se estabilizan).

**Interpretación en redes transaccionales**:
- Un nodo con alto PageRank recibe transacciones de otros nodos que también tienen alto PageRank.
- En el contexto financiero, un alto PageRank puede indicar:
  - Nodos que actúan como concentradores de fondos
  - Entidades que reciben fondos de múltiples fuentes importantes
  - Posibles puntos de consolidación en esquemas de lavado de dinero

**Diferencias con otras métricas de centralidad**:
- A diferencia de la centralidad de grado, PageRank considera no solo el número de conexiones, sino también la importancia de esas conexiones.
- A diferencia de la centralidad de eigenvector, PageRank está diseñado específicamente para grafos dirigidos y maneja mejor los casos de nodos sin conexiones salientes.

**Aplicación en la herramienta**:
- Se utiliza para identificar los nodos más importantes en la red
- Determina el tamaño de los nodos en la visualización
- Se emplea para filtrar nodos cuando la red es demasiado grande para visualizarse completa

**Complejidad computacional**:
- Tiempo: O(I * (N + E)) donde I es el número de iteraciones hasta la convergencia
- Espacio: O(N) para almacenar los valores de PageRank

### `get_node_connections(self, node_id: str) -> Tuple[List, List]`

**Función**: Obtiene las conexiones entrantes y salientes de un nodo.

**Parámetros**:
- `node_id`: Identificador del nodo a analizar

**Funcionamiento**:
1. Verifica que el nodo exista en el grafo
2. Obtiene las conexiones salientes (nodos a los que envía)
3. Obtiene las conexiones entrantes (nodos de los que recibe)
4. Retorna ambas listas como una tupla (incoming, outgoing)

**Requerimientos**: El nodo debe existir en el grafo.

**Complejidad computacional**:
- Tiempo: O(d) donde d es el grado del nodo (número de conexiones)
- Espacio: O(d) para almacenar las listas de conexiones

## Visualización

### `visualize_network(self, selected_node: str = None, max_nodes: int = 800, min_risk_score: float = 0.0, max_component_size: int = None, min_component_size: int = None, selected_component: int = None, color_by: str = "risk_score") -> go.Figure`

**Función**: Genera una visualización interactiva de la red con filtros.

**Parámetros**:
- `selected_node`: Nodo seleccionado para resaltar (opcional)
- `max_nodes`: Número máximo de nodos a mostrar (por defecto 800)
- `min_risk_score`: Score mínimo de riesgo para filtrar componentes (por defecto 0.0)
- `max_component_size`: Tamaño máximo de componente a mostrar (opcional)
- `min_component_size`: Tamaño mínimo de componente a mostrar (opcional)
- `selected_component`: Índice del componente específico a visualizar (opcional)
- `color_by`: Métrica para colorear los nodos (por defecto "risk_score")
  - Opciones: "risk_score", "pagerank", "degree", "betweenness", "community"

**Funcionamiento**:
1. Determina los nodos a mostrar según los filtros:
   - Si hay un componente seleccionado, muestra solo ese componente
   - Si no, aplica los filtros de tamaño y riesgo a todos los componentes
   - Si hay un nodo seleccionado, muestra su componente
2. Si el grafo resultante es muy grande, muestra solo los nodos más importantes (según PageRank)
3. Calcula el layout usando el algoritmo spring_layout
4. Prepara los datos para la visualización:
   - Crea trazas para las conexiones (edges)
   - Colorea los nodos según la métrica seleccionada:
     - "risk_score": Color basado en el nivel de riesgo del componente
     - "pagerank": Color basado en el PageRank del nodo
     - "degree": Color basado en la centralidad de grado
     - "betweenness": Color basado en la centralidad de intermediación
     - "community": Color basado en la comunidad a la que pertenece el nodo
   - Ajusta el tamaño de los nodos basado en PageRank
   - Agrega texto informativo al pasar el mouse
5. Crea la figura con Plotly
6. Si hay un nodo seleccionado, lo resalta en rojo
7. Retorna la figura para su visualización

**Requerimientos**:
- Requiere la biblioteca `plotly` para la visualización
- Maneja excepciones y las registra en el log

**Complejidad computacional**:
- Tiempo: O(N^2) para el cálculo del layout (spring_layout)
- Espacio: O(N + E) para almacenar los datos de visualización

**Uso en la interfaz**:
- Por defecto, muestra todos los componentes coloreados por nivel de riesgo
- Al seleccionar un componente específico, cambia automáticamente a colorear por PageRank
- El usuario puede cambiar la métrica de color mediante un selector en la interfaz
- Un botón "Ver Red Completa" permite volver a la visualización global

## Consideraciones Técnicas

### Dependencias

La clase `TransactionNetworkAnalyzer` depende de las siguientes bibliotecas:
- `networkx`: Para la creación y manipulación de grafos
- `pandas`: Para el manejo de datos tabulares
- `plotly`: Para la visualización interactiva
- `community`: Para la detección de comunidades (algoritmo de Louvain)
- `numpy`: Para operaciones numéricas
- `scipy`: Para cálculos de centralidad avanzados

### Escalabilidad

- La clase está diseñada para manejar redes de hasta 800 nodos en la visualización
- Para redes más grandes, se implementa un filtrado basado en PageRank para mostrar los nodos más importantes
- Los cálculos de métricas de centralidad pueden ser computacionalmente intensivos para redes muy grandes

### Optimizaciones

- Se utilizan algoritmos eficientes de NetworkX para los cálculos de métricas
- La visualización se optimiza para mostrar solo los componentes relevantes según los filtros
- El sistema de scoring de riesgo utiliza cálculos relativamente simples para mantener la eficiencia

### Limitaciones

- La detección de ciclos puede ser computacionalmente intensiva para redes muy densas
- El cálculo de centralidad de eigenvector requiere que el grafo sea fuertemente conectado
- La visualización puede volverse lenta para redes con muchas conexiones

### Extensibilidad

La clase está diseñada para ser extensible:
- Se pueden agregar nuevos indicadores de riesgo implementando métodos adicionales
- Las ponderaciones de los indicadores pueden ajustarse según las necesidades específicas
- Se pueden implementar algoritmos de layout alternativos para la visualización 