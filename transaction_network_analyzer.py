import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from community import community_louvain
import numpy as np
from typing import Tuple, Dict, List
import logging

class TransactionNetworkAnalyzer:
    def __init__(self):
        self.G = nx.DiGraph()
        self.df = None
        self.communities = None
        self.components = None
        self.risk_scores = {}
        
    def load_data(self, df: pd.DataFrame) -> None:
        """
        Carga los datos del DataFrame y crea el grafo dirigido
        """
        self.df = df
        # Crear el grafo con los pesos (montos totales)
        for _, row in df.iterrows():
            self.G.add_edge(
                row['origen'],
                row['destino'],
                weight=row['monto_total'],
                transactions=row['cantidad_envios']
            )
        
        # Calcular comunidades
        self.communities = community_louvain.best_partition(self.G.to_undirected())
        
        # Identificar componentes conectados
        self.components = list(nx.weakly_connected_components(self.G))
        
        # Calcular scores de riesgo para cada componente
        self._calculate_risk_scores()
    
    def get_network_statistics(self) -> Dict:
        """
        Calcula y retorna estadísticas básicas de la red de transacciones.   
        Retorna un Diccionario con las siguientes estadísticas:
        - Número total de nodos
        - Número total de aristas
        - Densidad de la red
        - Número de componentes conectados
        - Tamaño promedio de componentes
        - Monto total de transacciones
        - Número total de transacciones
        - Monto promedio por transacción
        - Grado promedio de entrada
        - Grado promedio de salida
        - Centralidad promedio
        """
        stats = {}

        # Estadísticas básicas de la red
        stats['total_nodos'] = len(self.G.nodes())
        stats['total_aristas'] = len(self.G.edges())
        stats['densidad'] = nx.density(self.G)
        stats['num_componentes'] = len(self.components)

        # Tamaño promedio de componentes
        component_sizes = [len(c) for c in self.components]
        stats['tamaño_promedio_componentes'] = np.mean(component_sizes) if component_sizes else 0

        # Estadísticas de transacciones
        total_amount = sum(d['weight'] for _, _, d in self.G.edges(data=True))
        total_transactions = sum(d['transactions'] for _, _, d in self.G.edges(data=True))
    
        stats['monto_total'] = total_amount
        stats['total_transacciones'] = total_transactions
        stats['monto_promedio_por_transaccion'] = total_amount / total_transactions if total_transactions > 0 else 0

        # Estadísticas de grados
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
    
        stats['grado_promedio_entrada'] = np.mean(in_degrees) if in_degrees else 0
        stats['grado_promedio_salida'] = np.mean(out_degrees) if out_degrees else 0

        # Centralidad promedio
        degree_centrality = nx.degree_centrality(self.G)
        stats['centralidad_promedio'] = np.mean(list(degree_centrality.values())) if degree_centrality else 0

        return stats
        
    def _calculate_risk_scores(self) -> None:
        """
        Calcula scores de riesgo para cada componente basado en varios indicadores
        """
        for i, component in enumerate(self.components):
            subgraph = self.G.subgraph(component)
            
            # Métricas básicas
            n_nodes = len(subgraph)
            n_edges = len(subgraph.edges())
            total_amount = sum(d['weight'] for _, _, d in subgraph.edges(data=True))
            total_transactions = sum(d['transactions'] for _, _, d in subgraph.edges(data=True))
            
            # Indicadores de riesgo
            risk_indicators = {
                'size_score': self._calculate_size_risk(n_nodes),
                'density_score': self._calculate_density_risk(subgraph),
                'amount_score': self._calculate_amount_risk(total_amount, total_transactions),
                'pattern_score': self._calculate_pattern_risk(subgraph),
                'reciprocity_score': self._calculate_reciprocity_risk(subgraph)
            }
            
            # Score final (promedio ponderado)
            weights = {'size_score': 0.15, 'density_score': 0.2, 
                      'amount_score': 0.3, 'pattern_score': 0.2, 
                      'reciprocity_score': 0.15}
            
            final_score = sum(score * weights[key] for key, score in risk_indicators.items())
            
            self.risk_scores[i] = {
                'score': final_score,
                'indicators': risk_indicators,
                'metrics': {
                    'nodes': n_nodes,
                    'edges': n_edges,
                    'total_amount': total_amount,
                    'total_transactions': total_transactions,
                    'avg_transaction': total_amount / total_transactions if total_transactions > 0 else 0
                }
            }

    def _calculate_size_risk(self, n_nodes: int) -> float:
        """
        Calcula riesgo basado en tamaño del componente
        Mayor riesgo para componentes medianos (10-30 nodos)
        """
        if 10 <= n_nodes <= 30:
            return 1.0
        elif 5 <= n_nodes < 10 or 30 < n_nodes <= 50:
            return 0.7
        elif n_nodes > 50:
            return 0.3
        return 0.4

    def _calculate_density_risk(self, subgraph: nx.DiGraph) -> float:
        """
        Calcula riesgo basado en densidad del componente
        Mayor riesgo para componentes muy densos
        """
        density = nx.density(subgraph)
        return min(1.0, density * 2)  # Normalizado a [0,1]

    def _calculate_amount_risk(self, total_amount: float, total_transactions: int) -> float:
        """
        Calcula riesgo basado en montos y frecuencia de transacciones en relación a toda la red
        
        El riesgo se calcula como la proporción del monto total y número de transacciones
        del componente respecto al total de la red. Componentes que manejan una mayor
        proporción del flujo total de la red tienen mayor riesgo.
        """
        if total_transactions == 0:
            return 0.0
            
        # Calcular totales de la red completa
        network_total_amount = sum(d['weight'] for _, _, d in self.G.edges(data=True))
        network_total_transactions = sum(d['transactions'] for _, _, d in self.G.edges(data=True))
        
        # Calcular proporciones
        amount_proportion = total_amount / network_total_amount if network_total_amount > 0 else 0
        transaction_proportion = total_transactions / network_total_transactions if network_total_transactions > 0 else 0
        
        # Aplicar una función no lineal para dar más peso a proporciones altas
        # Usando una función cuadrática: x^2 para enfatizar valores altos
        amount_risk = min(1.0, amount_proportion ** 0.5 * 2)  # Raíz cuadrada para suavizar, multiplicado por 2 para normalizar
        freq_risk = min(1.0, transaction_proportion ** 0.5 * 2)
        
        return (amount_risk + freq_risk) / 2

    def _calculate_pattern_risk(self, subgraph: nx.DiGraph) -> float:
        """
        Calcula riesgo basado en patrones de transacción
        """
        # Detectar ciclos
        try:
            cycles = list(nx.simple_cycles(subgraph))
            cycle_risk = min(1.0, len(cycles) / 10)  # Normalizado a [0,1]
        except:
            cycle_risk = 0.0
        
        # Detectar patrones de "ida y vuelta"
        reciprocal_edges = 0
        total_edges = len(subgraph.edges())
        
        if total_edges > 0:
            for edge in subgraph.edges():
                if subgraph.has_edge(edge[1], edge[0]):
                    reciprocal_edges += 1
            reciprocal_risk = reciprocal_edges / total_edges
        else:
            reciprocal_risk = 0.0
        
        return max(cycle_risk, reciprocal_risk)

    def _calculate_reciprocity_risk(self, subgraph: nx.DiGraph) -> float:
        """
        Calcula riesgo basado en reciprocidad de montos
        """
        reciprocal_amount_diff = []
        
        for edge in subgraph.edges():
            if subgraph.has_edge(edge[1], edge[0]):
                amount1 = subgraph[edge[0]][edge[1]]['weight']
                amount2 = subgraph[edge[1]][edge[0]]['weight']
                diff_ratio = abs(amount1 - amount2) / max(amount1, amount2)
                reciprocal_amount_diff.append(1 - diff_ratio)  # Mayor riesgo cuando montos son similares
        
        return np.mean(reciprocal_amount_diff) if reciprocal_amount_diff else 0.0

    def get_high_risk_components(self, threshold: float = 0.55) -> List[int]:
        """
        Retorna índices de componentes con alto riesgo
        
        Args:
            threshold: Umbral de riesgo (por defecto 0.55)
            
        Returns:
            Lista de índices de componentes que superan el umbral de riesgo
        """
        return [idx for idx, data in self.risk_scores.items() 
                if data['score'] >= threshold]

    def calculate_node_metrics(self, node_id: str) -> Dict:
        """
        Calcula métricas relevantes para un nodo específico
        """
        if node_id not in self.G.nodes():
            return None
            
        metrics = {
            "Grado de entrada (InDegree)": self.G.in_degree(node_id),
            "Grado de salida (OutDegree)": self.G.out_degree(node_id),
            "Centralidad de grado": nx.degree_centrality(self.G)[node_id],
            "Centralidad de intermediación": nx.betweenness_centrality(self.G)[node_id],
            "Centralidad de cercanía": nx.closeness_centrality(self.G)[node_id],
            "Centralidad de eigenvector": nx.eigenvector_centrality_numpy(self.G)[node_id],
            "Comunidad": self.communities[node_id],
            "PageRank": nx.pagerank(self.G)[node_id]
        }
        
        # Calcular métricas de transacciones
        out_transactions = [(n, self.G[node_id][n]) for n in self.G[node_id]]
        in_transactions = [(n, self.G[n][node_id]) for n in self.G.predecessors(node_id)]
        
        metrics.update({
            "Total enviado": sum(d['weight'] for _, d in out_transactions),
            "Total recibido": sum(d['weight'] for _, d in in_transactions),
            "Cantidad de envíos realizados": sum(d['transactions'] for _, d in out_transactions),
            "Cantidad de envíos recibidos": sum(d['transactions'] for _, d in in_transactions)
        })
        
        return metrics
    
    def get_node_connections(self, node_id: str) -> Tuple[List, List]:
        """
        Obtiene las conexiones entrantes y salientes de un nodo
        """
        if node_id not in self.G.nodes():
            return [], []
            
        outgoing = [(n, self.G[node_id][n]) for n in self.G[node_id]]
        incoming = [(n, self.G[n][node_id]) for n in self.G.predecessors(node_id)]
        
        return incoming, outgoing
    
    def visualize_network(self, selected_node: str = None, max_nodes: int = 800,
                         min_risk_score: float = 0.0, max_component_size: int = None,
                         min_component_size: int = None, selected_component: int = None,
                         color_by: str = "risk_score") -> go.Figure:
        """
        Visualización mejorada con filtros de riesgo y tamaño
        
        Args:
            selected_node: Nodo seleccionado para resaltar (opcional)
            max_nodes: Número máximo de nodos a mostrar (por defecto 800)
            min_risk_score: Score mínimo de riesgo para filtrar componentes
            max_component_size: Tamaño máximo de componente a mostrar
            min_component_size: Tamaño mínimo de componente a mostrar
            selected_component: Componente específico a visualizar (opcional)
            color_by: Métrica para colorear los nodos ("risk_score", "pagerank", "degree", "betweenness", "community")
            
        Returns:
            Figura de Plotly con la visualización de la red
        """
        try:
            # Información de depuración
            print(f"Visualizando red con parámetros:")
            print(f"- selected_node: {selected_node}")
            print(f"- min_risk_score: {min_risk_score}")
            print(f"- max_component_size: {max_component_size}")
            print(f"- min_component_size: {min_component_size}")
            print(f"- selected_component: {selected_component}")
            print(f"- color_by: {color_by}")
            print(f"- Total de componentes: {len(self.components)}")
            
            # Filtrar componentes según criterios
            all_nodes = []
            components_to_show = []
            
            # Recolectar todos los nodos para referencia
            for component in self.components:
                all_nodes.extend(list(component))
            
            print(f"Total de nodos en la red: {len(all_nodes)}")
            
            # Si hay un componente seleccionado, mostrar solo ese componente
            if selected_component is not None and selected_component < len(self.components):
                print(f"Mostrando solo el componente {selected_component}")
                components_to_show = list(self.components[selected_component])
            # Si no hay componente seleccionado, mostrar todos los componentes que cumplan con los filtros
            else:
                print("Mostrando todos los componentes que cumplen con los filtros")
                # Aplicar filtros normales
                for i, component in enumerate(self.components):
                    # Verificar tamaño mínimo si está especificado
                    if min_component_size is not None and len(component) < min_component_size:
                        print(f"Componente {i} filtrado por tamaño mínimo: {len(component)} < {min_component_size}")
                        continue
                    # Verificar tamaño máximo si está especificado
                    if max_component_size is not None and len(component) > max_component_size:
                        print(f"Componente {i} filtrado por tamaño máximo: {len(component)} > {max_component_size}")
                        continue
                    # Verificar score de riesgo si está especificado
                    if min_risk_score > 0 and self.risk_scores[i]['score'] < min_risk_score:
                        print(f"Componente {i} filtrado por score de riesgo: {self.risk_scores[i]['score']} < {min_risk_score}")
                        continue
                    # Agregar todos los nodos del componente
                    print(f"Agregando componente {i} con {len(component)} nodos")
                    components_to_show.extend(list(component))

            # Si no hay componentes para mostrar después de aplicar los filtros, mostrar todos
            if not components_to_show:
                print("No hay componentes que cumplan con los filtros. Mostrando todos los componentes.")
                for component in self.components:
                    components_to_show.extend(list(component))
            
            # Asegurarse de que el nodo seleccionado esté incluido en la visualización
            if selected_node and selected_node not in components_to_show:
                print(f"Añadiendo el nodo seleccionado {selected_node} a la visualización")
                components_to_show.append(selected_node)
                
                # Opcionalmente, añadir también sus conexiones directas para contexto
                for neighbor in self.G.predecessors(selected_node):
                    if neighbor not in components_to_show:
                        components_to_show.append(neighbor)
                for neighbor in self.G.successors(selected_node):
                    if neighbor not in components_to_show:
                        components_to_show.append(neighbor)
            
            print(f"Total de nodos a mostrar: {len(components_to_show)}")
            
            # Forzar la visualización de todos los componentes si no hay uno seleccionado
            if selected_component is None:
                print("Forzando la visualización de todos los componentes que cumplen con los filtros")
                # No necesitamos hacer nada aquí, ya que components_to_show ya contiene los nodos filtrados

            # Crear subgrafo para visualización
            graph_to_plot = self.G.subgraph(components_to_show)

            # Si aún es muy grande, mostrar los nodos más importantes
            if len(graph_to_plot) > max_nodes:
                pagerank = nx.pagerank(graph_to_plot)
                important_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                nodes_to_show = [node for node, _ in important_nodes]
                graph_to_plot = graph_to_plot.subgraph(nodes_to_show)

            # Calcular layout
            pos = nx.spring_layout(graph_to_plot, k=1/np.sqrt(len(graph_to_plot.nodes())), iterations=50)

            # Preparar datos para la visualización
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in graph_to_plot.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(f"Monto: {edge[2]['weight']:.2f}<br>Envíos: {edge[2]['transactions']}")
            
            # Crear líneas para las conexiones
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                text=edge_text,
                mode='lines')
            
            # Preparar nodos
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            # Calcular métricas para colorear según la opción seleccionada
            if color_by == "pagerank":
                pagerank_values = nx.pagerank(graph_to_plot)
                color_metric_name = "PageRank"
            elif color_by == "degree":
                degree_centrality = nx.degree_centrality(graph_to_plot)
                color_metric_name = "Centralidad de grado"
            elif color_by == "betweenness":
                betweenness_centrality = nx.betweenness_centrality(graph_to_plot)
                color_metric_name = "Centralidad de intermediación"
            elif color_by == "community":
                color_metric_name = "Comunidad"
            else:  # default: risk_score
                color_metric_name = "Nivel de Riesgo"
            
            for node in graph_to_plot.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Encontrar el componente al que pertenece el nodo y su score de riesgo
                node_risk_score = 0.0
                for i, component in enumerate(self.components):
                    if node in component:
                        node_risk_score = self.risk_scores[i]['score']
                        break
                
                # Determinar el color según la métrica seleccionada
                if color_by == "pagerank":
                    node_color_value = pagerank_values[node]
                elif color_by == "degree":
                    node_color_value = degree_centrality[node]
                elif color_by == "betweenness":
                    node_color_value = betweenness_centrality[node]
                elif color_by == "community":
                    node_color_value = self.communities[node]
                else:  # default: risk_score
                    node_color_value = node_risk_score
                
                node_color.append(node_color_value)
                
                # Texto para el hover
                metrics = self.calculate_node_metrics(node)
                node_text.append(
                    f"ID: {node}<br>"
                    f"Comunidad: {metrics['Comunidad']}<br>"
                    f"InDegree: {metrics['Grado de entrada (InDegree)']}<br>"
                    f"OutDegree: {metrics['Grado de salida (OutDegree)']}<br>"
                    f"PageRank: {metrics['PageRank']:.4f}<br>"
                    f"Nivel de Riesgo: {node_risk_score:.2f}"
                )
                
                # Tamaño basado en PageRank
                node_size.append(metrics['PageRank'] * 1000)
            
            # Crear scatter plot para los nodos
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    color=node_color,
                    size=node_size,
                    sizemode='area',
                    sizeref=2.*max(node_size)/(40.**2),
                    colorbar=dict(title=color_metric_name),
                    line_width=2))
            
            # Crear la figura
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[dict(
                                   text="Red Transaccional",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002)],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                           )
            
            # Resaltar nodo seleccionado si existe
            if selected_node and selected_node in graph_to_plot.nodes():
                selected_pos = pos[selected_node]
                fig.add_trace(go.Scatter(
                    x=[selected_pos[0]],
                    y=[selected_pos[1]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='circle-dot'
                    ),
                    showlegend=False
                ))

            return fig

        except Exception as e:
            logging.error(f"Error en visualización: {str(e)}")
            raise

def main():
    st.title("Analizador de Redes Transaccionales")
    
    # Subida de archivo
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Cargar datos
        df = pd.read_csv(uploaded_file)
        
        # Verificar columnas requeridas
        required_columns = ['origen', 'destino', 'monto_total', 'cantidad_envios']
        if not all(col in df.columns for col in required_columns):
            st.error("El archivo CSV debe contener las columnas: origen, destino, monto_total, cantidad_envios")
            return
        
        # Inicializar el analizador
        analyzer = TransactionNetworkAnalyzer()
        analyzer.load_data(df)
        
        # Filtros en sidebar
        st.sidebar.subheader("Filtros de Visualización")
        
        # Filtros de componentes
        min_component_size = st.sidebar.number_input(
            "Tamaño mínimo de componente", 
            min_value=2, 
            value=2
        )
        
        max_component_size = st.sidebar.number_input(
            "Tamaño máximo de componente",
            min_value=min_component_size,
            value=1000
        )
        
        # Filtro de riesgo
        min_risk_score = st.sidebar.slider(
            "Score mínimo de riesgo",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )
        
        # Inicializar session_state para las pestañas
        if 'open_tabs' not in st.session_state:
            st.session_state.open_tabs = ["Vista Global"]
        
        # Crear un diccionario para mapear componentes a sus nodos
        component_nodes = {}
        for idx, component in enumerate(analyzer.components):
            component_nodes[idx] = list(component)
        
        # Obtener componentes de alto riesgo
        high_risk = analyzer.get_high_risk_components(threshold=0.55)
        
        # Crear pestañas
        tabs = st.tabs(st.session_state.open_tabs)
        
        # Contenido de la pestaña principal (Vista Global)
        with tabs[0]:
            # Mostrar componentes de alto riesgo
            st.subheader("Componentes de Alto Riesgo")
            
            if high_risk:
                # Sort high-risk components by risk score in descending order
                high_risk_components = sorted(high_risk, key=lambda x: analyzer.risk_scores[x]['score'], reverse=True)
                
                for idx, component in enumerate(high_risk_components):
                    risk_data = analyzer.risk_scores[component]
                    metrics = risk_data['metrics']
                    
                    with st.expander(f"Componente {component} - Score: {risk_data['score']:.2f}"):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.write("Métricas:")
                            st.write(f"- Nodos: {metrics['nodes']}")
                            st.write(f"- Conexiones: {metrics['edges']}")
                            st.write(f"- Monto total: ${metrics['total_amount']:,.2f}")
                            
                        with col2:
                            st.write("Indicadores de Riesgo:")
                            for name, score in risk_data['indicators'].items():
                                st.write(f"- {name}: {score:.2f}")
                        
                        with col3:
                            # Botón para abrir una nueva pestaña con este componente
                            tab_name = f"Componente {component}"
                            if tab_name not in st.session_state.open_tabs:
                                if st.button(f"Analizar en nueva pestaña", key=f"btn_tab_{component}"):
                                    st.session_state.open_tabs.append(tab_name)
                                    st.experimental_rerun()
            else:
                st.write("No se encontraron componentes de alto riesgo")
                
            # Sección de búsqueda de nodos
            st.subheader("Búsqueda de Nodos")
            
            # Selector de componente (opcional)
            component_options = ["Todos"] + [f"Componente {i}" for i in range(len(analyzer.components))]
            selected_component_str = st.selectbox(
                "Filtrar nodos por componente:",
                options=component_options,
                index=0,
                key="component_filter"
            )
            
            # Determinar los nodos a mostrar en el selector
            if selected_component_str == "Todos":
                filtered_nodes = list(analyzer.G.nodes())
                selected_component = None
            else:
                # Extraer el índice del componente del string
                component_idx = int(selected_component_str.split(" ")[1])
                filtered_nodes = component_nodes[component_idx]
                selected_component = component_idx
            
            # Selector de nodos
            node_id = st.selectbox(
                "Seleccionar nodo para análisis:", 
                filtered_nodes,
                format_func=lambda x: f"Nodo {x}",
                key="node_selector_global"
            )
            
            # Variable para controlar si se debe mostrar solo el componente del nodo
            show_only_node_component = False
            
            if node_id:
                # Mostrar métricas del nodo
                st.subheader(f"Métricas del Nodo {node_id}")
                metrics = analyzer.calculate_node_metrics(node_id)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Métricas de Centralidad:")
                    st.write(f"- Centralidad de grado: {metrics['Centralidad de grado']:.4f}")
                    st.write(f"- Centralidad de intermediación: {metrics['Centralidad de intermediación']:.4f}")
                    st.write(f"- Centralidad de cercanía: {metrics['Centralidad de cercanía']:.4f}")
                    st.write(f"- Centralidad de eigenvector: {metrics['Centralidad de eigenvector']:.4f}")
                    st.write(f"- PageRank: {metrics['PageRank']:.4f}")
                
                with col2:
                    st.write("Métricas de Transacciones:")
                    st.write(f"- Grado de entrada: {metrics['Grado de entrada (InDegree)']}")
                    st.write(f"- Grado de salida: {metrics['Grado de salida (OutDegree)']}")
                    st.write(f"- Total enviado: ${metrics['Total enviado']:,.2f}")
                    st.write(f"- Total recibido: ${metrics['Total recibido']:,.2f}")
                    st.write(f"- Cantidad de envíos realizados: {metrics['Cantidad de envíos realizados']}")
                    st.write(f"- Cantidad de envíos recibidos: {metrics['Cantidad de envíos recibidos']}")
                
                # Mostrar conexiones del nodo
                st.subheader("Conexiones del Nodo")
                incoming, outgoing = analyzer.get_node_connections(node_id)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Conexiones Entrantes:")
                    for node, data in incoming:
                        st.write(f"De {node}:")
                        st.write(f"- Monto: ${data['weight']:,.2f}")
                        st.write(f"- Envíos: {data['transactions']}")
                
                with col2:
                    st.write("Conexiones Salientes:")
                    for node, data in outgoing:
                        st.write(f"A {node}:")
                        st.write(f"- Monto: ${data['weight']:,.2f}")
                        st.write(f"- Envíos: {data['transactions']}")
                
                # Opciones de visualización para el nodo seleccionado
                st.subheader("Opciones de Visualización para el Nodo")
                
                # Opción para mostrar solo el componente del nodo seleccionado
                show_only_node_component = st.checkbox(
                    "Mostrar solo el componente de este nodo", 
                    value=False, 
                    help="Si se activa, la visualización mostrará solo el componente al que pertenece este nodo. Si no, el nodo se resaltará en la visualización global.",
                    key="show_node_component"
                )
                
                # Encontrar el componente al que pertenece el nodo
                node_component_idx = None
                for i, component in enumerate(analyzer.components):
                    if node_id in component:
                        node_component_idx = i
                        break
                
                if node_component_idx is not None:
                    # Mostrar información sobre el componente al que pertenece el nodo
                    risk_data = analyzer.risk_scores[node_component_idx]
                    st.write(f"Este nodo pertenece al Componente {node_component_idx} (Score de riesgo: {risk_data['score']:.2f})")
                    
                    # Botón para abrir el componente en una nueva pestaña
                    tab_name = f"Componente {node_component_idx}"
                    if tab_name not in st.session_state.open_tabs:
                        if st.button(f"Analizar Componente {node_component_idx} en nueva pestaña"):
                            st.session_state.open_tabs.append(tab_name)
                            st.experimental_rerun()
            
            # Visualización de la red completa
            st.subheader("Visualización de la Red")
            
            # Selector de métrica para colorear los nodos
            color_options = {
                "risk_score": "Nivel de Riesgo",
                "pagerank": "PageRank",
                "degree": "Centralidad de grado",
                "betweenness": "Centralidad de intermediación",
                "community": "Comunidad"
            }
            
            selected_color = st.selectbox(
                "Colorear nodos por:",
                options=list(color_options.keys()),
                format_func=lambda x: color_options[x],
                index=0,  # Por defecto, colorear por nivel de riesgo
                key="color_selector_global"
            )
            
            # Generar la visualización de la red completa
            try:
                # Si se ha seleccionado mostrar solo el componente del nodo, pasar el nodo a la visualización
                # De lo contrario, pasar None para que no limite la visualización a un componente específico
                highlight_node = node_id if node_id else None
                component_for_node = None
                
                # Si se ha seleccionado mostrar solo el componente del nodo
                if show_only_node_component and node_id:
                    # Encontrar el componente al que pertenece el nodo
                    for i, component in enumerate(analyzer.components):
                        if node_id in component:
                            component_for_node = i
                            break
                
                fig = analyzer.visualize_network(
                    selected_node=highlight_node,
                    max_nodes=800,
                    min_risk_score=min_risk_score,
                    max_component_size=max_component_size,
                    min_component_size=min_component_size,
                    selected_component=component_for_node if show_only_node_component else selected_component,
                    color_by=selected_color
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error al generar la visualización: {str(e)}")
            
            # Estadísticas generales de la red
            st.subheader("Estadísticas Generales de la Red")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Métricas Básicas:")
                st.write(f"- Número de nodos: {len(analyzer.G.nodes())}")
                st.write(f"- Número de conexiones: {len(analyzer.G.edges())}")
                st.write(f"- Número de comunidades: {len(set(analyzer.communities.values()))}")
                
            with col2:
                st.write("Métricas de Red:")
                st.write(f"- Densidad de la red: {nx.density(analyzer.G):.4f}")
                st.write(f"- Reciprocidad: {nx.reciprocity(analyzer.G):.4f}")
                try:
                    st.write(f"- Diámetro de la red: {nx.diameter(analyzer.G.to_undirected())}")
                except:
                    st.write("- Diámetro de la red: Red no conectada")
        
        # Contenido de las pestañas de componentes
        for i, tab_name in enumerate(st.session_state.open_tabs[1:], 1):
            with tabs[i]:
                # Extraer el índice del componente del nombre de la pestaña
                component_idx = int(tab_name.split(" ")[1])
                
                # Botón para cerrar la pestaña
                if st.button("Cerrar pestaña", key=f"close_tab_{component_idx}"):
                    st.session_state.open_tabs.remove(tab_name)
                    st.experimental_rerun()
                
                # Mostrar información del componente
                risk_data = analyzer.risk_scores[component_idx]
                metrics = risk_data['metrics']
                
                st.subheader(f"Componente {component_idx} - Score de Riesgo: {risk_data['score']:.2f}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Métricas:")
                    st.write(f"- Nodos: {metrics['nodes']}")
                    st.write(f"- Conexiones: {metrics['edges']}")
                    st.write(f"- Monto total: ${metrics['total_amount']:,.2f}")
                    
                with col2:
                    st.write("Indicadores de Riesgo:")
                    for name, score in risk_data['indicators'].items():
                        st.write(f"- {name}: {score:.2f}")
                
                # Selector de nodos del componente
                st.subheader("Nodos del Componente")
                component_node_id = st.selectbox(
                    "Seleccionar nodo:", 
                    component_nodes[component_idx],
                    format_func=lambda x: f"Nodo {x}",
                    key=f"node_selector_comp_{component_idx}"
                )
                
                if component_node_id:
                    # Mostrar métricas del nodo
                    st.subheader(f"Métricas del Nodo {component_node_id}")
                    metrics = analyzer.calculate_node_metrics(component_node_id)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Métricas de Centralidad:")
                        st.write(f"- Centralidad de grado: {metrics['Centralidad de grado']:.4f}")
                        st.write(f"- Centralidad de intermediación: {metrics['Centralidad de intermediación']:.4f}")
                        st.write(f"- Centralidad de cercanía: {metrics['Centralidad de cercanía']:.4f}")
                        st.write(f"- Centralidad de eigenvector: {metrics['Centralidad de eigenvector']:.4f}")
                        st.write(f"- PageRank: {metrics['PageRank']:.4f}")
                    
                    with col2:
                        st.write("Métricas de Transacciones:")
                        st.write(f"- Grado de entrada: {metrics['Grado de entrada (InDegree)']}")
                        st.write(f"- Grado de salida: {metrics['Grado de salida (OutDegree)']}")
                        st.write(f"- Total enviado: ${metrics['Total enviado']:,.2f}")
                        st.write(f"- Total recibido: ${metrics['Total recibido']:,.2f}")
                        st.write(f"- Cantidad de envíos realizados: {metrics['Cantidad de envíos realizados']}")
                        st.write(f"- Cantidad de envíos recibidos: {metrics['Cantidad de envíos recibidos']}")
                
                # Visualización del componente
                st.subheader("Visualización del Componente")
                
                # Selector de métrica para colorear los nodos (por defecto PageRank)
                color_options = {
                    "pagerank": "PageRank",
                    "risk_score": "Nivel de Riesgo",
                    "degree": "Centralidad de grado",
                    "betweenness": "Centralidad de intermediación",
                    "community": "Comunidad"
                }
                
                selected_color = st.selectbox(
                    "Colorear nodos por:",
                    options=list(color_options.keys()),
                    format_func=lambda x: color_options[x],
                    index=0,  # Por defecto, colorear por PageRank
                    key=f"color_selector_comp_{component_idx}"
                )
                
                # Generar la visualización del componente
                try:
                    fig = analyzer.visualize_network(
                        selected_node=component_node_id,
                        max_nodes=800,
                        selected_component=component_idx,
                        color_by=selected_color
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al generar la visualización: {str(e)}")

if __name__ == "__main__":
    main() 