import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline
import streamlit as st
import plotly.graph_objects as go

# Título de la aplicación
st.title("Visualización de Entidades Nombradas con NetworkX")

# Entrada de texto para la oración
sentence = st.text_input("Introduce una oración:", "Barack Obama was born in Hawaii. He was elected president in 2008.")

# Cargar el modelo preentrenado con st.cache_resource
@st.cache_resource
def load_model():
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

ner_pipeline = load_model()

# Extraer las entidades
entities = ner_pipeline(sentence)

# Filtrar entidades repetidas y mantener la primera aparición
unique_entities = {}
for entity in entities:
    if entity['word'] not in unique_entities:
        unique_entities[entity['word']] = entity['entity']

# Mostrar las entidades extraídas
st.write("Entidades extraídas:")
for word, entity_type in unique_entities.items():
    st.write(f"{word}: {entity_type}")

# Crear el grafo
G = nx.DiGraph()

# Añadir nodos
for word, entity_type in unique_entities.items():
    G.add_node(word, label=entity_type)

# Añadir aristas
entity_words = list(unique_entities.keys())
for i in range(len(entity_words) - 1):
    G.add_edge(entity_words[i], entity_words[i + 1])

# Dibujar el grafo usando Plotly
pos = nx.spring_layout(G)
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}: {G.nodes[node]['label']}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=20,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Network Graph of Named Entities',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Network graph visualization using Plotly",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
                )

# Mostrar la visualización en Streamlit
st.plotly_chart(fig, use_container_width=True)

# Mostrar nodos y aristas del grafo cargado
st.write("Nodos del grafo:", list(G.nodes()))
st.write("Aristas del grafo:", list(G.edges()))
