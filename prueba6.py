import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline
import streamlit as st
import plotly.graph_objects as go

# Título de la aplicación
st.title("Visualización de Entidades Nombradas con NetworkX")

# Entrada de texto para la oración
sentence = st.text_input("Introduce una oración:", "Barack Obama was born in Hawaii. He was elected president in 2008.")

# Cargar los modelos preentrenados con st.cache_resource
@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

@st.cache_resource
def load_classifier_model():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

ner_pipeline = load_ner_model()
classifier_pipeline = load_classifier_model()

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

# Clasificación del texto (positivo o negativo como proxy de veracidad)
classification = classifier_pipeline(sentence)
label = classification[0]['label']
score = classification[0]['score']
veracity = "REAL" if label == "POSITIVE" else "FAKE"
st.write(f"Clasificación del texto: {veracity} (Confianza: {score:.2f})")

# Gráfica del porcentaje de veracidad
fig_veracity = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score * 100 if label == "POSITIVE" else (1 - score) * 100,
    title={'text': "Porcentaje de Veracidad"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "green" if label == "POSITIVE" else "red"}}
))

st.plotly_chart(fig_veracity, use_container_width=True)

# Texto correcto sugerido (placeholder)
st.write("Texto correcto sugerido:")
correct_text = st.text_area("Introduce el texto correcto:", "Barack Obama was born in Hawaii. He was elected president in 2008.")

# Configuración de nodos y aristas
node_color = st.color_picker("Selecciona el color de los nodos", '#1f77b4')
node_size = st.slider("Selecciona el tamaño de los nodos", 10, 100, 20)
edge_color = st.color_picker("Selecciona el color de las aristas", '#888')
edge_width = st.slider("Selecciona el grosor de las aristas", 1, 10, 2)

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
    line=dict(width=edge_width, color=edge_color),
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
        size=node_size,
        color=node_color,
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

# Agregar opciones para guardar y cargar el grafo
if st.button("Guardar el grafo"):
    nx.write_gexf(G, "graph.gexf")
    st.success("Grafo guardado como graph.gexf")

if st.button("Cargar el grafo"):
    G_loaded = nx.read_gexf("graph.gexf")
    st.write("Nodos del grafo cargado:", list(G_loaded.nodes()))
    st.write("Aristas del grafo cargado:", list(G_loaded.edges()))

    # Dibujar el grafo cargado
    pos_loaded = nx.spring_layout(G_loaded)
    edge_x_loaded = []
    edge_y_loaded = []
    for edge in G_loaded.edges():
        x0, y0 = pos_loaded[edge[0]]
        x1, y1 = pos_loaded[edge[1]]
        edge_x_loaded.append(x0)
        edge_x_loaded.append(x1)
        edge_x_loaded.append(None)
        edge_y_loaded.append(y0)
        edge_y_loaded.append(y1)
        edge_y_loaded.append(None)

    edge_trace_loaded = go.Scatter(
        x=edge_x_loaded, y=edge_y_loaded,
        line=dict(width=edge_width, color=edge_color),
        hoverinfo='none',
        mode='lines')

    node_x_loaded = []
    node_y_loaded = []
    node_text_loaded = []
    for node in G_loaded.nodes():
        x, y = pos_loaded[node]
        node_x_loaded.append(x)
        node_y_loaded.append(y)
        node_text_loaded.append(f"{node}: {G_loaded.nodes[node]['label']}")

    node_trace_loaded = go.Scatter(
        x=node_x_loaded, y=node_y_loaded,
        mode='markers+text',
        text=node_text_loaded,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig_loaded = go.Figure(data=[edge_trace_loaded, node_trace_loaded],
                           layout=go.Layout(
                               title='Loaded Network Graph of Named Entities',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20, l=5, r=5, t=40),
                               annotations=[dict(
                                   text="Loaded network graph visualization using Plotly",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002)],
                               xaxis=dict(showgrid=False, zeroline=False),
                               yaxis=dict(showgrid=False, zeroline=False))
                           )

    st.plotly_chart(fig_loaded, use_container_width=True)
