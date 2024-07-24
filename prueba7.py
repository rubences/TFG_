import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModel
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
import scann

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
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

ner_pipeline = load_ner_model()
tokenizer, classifier_model = load_classifier_model()

# Función para obtener embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = classifier_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Crear base de datos de ejemplos
examples = [
    ("Barack Obama was born in Hawaii. He was elected president in 2008.", "REAL"),
    ("The moon is made of cheese and it's owned by NASA.", "FAKE"),
    ("Albert Einstein developed the theory of relativity.", "REAL"),
    ("Unicorns are real and live in the Himalayas.", "FAKE"),
    ("The Great Wall of China is visible from space.", "FAKE"),
    ("The capital of France is Paris.", "REAL"),
    ("Dinosaurs still roam the Earth today.", "FAKE"),
    ("The human body has 206 bones.", "REAL"),
    ("There is a city called Atlantis underwater.", "FAKE"),
    ("Mount Everest is the highest mountain in the world.", "REAL"),
    ("Cows can fly during a full moon.", "FAKE"),
    ("The Pacific Ocean is the largest ocean on Earth.", "REAL"),
    ("Leprechauns hide pots of gold at the end of rainbows.", "FAKE"),
    ("Honey never spoils and can be eaten even after thousands of years.", "REAL"),
    ("The Earth is flat and rests on the back of a giant turtle.", "FAKE"),
    ("Shakespeare wrote 'Hamlet' and 'Romeo and Juliet'.", "REAL"),
    ("Eating carrots gives you night vision.", "FAKE"),
    ("The Amazon rainforest produces 20% of the world's oxygen.", "REAL"),
    ("Vampires are real and live in Transylvania.", "FAKE"),
    ("Lightning never strikes the same place twice.", "FAKE"),
    ("Bananas are berries but strawberries are not.", "REAL"),
    ("The Eiffel Tower can be taller during hot days.", "REAL"),
    ("An octopus has three hearts.", "REAL"),
    ("Goldfish have a memory span of three seconds.", "FAKE"),
    ("A group of flamingos is called a 'flamboyance'.", "REAL"),
    ("Humans only use 10% of their brains.", "FAKE"),
    ("The longest river in the world is the Nile.", "REAL"),
    ("Eating turkey makes you sleepy because of tryptophan.", "FAKE"),
    ("The shortest war in history lasted 38 minutes.", "REAL"),
    ("A day on Venus is longer than a year on Venus.", "REAL"),
    ("Bulls get angry when they see the color red.", "FAKE"),
    ("A sneeze can travel up to 100 miles per hour.", "REAL"),
    ("Water spirals down the drain in opposite directions in the Northern and Southern Hemispheres.", "FAKE"),
    ("The inventor of the Pringles can is now buried in one.", "REAL"),
    ("Humans share 50% of their DNA with bananas.", "REAL"),
    ("Napoleon Bonaparte was extremely short.", "FAKE"),
    ("Walt Disney was cryogenically frozen.", "FAKE"),
    ("You can see the Great Wall of China from the moon.", "FAKE"),
    ("There are more stars in the universe than grains of sand on all the world's beaches.", "REAL"),
    ("An apple a day keeps the doctor away.", "FAKE"),
    ("The heart of a shrimp is located in its head.", "REAL"),
    ("Your fingernails and hair continue to grow after you die.", "FAKE"),
    ("A bolt of lightning contains enough energy to toast 100,000 slices of bread.", "REAL"),
    ("Humans have only five senses.", "FAKE"),
    ("The unicorn is the national animal of Scotland.", "REAL"),
    ("A snail can sleep for three years.", "REAL"),
    ("Coffee is made from berries.", "REAL"),
    ("The shortest verse in the Bible is 'Jesus wept'.", "REAL"),
    ("Goldfish can see infrared and ultraviolet light.", "REAL"),
    ("It takes seven years to digest swallowed gum.", "FAKE"),
    ("The inventor of the light bulb was Thomas Edison.", "REAL"),
    ("Twinkies have an infinite shelf life.", "FAKE"),
    ("The Atlantic Ocean is saltier than the Pacific Ocean.", "REAL"),
]

# Crear embeddings para ejemplos y asegurarse de que sean bidimensionales
example_embeddings = np.array([get_embedding(text).reshape(-1) for text, label in examples])
labels = [label for text, label in examples]

# Configurar ScaNN
searcher = scann.scann_ops_pybind.builder(example_embeddings, 10, "dot_product").tree(
    num_leaves=1, num_leaves_to_search=1, training_sample_size=len(example_embeddings)).score_ah(
    2, anisotropic_quantization_threshold=0.2).build()

# Obtener embedding para la oración introducida y asegurarse de que sea unidimensional
sentence_embedding = get_embedding(sentence).reshape(-1)

# Buscar los ejemplos más similares
neighbors, distances = searcher.search(sentence_embedding)

# Determinar veracidad basada en los ejemplos más similares
neighbor_labels = [labels[i] for i in neighbors]
veracity = max(set(neighbor_labels), key=neighbor_labels.count)
confidence = distances.mean() if veracity == "REAL" else (1 - distances.mean())
st.write(f"Clasificación del texto: {veracity} (Confianza: {confidence:.2f})")

# Gráfica del porcentaje de veracidad
fig_veracity = go.Figure(go.Indicator(
    mode="gauge+number",
    value=confidence * 100,
    title={'text': "Porcentaje de Veracidad"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "green" if veracity == "REAL" else "red"}}
))

st.plotly_chart(fig_veracity, use_container_width=True)

# Visualizar similitud del coseno
similarity_scores = [1 - distance for distance in distances]  # Cosine similarity
fig_similarity = go.Figure(data=[go.Bar(x=[f"Ejemplo {i+1}" for i in range(len(similarity_scores))],
                                        y=similarity_scores)])
fig_similarity.update_layout(title="Similitud del Coseno con Ejemplos",
                             xaxis_title="Ejemplos",
                             yaxis_title="Similitud del Coseno")
st.plotly_chart(fig_similarity, use_container_width=True)

# Mostrar ejemplos similares
st.write("Ejemplos más similares:")
for i, neighbor in enumerate(neighbors):
    st.write(f"Ejemplo {i+1}: {examples[neighbor][0]} (Similitud: {similarity_scores[i]:.2f})")

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

   
