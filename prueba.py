import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the pre-trained model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Define the text
sentence = "Barack Obama was born in Hawaii. He was elected president in 2008."

# Extract the entities
entities = ner_pipeline(sentence)

# Print the entities
for entity in entities:
    print(entity)

# Add the entities as nodes

G = nx.DiGraph()

# Add nodes
for entity in entities:
    G.add_node(entity['word'], label=entity['entity'])

# Add edges
for i in range(len(entities) - 1):
    G.add_edge(entities[i]['word'], entities[i+1]['word'])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_color='black')
labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Save the graph
nx.write_gexf(G, "graph.gexf")

# Load the graph

G = nx.read_gexf("graph.gexf")

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_color='black')
labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Print the nodes and edges
print("Nodes:", G.nodes())
print("Edges:", G.edges())

