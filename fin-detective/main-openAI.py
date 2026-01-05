import json # Used to save the final graph to a file
import os # Used to handle file paths and environment variables
from typing import List # For type hinting our JSON structure
from pydantic import BaseModel, Field # To enforce a strict JSON schema
import networkx as nx # To create the mathematical graph structure
import matplotlib.pyplot as plt # To draw and save the graph visualization
from openai import OpenAI # The interface to talk to the LLM (GPT-4o)

# --- STEP 1: Define the "Blueprint" (Schema) ---
# We define exactly what an "Entity" looks like
class Entity(BaseModel):
    name: str = Field(description="The name of the company, person, or dollar amount")
    label: str = Field(description="The category: e.g., COMPANY, RISK, AMOUNT")

# We define exactly what a "Relationship" looks like
class Relationship(BaseModel):
    source: str = Field(description="The starting entity name")
    target: str = Field(description="The ending entity name")
    type: str = Field(description="The verb connecting them, e.g., OWNS, DEBT_OF")

# This is the master container for our JSON output
class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# --- STEP 2: The Extraction Function ---
def extract_financial_graph(text_content):
    # Initialize the OpenAI client (requires an API key in your environment)
    client = OpenAI(api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    # The prompt explains the 'rules' to the AI
    system_prompt = (
        "You are a Financial Detective. Extract entities and relationships from the text. "
        "Output ONLY valid JSON. No talk, no regex, just the structure provided."
    )

    # Call the LLM with 'Structured Output' enabled
    # We use response_format to force the LLM to follow our Pydantic model
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract the graph from this text: {text_content}"}
        ],
        response_format=KnowledgeGraph,
    )

    # Return the validated data as a Python dictionary
    return completion.choices[0].message.parsed

# --- STEP 3: The Visualization Function ---
def visualize_graph(graph_data):
    # Create an empty Directed Graph object
    G = nx.DiGraph()

    # Loop through the extracted entities and add them as 'nodes'
    for ent in graph_data.entities:
        G.add_node(ent.name, label=ent.label)

    # Loop through the extracted relationships and add them as 'edges' (lines)
    for rel in graph_data.relationships:
        G.add_edge(rel.source, rel.target, type=rel.type)

    # Set the layout for the nodes (spring layout looks like a web)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8)) # Set the size of the image

    # Draw nodes and lines
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    
    # Draw the labels on the lines (the relationship types)
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Save the result as a screenshot/image
    plt.savefig("graph_screenshot.png")
    print("Graph visualization saved as graph_screenshot.png")

# --- STEP 4: Execution Logic ---
if __name__ == "__main__":
    # 1. Read the raw text file (Version 1)
    with open("C:\\learn-AI\\fin-detective\\finreport.txt", "r") as file:
        raw_text = file.read()

    # 2. Extract the data using the LLM
    structured_data = extract_financial_graph(raw_text)

    # 3. Save to a strict JSON file
    with open("graph_output.json", "w") as f:
        # Convert Pydantic object to a dictionary, then to a JSON string
        f.write(structured_data.model_dump_json(indent=4))
    
    # 4. Create the visual representation
    visualize_graph(structured_data)
    print("Task Complete: JSON and Screenshot generated.")