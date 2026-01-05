import json
import ollama # Local LLM interface
from pydantic import BaseModel, Field # For JSON structure validation
from typing import List
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. DEFINE THE DATA STRUCTURE ---
# This ensures the LLM output is always in the same format
class Entity(BaseModel):
    name: str = Field(description="Name of the company, risk, or amount")
    label: str = Field(description="Type: COMPANY, RISK, or CURRENCY")

class Relationship(BaseModel):
    source: str = Field(description="The starting entity")
    target: str = Field(description="The ending entity")
    type: str = Field(description="Relationship type (e.g., OWNS, IMPACTS)")

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# --- 2. EXTRACTION LOGIC USING LOCAL LLAMA ---
def extract_with_llama(text_content):
    # Call the local Ollama server
    response = ollama.chat(
        model='llama3:8b-instruct-q4_K_M',
        messages=[
            {
                'role': 'system', 
                'content': 'You are a financial analyst. Extract entities and relationships into JSON. Use only the provided schema.'
            },
            {
                'role': 'user', 
                'content': f'Extract from this text: {text_content}'
            }
        ],
        # format=... uses Pydantic to force the model to output valid JSON
        format=KnowledgeGraph.model_json_schema(),
        options={'temperature': 0} # Set to 0 for consistent, factual results
    )
    
    # Parse the raw string response into our Pydantic model
    return KnowledgeGraph.model_validate_json(response['message']['content'])

# --- 3. VISUALIZATION ---
def create_visual(graph_data):
    G = nx.DiGraph() # Create a Directed Graph
    
    # Add nodes (Entities)
    for ent in graph_data.entities:
        G.add_node(ent.name, label=ent.label)
    
    # Add edges (Relationships)
    for rel in graph_data.relationships:
        G.add_edge(rel.source, rel.target, type=rel.type)

    # Drawing the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5) # Space out the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2500, font_size=10, font_weight='bold')
    
    # Label the connection lines
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.savefig("financial_graph.png")
    print("Graph image saved as 'financial_graph.png'")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # Load your messy text file
        with open("C:\\learn-AI\\fin-detective\\finreport.txt", "r", encoding="utf-8") as f:
            report_text = f.read()

        # Run the "Detective"
        print("Analyzing with local Llama 3... this may take a moment.")
        result_graph = extract_with_llama(report_text)

        # Save the JSON file
        with open("graph_output.json", "w") as f:
            json.dump(result_graph.model_dump(), f, indent=4)
        
        # Create the picture
        create_visual(result_graph)
        print("Success! JSON and Visual representation generated.")

    except FileNotFoundError:
        print("Error: 'finreport.txt' not found in this folder.")
    except Exception as e:
        print(f"An error occurred: {e}")