from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    load_assets_node,
    prepare_assets_node,
    generative_api_node,
    placement_node,
    save_result_node,
    route_by_message,
)
import os
from dotenv import load_dotenv

load_dotenv()

def create_guitar_sticker_graph():
    """
    Creates and compiles a conditional LangGraph for the guitar sticker application.
    The graph has two main branches:
    1. Generative Path: Uses a text prompt to generate a sticker with an API.
    2. Placement Path: Uses OpenCV to blend a pre-existing sticker image.
    """
    workflow = StateGraph(GraphState)

    # --- Add nodes to the graph ---
    workflow.add_node("load_assets", load_assets_node)
    workflow.add_node("prepare_assets", prepare_assets_node)
    workflow.add_node("generative_api", generative_api_node)
    workflow.add_node("placement", placement_node)
    workflow.add_node("save_result", save_result_node)

    # --- Define the graph's structure ---
    
    # Start by loading and preparing assets
    workflow.set_entry_point("load_assets")
    workflow.add_edge("load_assets", "prepare_assets")

    # Add the conditional routing
    workflow.add_conditional_edges(
        "prepare_assets",
        route_by_message,
        {
            "generative_api": "generative_api",
            "placement": "placement",
        }
    )

    # Define how each branch proceeds
    workflow.add_edge("generative_api", "save_result")
    workflow.add_edge("placement", "save_result")

    # The graph finishes after the 'save_result' node
    workflow.add_edge("save_result", END)

    # Compile the graph
    app = workflow.compile()
    return app

guitar_sticker_graph = create_guitar_sticker_graph()