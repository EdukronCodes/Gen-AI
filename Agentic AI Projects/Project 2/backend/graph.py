from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from backend.models import InvoiceState
from backend.agents import (
    extractor_agent,
    validator_agent,
    categorizer_agent,
    auditor_agent,
    analyst_agent
)

# Define the nodes
def extractor_node(state: InvoiceState):
    print("--- Extractor Node ---")
    invoice_data = extractor_agent(state.image_path)
    return {"extracted_data": invoice_data}

def validator_node(state: InvoiceState):
    print("--- Validator Node ---")
    validation_result = validator_agent(state.extracted_data)
    return {"validation_result": validation_result}

def categorizer_node(state: InvoiceState):
    print("--- Categorizer Node ---")
    categorization_result = categorizer_agent(state.extracted_data)
    return {"categorization_result": categorization_result}

def auditor_node(state: InvoiceState):
    print("--- Auditor Node ---")
    audit_result = auditor_agent(state.extracted_data)
    return {"audit_result": audit_result}

def analyst_node(state: InvoiceState):
    print("--- Analyst Node ---")
    analysis_result = analyst_agent(
        state.extracted_data,
        state.categorization_result.category,
        state.audit_result.risk_score
    )
    return {"analysis_result": analysis_result}

# Build the graph
workflow = StateGraph(InvoiceState)

workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)
workflow.add_node("categorizer", categorizer_node)
workflow.add_node("auditor", auditor_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("extractor")

workflow.add_edge("extractor", "validator")
workflow.add_edge("validator", "categorizer")
workflow.add_edge("categorizer", "auditor")
workflow.add_edge("auditor", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()
