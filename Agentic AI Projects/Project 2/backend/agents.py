import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from backend.models import InvoiceData, ValidationResult, CategorizationResult, AuditResult, AnalysisResult

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# --- Agent 1: Extractor ---
extractor_parser = PydanticOutputParser(pydantic_object=InvoiceData)

def extractor_agent(image_path: str) -> InvoiceData:
    image_data = encode_image(image_path)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert invoice extractor. Extract the following information from the invoice image. Return JSON only."),
        ("user", [
            {"type": "text", "text": "Extract the invoice data."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ])
    ])
    
    chain = prompt | llm.with_structured_output(InvoiceData)
    return chain.invoke({})

# --- Agent 2: Validator ---
validator_parser = PydanticOutputParser(pydantic_object=ValidationResult)

validator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data validator. Check if the invoice data is mathematically correct and complete."),
    ("user", "Invoice Data: {invoice_data}\n\nCheck if subtotal + tax equals total amount (allow small rounding errors). Check if date and invoice number are present. Return JSON.")
])

validator_chain = validator_prompt | llm.with_structured_output(ValidationResult)

def validator_agent(invoice_data: InvoiceData) -> ValidationResult:
    return validator_chain.invoke({"invoice_data": invoice_data.model_dump_json()})

# --- Agent 3: Categorizer ---
categorizer_parser = PydanticOutputParser(pydantic_object=CategorizationResult)

categorizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an accounting assistant. Categorize the invoice based on the vendor and items."),
    ("user", "Invoice Data: {invoice_data}\n\nDetermine the expense category (e.g., Software, Hardware, Travel, Meals, Office Supplies) and the likely department (e.g., IT, Sales, HR, Operations). Return JSON.")
])

categorizer_chain = categorizer_prompt | llm.with_structured_output(CategorizationResult)

def categorizer_agent(invoice_data: InvoiceData) -> CategorizationResult:
    return categorizer_chain.invoke({"invoice_data": invoice_data.model_dump_json()})

# --- Agent 4: Auditor ---
auditor_parser = PydanticOutputParser(pydantic_object=AuditResult)

auditor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a risk auditor. Analyze the invoice for potential fraud or anomalies."),
    ("user", "Invoice Data: {invoice_data}\n\nCalculate a risk score (0-100, where 100 is high risk). Look for:\n- Round number totals\n- Weekend dates\n- Unusually high amounts\n- Missing details\n\nList any flags found. Return JSON.")
])

auditor_chain = auditor_prompt | llm.with_structured_output(AuditResult)

def auditor_agent(invoice_data: InvoiceData) -> AuditResult:
    return auditor_chain.invoke({"invoice_data": invoice_data.model_dump_json()})

# --- Agent 5: Analyst ---
analyst_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a business analyst. Provide a summary and insights for this invoice."),
    ("user", "Invoice Data: {invoice_data}\nCategory: {category}\nRisk Score: {risk_score}\n\nWrite a brief executive summary and list 2-3 key insights (e.g., spending trends, vendor relationship). Return JSON.")
])

analyst_chain = analyst_prompt | llm.with_structured_output(AnalysisResult)

def analyst_agent(invoice_data: InvoiceData, category: str, risk_score: int) -> AnalysisResult:
    return analyst_chain.invoke({
        "invoice_data": invoice_data.model_dump_json(),
        "category": category,
        "risk_score": risk_score
    })
