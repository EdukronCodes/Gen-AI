from typing import List, Optional
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    description: str = Field(description="Description of the item")
    quantity: float = Field(description="Quantity of the item")
    unit_price: float = Field(description="Unit price of the item")
    amount: float = Field(description="Total amount for the line item")

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = Field(description="Invoice number")
    date: Optional[str] = Field(description="Invoice date")
    vendor_name: Optional[str] = Field(description="Name of the vendor")
    line_items: List[LineItem] = Field(description="List of line items")
    subtotal: float = Field(description="Subtotal amount")
    tax: float = Field(description="Tax amount")
    total_amount: float = Field(description="Total invoice amount")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the invoice is valid")
    errors: List[str] = Field(description="List of validation errors")

class CategorizationResult(BaseModel):
    category: str = Field(description="Category of the expense (e.g., Software, Travel)")
    department: str = Field(description="Department responsible (e.g., IT, Sales)")

class AuditResult(BaseModel):
    risk_score: int = Field(description="Risk score from 0 to 100")
    flags: List[str] = Field(description="List of risk flags or anomalies")

class AnalysisResult(BaseModel):
    summary: str = Field(description="Executive summary of the invoice")
    insights: List[str] = Field(description="Key insights derived from the invoice")

class InvoiceState(BaseModel):
    image_path: str
    extracted_data: Optional[InvoiceData] = None
    validation_result: Optional[ValidationResult] = None
    categorization_result: Optional[CategorizationResult] = None
    audit_result: Optional[AuditResult] = None
    analysis_result: Optional[AnalysisResult] = None
