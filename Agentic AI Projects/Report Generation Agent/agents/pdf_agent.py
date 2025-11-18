"""
PDF Generation Agent - Specialized in creating PDF summaries
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os
import json

class PDFAgent(BaseAgent):
    """Agent responsible for generating PDF summaries"""
    
    def __init__(self, output_dir: str = "pdf_outputs", gemini_client=None):
        super().__init__("PDFAgent", "PDF Generation Specialist", gemini_client)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate PDF summary from question and analysis
        """
        self.log("Generating PDF summary")
        
        if not context:
            return {
                "agent": self.name,
                "pdf_path": None,
                "status": "error",
                "error": "No context provided"
            }
        
        original_question = context.get("original_question", task)
        analysis = context.get("analysis", "")
        summary = context.get("summary", "")
        query_results = context.get("query_results", [])
        sql_query = context.get("sql_query", "")
        
        # Generate PDF
        pdf_path = await self._generate_pdf(
            question=original_question,
            summary=summary,
            analysis=analysis,
            query_results=query_results,
            sql_query=sql_query
        )
        
        return {
            "agent": self.name,
            "pdf_path": pdf_path,
            "status": "success"
        }
    
    async def _generate_pdf(self, question: str, summary: str, analysis: str, 
                           query_results: list, sql_query: str) -> str:
        """Generate the PDF document"""
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_summary_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("Query Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadata
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Question Section
        story.append(Paragraph("Question Asked", heading_style))
        story.append(Paragraph(f'<font color="#424242">{question}</font>', styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Summary Section
        if summary:
            story.append(Paragraph("Executive Summary", heading_style))
            # Split summary into paragraphs
            summary_paragraphs = summary.split('\n\n')
            for para in summary_paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Analysis Section
        if analysis:
            story.append(Paragraph("Detailed Analysis", heading_style))
            # Split analysis into paragraphs
            analysis_paragraphs = analysis.split('\n\n')
            for para in analysis_paragraphs:
                if para.strip():
                    # Check if it's a heading (starts with number or bold)
                    if para.strip()[0].isdigit() or para.strip().startswith('**'):
                        story.append(Paragraph(f'<b>{para.strip()}</b>', styles['Normal']))
                    else:
                        story.append(Paragraph(para.strip(), styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # SQL Query Section
        if sql_query:
            story.append(Paragraph("SQL Query Executed", heading_style))
            story.append(Paragraph(f'<font face="Courier" size="9" color="#616161">{sql_query}</font>', styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Results Section
        if query_results:
            story.append(PageBreak())
            story.append(Paragraph("Query Results", heading_style))
            story.append(Paragraph(f"Total Records: {len(query_results)}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Create table from results (limit to first 20 rows for readability)
            display_results = query_results[:20]
            if display_results:
                # Get column names
                columns = list(display_results[0].keys())
                
                # Create table data
                table_data = [columns]  # Header
                for row in display_results:
                    table_data.append([str(row.get(col, ''))[:50] for col in columns])  # Truncate long values
                
                # Create table
                table = Table(table_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]))
                
                story.append(table)
                
                if len(query_results) > 20:
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph(
                        f"<i>Note: Showing first 20 of {len(query_results)} results</i>",
                        styles['Normal']
                    ))
        
        # Build PDF
        doc.build(story)
        
        self.log(f"PDF generated: {filepath}")
        return filepath

