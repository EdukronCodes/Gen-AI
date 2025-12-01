"""
Reporting Service
Generates management reports
"""
from typing import Dict, Any
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.ticket import Ticket, TicketStatus
from datetime import datetime, timedelta


class ReportingService:
    """Service for generating reports"""
    
    def __init__(self):
        pass
    
    def _get_db(self):
        """Get database session"""
        return next(get_db())
    
    async def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly management report"""
        db = self._get_db()
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)
            
            # Get tickets
            tickets = db.query(Ticket).filter(
                Ticket.created_at >= start_date,
                Ticket.created_at <= end_date
            ).all()
        
        total_tickets = len(tickets)
        resolved_tickets = [t for t in tickets if t.status == TicketStatus.RESOLVED]
        auto_resolved = [t for t in resolved_tickets if t.auto_resolved == "true"]
        
        # Calculate metrics
        auto_resolution_rate = (len(auto_resolved) / total_tickets * 100) if total_tickets > 0 else 0
        
        # Calculate MTTR (Mean Time To Resolution)
        resolved_with_times = [
            t for t in resolved_tickets 
            if t.resolved_at and t.created_at
        ]
        if resolved_with_times:
            mttr_hours = sum(
                [(t.resolved_at - t.created_at).total_seconds() / 3600 
                 for t in resolved_with_times]
            ) / len(resolved_with_times)
        else:
            mttr_hours = 0
        
        # SLA compliance
        sla_met = sum(1 for t in resolved_tickets 
                     if t.sla_deadline and t.resolved_at and t.resolved_at <= t.sla_deadline)
        sla_compliance = (sla_met / len(resolved_tickets) * 100) if resolved_tickets else 0
        
        report = {
            "id": f"RPT-{datetime.utcnow().strftime('%Y%m%d')}",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": {
                "total_tickets": total_tickets,
                "resolved_tickets": len(resolved_tickets),
                "auto_resolved_tickets": len(auto_resolved),
                "auto_resolution_rate": round(auto_resolution_rate, 2),
                "mttr_hours": round(mttr_hours, 2),
                "sla_compliance_percent": round(sla_compliance, 2)
            },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return report
        finally:
            db.close()
    
    async def send_report(self, report: Dict[str, Any]):
        """Send report to management"""
        # In production, this would send email/PDF
        print(f"📊 Weekly Report Generated: {report.get('id')}")
        print(f"   Auto-Resolution Rate: {report['metrics']['auto_resolution_rate']}%")
        print(f"   SLA Compliance: {report['metrics']['sla_compliance_percent']}%")
        print(f"   MTTR: {report['metrics']['mttr_hours']} hours")

