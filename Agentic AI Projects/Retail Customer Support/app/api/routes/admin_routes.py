from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.config.database import get_db
from app.database.models import Customer, Order, Ticket, Refund, Product
from app.api.dependencies.auth_dependency import get_current_user

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


@router.get("/analytics")
def analytics(db: Session = Depends(get_db), _user=Depends(get_current_user)):
    return {
        "customers": db.query(func.count(Customer.id)).scalar(),
        "orders": db.query(func.count(Order.id)).scalar(),
        "open_tickets": db.query(func.count(Ticket.id)).filter(Ticket.status == "open").scalar(),
        "pending_refunds": db.query(func.count(Refund.id)).filter(Refund.status == "pending").scalar(),
        "products": db.query(func.count(Product.id)).scalar(),
        "orders_by_status": {
            row[0]: row[1]
            for row in db.query(Order.status, func.count(Order.id)).group_by(Order.status).all()
        },
    }


@router.get("/tickets")
def list_tickets(db: Session = Depends(get_db), _user=Depends(get_current_user)):
    tickets = db.query(Ticket).order_by(Ticket.created_at.desc()).limit(50).all()
    return [
        {
            "ticket_number": t.ticket_number,
            "subject": t.subject,
            "status": t.status,
            "priority": t.priority,
            "created_at": t.created_at.isoformat(),
        }
        for t in tickets
    ]
