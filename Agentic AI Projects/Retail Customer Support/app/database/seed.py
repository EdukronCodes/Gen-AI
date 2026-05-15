"""Seed database with dummy retail data."""
from datetime import datetime, timedelta

from app.config.database import Base, SessionLocal, engine
from app.database.models import Customer, Order, OrderItem, Product, Refund, Ticket


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Customer).first():
            print("Database already seeded.")
            return

        customers = [
            Customer(email="alice@example.com", name="Alice Johnson", phone="+1-555-0101"),
            Customer(email="bob@example.com", name="Bob Smith", phone="+1-555-0102"),
            Customer(email="carol@example.com", name="Carol Williams", phone="+1-555-0103"),
        ]
        db.add_all(customers)
        db.flush()

        products = [
            Product(sku="WH-001", name="Wireless Headphones", description="Noise-cancelling over-ear headphones", category="Electronics", price=79.99, stock_quantity=150, image_url="/images/headphones.jpg"),
            Product(sku="KB-002", name="Mechanical Keyboard", description="RGB mechanical keyboard with Cherry MX switches", category="Electronics", price=129.99, stock_quantity=80, image_url="/images/keyboard.jpg"),
            Product(sku="MS-003", name="Ergonomic Mouse", description="Wireless ergonomic mouse for productivity", category="Electronics", price=49.99, stock_quantity=200, image_url="/images/mouse.jpg"),
            Product(sku="TS-004", name="Organic Cotton T-Shirt", description="Soft organic cotton crew neck tee", category="Apparel", price=24.99, stock_quantity=500, image_url="/images/tshirt.jpg"),
            Product(sku="JN-005", name="Slim Fit Jeans", description="Classic slim fit denim jeans", category="Apparel", price=59.99, stock_quantity=120, image_url="/images/jeans.jpg"),
            Product(sku="BK-006", name="Running Shoes", description="Lightweight running shoes with cushioned sole", category="Footwear", price=89.99, stock_quantity=75, image_url="/images/shoes.jpg"),
            Product(sku="MG-007", name="Stainless Water Bottle", description="32oz insulated stainless steel bottle", category="Home", price=34.99, stock_quantity=300, image_url="/images/bottle.jpg"),
            Product(sku="LP-008", name="Laptop Stand", description="Adjustable aluminum laptop stand", category="Electronics", price=39.99, stock_quantity=90, image_url="/images/stand.jpg"),
        ]
        db.add_all(products)
        db.flush()

        orders = [
            Order(order_number="ORD-10001", customer_id=customers[0].id, status="delivered", total_amount=209.98,
                  shipping_address="123 Oak St, Portland, OR 97201", tracking_number="1Z999AA10123456784",
                  carrier="UPS", estimated_delivery=datetime.utcnow() - timedelta(days=2)),
            Order(order_number="ORD-10002", customer_id=customers[0].id, status="shipped", total_amount=79.99,
                  shipping_address="123 Oak St, Portland, OR 97201", tracking_number="1Z999AA10123456785",
                  carrier="FedEx", estimated_delivery=datetime.utcnow() + timedelta(days=3)),
            Order(order_number="ORD-10003", customer_id=customers[1].id, status="processing", total_amount=154.98,
                  shipping_address="456 Pine Ave, Seattle, WA 98101"),
            Order(order_number="ORD-10004", customer_id=customers[2].id, status="cancelled", total_amount=59.99,
                  shipping_address="789 Maple Dr, Austin, TX 78701"),
        ]
        db.add_all(orders)
        db.flush()

        items = [
            OrderItem(order_id=orders[0].id, product_id=products[0].id, quantity=1, unit_price=79.99),
            OrderItem(order_id=orders[0].id, product_id=products[1].id, quantity=1, unit_price=129.99),
            OrderItem(order_id=orders[1].id, product_id=products[0].id, quantity=1, unit_price=79.99),
            OrderItem(order_id=orders[2].id, product_id=products[2].id, quantity=1, unit_price=49.99),
            OrderItem(order_id=orders[2].id, product_id=products[7].id, quantity=1, unit_price=39.99),
            OrderItem(order_id=orders[2].id, product_id=products[6].id, quantity=2, unit_price=34.99),
            OrderItem(order_id=orders[3].id, product_id=products[4].id, quantity=1, unit_price=59.99),
        ]
        db.add_all(items)

        refunds = [
            Refund(refund_number="REF-5001", order_id=orders[0].id, customer_id=customers[0].id,
                   amount=79.99, reason="Product arrived damaged", status="approved",
                   processed_at=datetime.utcnow() - timedelta(days=1)),
            Refund(refund_number="REF-5002", order_id=orders[3].id, customer_id=customers[2].id,
                   amount=59.99, reason="Order cancelled before shipping", status="completed",
                   processed_at=datetime.utcnow() - timedelta(days=5)),
        ]
        db.add_all(refunds)

        tickets = [
            Ticket(ticket_number="TKT-9001", customer_id=customers[0].id, subject="Headphones not pairing",
                   description="Bluetooth won't connect to my phone", status="open", priority="medium"),
            Ticket(ticket_number="TKT-9002", customer_id=customers[1].id, subject="Delayed shipment",
                   description="Order ORD-10003 has been processing for 5 days", status="in_progress", priority="high"),
        ]
        db.add_all(tickets)
        db.commit()
        print("Database seeded successfully with dummy data.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
