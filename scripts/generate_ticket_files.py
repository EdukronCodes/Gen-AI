#!/usr/bin/env python3
import os
import json
import random
from datetime import datetime, timedelta

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(OUT_DIR, exist_ok=True)

first_names = ["Aisha","Ben","Carlos","Deepa","Elena","Farah","George","Hannah","Ibrahim","Jia","Kavya","Liam","Maya","Noah","Olivia","Priya","Quentin","Ravi","Sara","Tomas","Uma","Victor","Wen","Xavier","Yara","Zain"]
last_names = ["Khan","Smith","Garcia","Patel","Wang","Brown","Martinez","Singh","Nguyen","Ramos","Lee","Johnson"]
products = [
    {"sku":"TSH-001","name":"Classic Cotton T-Shirt","category":"Apparel"},
    {"sku":"HDP-002","name":"Noise-Cancelling Headphones","category":"Electronics"},
    {"sku":"BLD-003","name":"Ergo Office Chair","category":"Furniture"},
    {"sku":"KTC-004","name":"12-piece Cookware Set","category":"Home & Kitchen"},
    {"sku":"SPT-005","name":"Trail Running Shoes","category":"Footwear"},
    {"sku":"ACC-006","name":"Leather Wallet","category":"Accessories"},
    {"sku":"BKS-007","name":"Bestseller Fiction Novel","category":"Books"},
    {"sku":"GAD-008","name":"Smart Home Hub","category":"Electronics"},
    {"sku":"CLN-009","name":"Cordless Vacuum","category":"Home Appliances"},
    {"sku":"TYS-010","name":"Kids Toy Set","category":"Toys"}
]

complaint_templates = [
    ("Wrong Item Delivered", "Customer received a different item than ordered. Expected {expected}, but received {received}."),
    ("Damaged on Arrival", "Product arrived damaged: {issue_description}.") ,
    ("Late Delivery", "Order did not arrive by the promised delivery date ({promised_date}). Customer still waiting.") ,
    ("Missing Parts/Accessories", "Main product arrived but missing accessory: {missing}.") ,
    ("Defective Product", "Product is not functioning as expected: {issue_description}.") ,
    ("Sizing Issue", "Item size does not match description (ordered {ordered_size}, received {received_size}).") ,
    ("Billing Issue", "Customer was charged incorrectly: charged {charged_amount} for order total {order_total}.") ,
    ("Return/Refund Request", "Customer wants to return item and request refund due to {reason}.") ,
    ("Account/Login Problem", "Customer cannot access their account; receives error: {error_msg}.") ,
    ("Product Not As Described", "Product differs from online description: {difference}.")
]

solutions_examples = {
    'Wrong Item Delivered': [
        "Apologize, request a photo of the delivered item, offer prepaid return label, and arrange a replacement shipment of {expected} with expedited shipping at no charge.",
        "Offer full refund if customer prefers; follow up when return tracking shows item received."
    ],
    'Damaged on Arrival': [
        "Ask for photos of damage, provide immediate replacement or full refund, and arrange pickup of damaged unit if needed.",
        "Escalate to quality team and provide 15% goodwill credit for inconvenience."
    ],
    'Late Delivery': [
        "Check carrier tracking, provide updated ETA, and offer 10% refund or coupon for future purchase. If lost, reorder item with expedited shipping.",
        "If delivery delayed beyond 7 days, offer full refund and free return label."
    ],
    'Missing Parts/Accessories': [
        "Confirm missing item, immediately ship the missing accessory with 2-day shipping and apply 5% discount for the inconvenience.",
        "Offer refund for accessory or replacement of entire unit if accessory required for use."
    ],
    'Defective Product': [
        "Troubleshoot remotely (steps provided), if unresolved arrange replacement or RMA; provide return label and 20% discount on next purchase.",
        "Escalate to technical team and keep customer informed every 24 hours until resolved."
    ],
    'Sizing Issue': [
        "Provide free return for exchange; include sizing guide and recommend alternate size based on measurements.",
        "Offer partial refund if customer keeps the item and it fits acceptably."
    ],
    'Billing Issue': [
        "Review transaction logs and provide corrected refund for the overcharge within 3-5 business days; apologize and offer $5 credit.",
        "If duplicate charge, reverse duplicate immediately and notify bank."
    ],
    'Return/Refund Request': [
        "Provide return authorization, pre-paid label, and process refund within 3 business days of receiving returned item.",
        "If out of return window, offer store credit and explain policy."
    ],
    'Account/Login Problem': [
        "Reset password flow, verify identity, check account lockouts and guide the user step-by-step. If issue persists, escalate to engineering with log details.",
        "Offer temporary access link valid for 1 hour while we troubleshoot."
    ],
    'Product Not As Described': [
        "Offer return/refund or replacement, investigate listing accuracy and update product page if incorrect; give 15% coupon for future purchase.",
        "Collect photos and details to escalate to catalog team for correction."
    ]
}

random.seed(42)

for i in range(1, 101):
    cust = f"{random.choice(first_names)} {random.choice(last_names)}"
    product = random.choice(products)
    tpl = random.choice(complaint_templates)
    title = tpl[0]

    # create some dynamic fields used in templates
    expected = product['name']
    received = random.choice([p['name'] for p in products if p['name'] != expected])
    issue_description = random.choice([
        "cracked screen","motor not spinning","deep scratch on surface","detached handle","units leaking liquid","buttons not responsive"
    ])
    missing = random.choice(["power adapter","user manual","battery pack","charging cable","mounting screws"]) 
    promised_date = (datetime.now() - timedelta(days=random.randint(0,7))).strftime('%Y-%m-%d')
    ordered_size = random.choice(["S","M","L","XL"]) 
    received_size = random.choice(["XS","S","M","L","XL"]) 
    charged_amount = f"${random.randint(10,500)}.00"
    order_total = f"${random.randint(10,500)}.00"
    reason = random.choice(["found a defect","changed mind","not as described","wrong size","delivered late"]) 
    error_msg = random.choice(["Invalid credentials","Account locked","Two-factor failed","Token expired"]) 
    difference = random.choice(["color mismatch","missing features","smaller than expected","different material"])

    # Format description using template
    description = tpl[1].format(
        expected=expected,
        received=received,
        issue_description=issue_description,
        missing=missing,
        promised_date=promised_date,
        ordered_size=ordered_size,
        received_size=received_size,
        charged_amount=charged_amount,
        order_total=order_total,
        reason=reason,
        error_msg=error_msg,
        difference=difference
    )

    # pick resolution suggestions
    solution_options = solutions_examples.get(title, ["Investigate and follow-up."])
    chosen_solution = random.choice(solution_options).format(expected=expected)

    resolution_steps = [
        "Acknowledge message and apologize for inconvenience.",
        "Request any required proof (photos, order number).",
        chosen_solution,
        "Confirm with customer when resolution completed and close the ticket."
    ]

    status = random.choices(["Resolved","Pending","In Progress","Escalated"], weights=[0.6,0.15,0.2,0.05])[0]

    ticket = {
        "id": f"TKT-{datetime.now().strftime('%Y%m%d')}-{i:03}",
        "customer_name": cust,
        "order_id": f"ORD{datetime.now().strftime('%Y%m')}-{i:05}",
        "product": product,
        "title": title,
        "complaint_description": description,
        "date_reported": (datetime.now() - timedelta(days=random.randint(0,30))).strftime('%Y-%m-%d'),
        "customer_sentiment": random.choice(["angry", "neutral", "frustrated", "concerned", "satisfied"]),
        "resolution_status": status,
        "resolution_steps": resolution_steps,
        "agent_notes": f"Assigned to support agent #{random.randint(1000,9999)}. Follow-up in 24-48 hours.",
        "priority": random.choice(["Low","Medium","High"]) 
    }

    out_path = os.path.join(OUT_DIR, f"complaint_{i:03}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ticket, f, indent=2, ensure_ascii=False)

print(f"Generated 100 ticket files in: {OUT_DIR}")
