#!/usr/bin/env python3
import os
import json
#!/usr/bin/env python3
"""
Improved ticket generator that ensures each complaint description and its
resolution are unique and detailed. Overwrites existing `data/complaint_###.json` files.
"""
import os
import json
import random
from datetime import datetime, timedelta


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(20251115)

first_names = [
    "Aisha","Ben","Carlos","Deepa","Elena","Farah","George","Hannah",
    "Ibrahim","Jia","Kavya","Liam","Maya","Noah","Olivia","Priya",
    "Quentin","Ravi","Sara","Tomas","Uma","Victor","Wen","Xavier",
    "Yara","Zain"
]
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

# Richer complaint templates with slots; we'll make each final text unique by
# adding specific details per ticket (timestamps, serials, descriptors).
complaint_templates = [
    ("Wrong Item Delivered", "I ordered a {expected} (SKU {sku}) but received a {received} instead. The box label shows SKU {received_sku}.") ,
    ("Damaged on Arrival", "The item arrived damaged: {issue_description}. Damage appears to be from transit; packaging was {pack_condition}.") ,
    ("Late Delivery", "Order promised by {promised_date} did not arrive. Tracking shows last scan was on {last_scan_date}.") ,
    ("Missing Parts/Accessories", "Main product arrived but missing accessory: {missing}. Item requires this part (part code {part_code}) to function.") ,
    ("Defective Product", "Product fails to operate: {issue_description}. Attempts to power/reset produced error code {error_code}.") ,
    ("Sizing Issue", "Received item is the wrong size: ordered {ordered_size}, received {received_size}. Measurements differ by {diff_cm} cm in chest/width.") ,
    ("Billing Issue", "I was charged {charged_amount} but order total in confirmation was {order_total}. Transaction id: {txn_id}.") ,
    ("Return/Refund Request", "I'd like to return the product and request a refund due to {reason}. Order placed on {order_date}.") ,
    ("Account/Login Problem", "Cannot log in: I see error '{error_msg}' when attempting to sign in with email {email}.") ,
    ("Product Not As Described", "Product differs from listing: {difference}. The listing stated {listed_feature} but the received item {actual_feature}.")
]

resolution_templates = {
    'Wrong Item Delivered': [
        "Apologize, request a photo and the order packing slip, issue a prepaid return label, and dispatch the correct item ({expected}) with expedited delivery. Apply ${compensation} credit for inconvenience.",
        "Offer full refund if customer prefers; when return is received process refund within 3 business days and send follow-up survey."
    ],
    'Damaged on Arrival': [
        "Request photos, open a claim with the carrier, arrange replacement shipment, and provide ${compensation} coupon. If customer prefers, issue full refund upon return.",
        "Arrange immediate RMA with free pickup, fast-track replacement, and escalate to Quality for batch inspection."
    ],
    'Late Delivery': [
        "Check carrier proof-of-delivery, offer a 20% refund if late, and re-ship with overnight delivery if item is lost. Provide ongoing tracking updates.",
        "If missing >7 days, cancel order, refund in full, and provide $10 voucher for future purchase."
    ],
    'Missing Parts/Accessories': [
        "Confirm missing part code, ship replacement part with 1-day shipping, and apply ${compensation} goodwill credit.",
        "If part out of stock, offer full refund for accessory portion and provide alternative compatible part."
    ],
    'Defective Product': [
        "Provide guided troubleshooting steps; if unresolved, send return label and ship replacement. Offer ${compensation} discount on next order.",
        "Escalate to Technical team with device logs; offer a temporary replacement while we inspect returned unit."
    ],
    'Sizing Issue': [
        "Provide pre-paid return for exchange, include recommended size based on provided measurements, and apply ${compensation} off next purchase.",
        "Offer partial refund if customer keeps item; document feedback for product page sizing guidance."
    ],
    'Billing Issue': [
        "Review billing and transaction history, if overcharged issue refund to original payment within 3-5 business days and notify customer when complete.",
        "If duplicate charge found, reverse duplicate immediately and escalate to Finance for reconciliation."
    ],
    'Return/Refund Request': [
        "Provide RMA, pre-paid label, and process refund within 3 business days of receipt; if outside policy, offer store credit and explain options.",
        "Prioritize refund and offer a courtesy ${compensation} coupon for future purchase."
    ],
    'Account/Login Problem': [
        "Verify identity, reset password, clear any account lockouts, and provide one-time login link. Escalate to Engineering if persistent.",
        "Create temporary access and walk the user through multi-factor setup to prevent recurrence."
    ],
    'Product Not As Described': [
        "Offer return/refund or immediate replacement, collect photos, and open a catalog correction ticket; provide ${compensation} coupon.",
        "Provide partial refund if customer keeps the item and update listing copy to correct discrepancy."
    ]
}


def unique_serial(i):
    return f"SN-{20251115}-{i:04}-{random.randint(100,999)}"


for i in range(1, 101):
    # deterministic-ish but varied customer name
    cust = f"{random.choice(first_names)} {random.choice(last_names)}"
    product = random.choice(products)
    tpl = random.choice(complaint_templates)
    title = tpl[0]

    # dynamic fields to guarantee uniqueness
    expected = product['name']
    sku = product['sku']
    received_product = random.choice([p for p in products if p['sku'] != sku])
    received = received_product['name']
    received_sku = received_product['sku']
    issue_description = random.choice([
        "cracked plastic casing","motor failure on first use","deep paint scratch across panel","detached handle on opening",
        "leak from internal seal","unresponsive touch controls","battery not holding charge","straps tearing after one use"
    ])
    pack_condition = random.choice(["intact","torn","wet","crushed","partially open"])
    missing = random.choice(["power adapter","user manual","battery pack","charging cable","mounting screws","remote control"]) 
    part_code = f"PT-{random.randint(1000,9999)}"
    promised_date = (datetime.now() - timedelta(days=random.randint(2,14))).strftime('%Y-%m-%d')
    last_scan_date = (datetime.now() - timedelta(days=random.randint(1,10))).strftime('%Y-%m-%d')
    ordered_size = random.choice(["S","M","L","XL"]) 
    received_size = random.choice(["XS","S","M","L","XL"]) 
    diff_cm = round(random.uniform(1.0,6.5),1)
    charged_amount = f"${random.randint(10,500)}.00"
    order_total = f"${random.randint(10,500)}.00"
    txn_id = f"TXN{random.randint(100000,999999)}"
    reason = random.choice(["found a defect","changed mind","not as described","wrong size","delivered late"]) 
    order_date = (datetime.now() - timedelta(days=random.randint(5,60))).strftime('%Y-%m-%d')
    error_msg = random.choice(["Invalid credentials","Account locked","Two-factor failed","Token expired","Session mismatch"])
    email = f"{cust.split()[0].lower()}.{cust.split()[1].lower()}@example.com"
    difference = random.choice(["color mismatch","missing features","smaller than expected","different material","wrong model"])
    listed_feature = random.choice(["stated red finish","advertised 20hr battery","includes 4 accessories","size chart S=34in"])
    actual_feature = random.choice(["delivered blue finish","battery lasts 3hr","accessories missing","size runs small"])
    compensation = random.choice([5,10,15,20])
    error_code = f"ERR{random.randint(100,999)}"
    txn_id = f"TXN-{random.randint(1000000,9999999)}"
    serial = unique_serial(i)

    # craft complaint description and ensure uniqueness by appending serial/context
    description = tpl[1].format(
        expected=expected,
        sku=sku,
        received=received,
        received_sku=received_sku,
        issue_description=issue_description,
        pack_condition=pack_condition,
        missing=missing,
        part_code=part_code,
        promised_date=promised_date,
        last_scan_date=last_scan_date,
        ordered_size=ordered_size,
        received_size=received_size,
        diff_cm=diff_cm,
        charged_amount=charged_amount,
        order_total=order_total,
        txn_id=txn_id,
        reason=reason,
        order_date=order_date,
        error_msg=error_msg,
        email=email,
        difference=difference,
        listed_feature=listed_feature,
        actual_feature=actual_feature,
        error_code=error_code
    )
    # append a unique note for traceability
    description = f"{description} (ref: {serial})"

    # pick resolution and make it specific
    possible_res = resolution_templates.get(title, ["Investigate and follow-up."])
    chosen_res = random.choice(possible_res).replace("${compensation}", f"${compensation}")
    chosen_res = chosen_res.format(expected=expected)

    # build multi-step resolution (unique by including order id / serial / agent id specifics)
    resolution_steps = [
        "Apologize and acknowledge the issue within 2 business hours.",
        f"Request required proof (photo, order number {i:05}, item serial {serial}).",
        chosen_res,
        f"Confirm completion with customer and close ticket; log resolution code {random.randint(2000,9999)}."
    ]

    status = random.choices(["Resolved","Pending","In Progress","Escalated"], weights=[0.6,0.15,0.2,0.05])[0]

    ticket = {
        "id": f"TKT-{datetime.now().strftime('%Y%m%d')}-{i:03}",
        "customer_name": cust,
        "order_id": f"ORD{datetime.now().strftime('%Y%m')}-{i:05}",
        "product": {"sku": sku, "name": expected, "category": product['category']},
        "title": title,
        "complaint_description": description,
        "date_reported": (datetime.now() - timedelta(days=random.randint(0,60))).strftime('%Y-%m-%d'),
        "customer_sentiment": random.choice(["angry", "neutral", "frustrated", "concerned", "satisfied"]),
        "resolution_status": status,
        "resolution_steps": resolution_steps,
        "agent_notes": f"Assigned to support agent #{random.randint(1000,9999)}. Ticket serial {serial}.",
        "priority": random.choice(["Low","Medium","High"]),
        "internal": {
            "serial": serial,
            "txn_id": txn_id
        }
    }

    out_path = os.path.join(OUT_DIR, f"complaint_{i:03}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ticket, f, indent=2, ensure_ascii=False)

print(f"Generated 100 unique ticket files in: {OUT_DIR}")
