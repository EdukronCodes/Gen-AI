## Retail Assistant Knowledge Base (sample docs to index)

This file is an example of **unstructured operational content** you would index into FAISS/ChromaDB/Azure Cognitive Search.
In production these would be multiple documents with metadata (doc_type, effective_date, store_scope, region_scope).

---

### DOC:POLICY:INVENTORY_TERMS (effective: 2026-01-01)
**Definitions**
- **On-hand**: physical units present in store/DC at the time of snapshot.
- **Reserved**: units allocated to open orders or internal holds (do not promise to new customers).
- **Available-to-Promise (ATP)**: \( \max(\text{on-hand} - \text{reserved}, 0) \)

**Low-stock guidance**
- If ATP < reorder point, mark as **LOW**.
- If ATP < safety stock, mark as **CRITICAL** and prioritize replenishment or transfer.

---

### DOC:POLICY:OFFERS_AND_COUPONS (effective: 2026-02-01)
- Offers are identified by `offer_id` and may apply by **SKU** or **category**.
- **Non-combinable rule**: SKU-specific flash sale offers are not combinable with other coupons unless explicitly stated.
- **Max quantity rule**: category-wide electronics offers may cap quantity to **1 per SKU per order**.
- If a customer asks to “stack” offers, respond with:
  - whether stacking is allowed
  - the maximum discount policy (if applicable)
  - how the system chooses the best single offer when stacking is disallowed

---

### DOC:SOP:PRICE_QUERIES (effective: 2026-01-01)
When asked about product price:
- Prefer the latest **list price** from the pricing table for the current effective date.
- If an offer applies:
  - show both list price and discounted price
  - include offer name, discount %, and validity window

If the user does not specify a channel:
- assume “ALL” where available; otherwise list channel-specific differences.

---

### DOC:POLICY:RETURNS (effective: 2025-10-01)
- Electronics: return within **7 days** of delivery, unopened packaging preferred.
- Apparel/Footwear: return within **15 days** with tags/unused condition.
- FMCG/Grocery: returns only for damaged/incorrect items within **48 hours**.

Refund timelines:
- Card/UPI: 3–5 business days after inspection approval.
- Cash: in-store refund depends on store policy; confirm at POS.

---

### DOC:SLA:SUPPLIER_SUPPORT (effective: 2026-01-01)
- For supplier issues (delays/defects), collect:
  - supplier name/id, sku, store, incident date, and evidence (photos if defects)
- Standard escalation:
  - T+0: open ticket via supplier support email
  - T+2 days: escalate to procurement if no response

---

### DOC:STORE:CONTACTS (effective: 2026-01-01)
Sample contacts (replace with your org directory):
- Procurement Desk: procurement@example
- Ops Control Tower: ops@example
- Store Support: store-support@example

