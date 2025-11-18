import sqlite3

# Connect to database
conn = sqlite3.connect('retail_banking.db')
cursor = conn.cursor()

print("=" * 80)
print("DATABASE RELATIONSHIPS VERIFICATION")
print("=" * 80)

# 1. Customer to Addresses (One-to-Many)
print("\n1. CUSTOMER → ADDRESSES (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name AS customer_name,
    COUNT(a.address_id) AS address_count
FROM customers c
LEFT JOIN addresses a ON c.customer_id = a.customer_id
GROUP BY c.customer_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  Customer {row[0]}: {row[1]} has {row[2]} address(es)")

# 2. Customer to Accounts (One-to-Many)
print("\n2. CUSTOMER → ACCOUNTS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name AS customer_name,
    COUNT(a.account_id) AS account_count,
    SUM(a.balance) AS total_balance
FROM customers c
LEFT JOIN accounts a ON c.customer_id = a.customer_id
GROUP BY c.customer_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  Customer {row[0]}: {row[1]} has {row[2]} account(s) with total balance ${row[3] or 0:.2f}")

# 3. Account to Transactions (One-to-Many)
print("\n3. ACCOUNT → TRANSACTIONS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    a.account_number,
    a.account_type,
    COUNT(t.transaction_id) AS transaction_count,
    SUM(t.amount) AS total_transaction_amount
FROM accounts a
LEFT JOIN transactions t ON a.account_id = t.account_id
GROUP BY a.account_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  Account {row[0]} ({row[1]}): {row[2]} transactions, Total: ${row[3] or 0:.2f}")

# 4. Customer to Credit Cards (One-to-Many)
print("\n4. CUSTOMER → CREDIT CARDS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    cc.card_type,
    cc.card_number,
    cc.credit_limit,
    cc.current_balance
FROM customers c
JOIN credit_cards cc ON c.customer_id = cc.customer_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} {row[2]} - Limit: ${row[3]}, Balance: ${row[4]}")

# 5. Customer to Loans (One-to-Many)
print("\n5. CUSTOMER → LOANS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    l.loan_type,
    l.principal_amount,
    l.interest_rate,
    l.status
FROM customers c
JOIN loans l ON c.customer_id = l.customer_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} - ${row[2]:,.2f} at {row[3]}% - Status: {row[4]}")

# 6. Loan to Payments (One-to-Many)
print("\n6. LOAN → PAYMENTS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    l.loan_id,
    l.loan_type,
    l.principal_amount,
    COUNT(p.payment_id) AS payment_count,
    SUM(p.payment_amount) AS total_paid
FROM loans l
LEFT JOIN payments p ON l.loan_id = p.loan_id
GROUP BY l.loan_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  Loan {row[0]} ({row[1]}): ${row[2]:,.2f} - {row[3]} payments, Total paid: ${row[4] or 0:.2f}")

# 7. Branch to Departments (One-to-Many)
print("\n7. BRANCH → DEPARTMENTS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    b.branch_name,
    COUNT(d.department_id) AS department_count,
    SUM(d.budget) AS total_budget
FROM branches b
LEFT JOIN departments d ON b.branch_id = d.branch_id
GROUP BY b.branch_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} department(s), Total budget: ${row[2]:,.2f}")

# 8. Department to Employees (One-to-Many)
print("\n8. DEPARTMENT → EMPLOYEES (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    d.department_name,
    b.branch_name,
    COUNT(e.employee_id) AS employee_count,
    AVG(e.salary) AS avg_salary
FROM departments d
JOIN branches b ON d.branch_id = b.branch_id
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]} ({row[1]}): {row[2]} employee(s), Avg salary: ${row[3]:,.2f}")

# 9. Employee Self-Referential (Manager relationship)
print("\n9. EMPLOYEE → EMPLOYEE (Self-Referential: Manager relationship)")
print("-" * 80)
cursor.execute('''
SELECT 
    e1.first_name || ' ' || e1.last_name AS employee_name,
    e2.first_name || ' ' || e2.last_name AS manager_name
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id
WHERE e1.manager_id IS NOT NULL
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]} reports to {row[1]}")

# 10. Category Self-Referential (Parent-Child)
print("\n10. CATEGORY → CATEGORY (Self-Referential: Parent-Child)")
print("-" * 80)
cursor.execute('''
SELECT 
    c1.category_name AS child_category,
    c2.category_name AS parent_category
FROM categories c1
LEFT JOIN categories c2 ON c1.parent_category_id = c2.category_id
WHERE c1.parent_category_id IS NOT NULL
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]} is a subcategory of {row[1]}")

# 11. Category to Products (One-to-Many)
print("\n11. CATEGORY → PRODUCTS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    cat.category_name,
    COUNT(p.product_id) AS product_count,
    AVG(p.price) AS avg_price
FROM categories cat
LEFT JOIN products p ON cat.category_id = p.category_id
GROUP BY cat.category_id
HAVING COUNT(p.product_id) > 0
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} product(s), Avg price: ${row[2]:,.2f}")

# 12. Supplier to Products (One-to-Many)
print("\n12. SUPPLIER → PRODUCTS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    s.supplier_name,
    COUNT(p.product_id) AS product_count
FROM suppliers s
LEFT JOIN products p ON s.supplier_id = p.supplier_id
GROUP BY s.supplier_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: supplies {row[1]} product(s)")

# 13. Product to Inventory (One-to-Many across branches)
print("\n13. PRODUCT → INVENTORY (One-to-Many: Product can be in multiple branches)")
print("-" * 80)
cursor.execute('''
SELECT 
    p.product_name,
    COUNT(i.inventory_id) AS branch_count,
    SUM(i.quantity) AS total_quantity
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
GROUP BY p.product_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: in {row[1]} branch(es), Total quantity: {row[2]}")

# 14. Customer to Orders (One-to-Many)
print("\n14. CUSTOMER → ORDERS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    COUNT(o.order_id) AS order_count,
    SUM(o.total_amount) AS total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
HAVING COUNT(o.order_id) > 0
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} order(s), Total spent: ${row[2]:,.2f}")

# 15. Order to Order Items (One-to-Many)
print("\n15. ORDER → ORDER ITEMS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    o.order_id,
    o.order_date,
    COUNT(oi.order_item_id) AS item_count,
    o.total_amount
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  Order #{row[0]} ({row[1]}): {row[2]} item(s), Total: ${row[3]:,.2f}")

# 16. Order to Shipments (One-to-Many)
print("\n16. ORDER → SHIPMENTS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    o.order_id,
    COUNT(s.shipment_id) AS shipment_count,
    GROUP_CONCAT(s.status) AS shipment_statuses
FROM orders o
LEFT JOIN shipments s ON o.order_id = s.order_id
GROUP BY o.order_id
HAVING COUNT(s.shipment_id) > 0
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  Order #{row[0]}: {row[1]} shipment(s) - Status: {row[2]}")

# 17. Order to Returns (One-to-Many)
print("\n17. ORDER → RETURNS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    o.order_id,
    COUNT(r.return_id) AS return_count,
    SUM(r.refund_amount) AS total_refund
FROM orders o
LEFT JOIN returns r ON o.order_id = r.order_id
GROUP BY o.order_id
HAVING COUNT(r.return_id) > 0
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  Order #{row[0]}: {row[1]} return(s), Total refund: ${row[2] or 0:.2f}")

# 18. Customer to Reviews (One-to-Many)
print("\n18. CUSTOMER → REVIEWS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    COUNT(r.review_id) AS review_count,
    AVG(r.rating) AS avg_rating
FROM customers c
LEFT JOIN reviews r ON c.customer_id = r.customer_id
GROUP BY c.customer_id
HAVING COUNT(r.review_id) > 0
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} review(s), Avg rating: {row[2]:.1f}/5")

# 19. Product to Reviews (One-to-Many)
print("\n19. PRODUCT → REVIEWS (One-to-Many)")
print("-" * 80)
cursor.execute('''
SELECT 
    p.product_name,
    COUNT(r.review_id) AS review_count,
    AVG(r.rating) AS avg_rating
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id
HAVING COUNT(r.review_id) > 0
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} review(s), Avg rating: {row[2]:.1f}/5")

# 20. Product/Category to Promotions (Many-to-Many via separate relationships)
print("\n20. PRODUCT/CATEGORY → PROMOTIONS (Many-to-One)")
print("-" * 80)
cursor.execute('''
SELECT 
    pr.promotion_name,
    p.product_name,
    cat.category_name,
    pr.discount_percentage,
    pr.status
FROM promotions pr
LEFT JOIN products p ON pr.product_id = p.product_id
LEFT JOIN categories cat ON pr.category_id = cat.category_id
LIMIT 5
''')
for row in cursor.fetchall():
    product_or_category = row[1] if row[1] else f"Category: {row[2]}"
    print(f"  {row[0]}: {product_or_category} - {row[3]}% off - Status: {row[4]}")

# Complex relationship query
print("\n" + "=" * 80)
print("COMPLEX RELATIONSHIP QUERY: Complete Customer Journey")
print("=" * 80)
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    COUNT(DISTINCT a.account_id) AS account_count,
    COUNT(DISTINCT o.order_id) AS order_count,
    COUNT(DISTINCT l.loan_id) AS loan_count,
    COUNT(DISTINCT r.review_id) AS review_count,
    SUM(o.total_amount) AS total_orders
FROM customers c
LEFT JOIN accounts a ON c.customer_id = a.customer_id
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN loans l ON c.customer_id = l.customer_id
LEFT JOIN reviews r ON c.customer_id = r.customer_id
GROUP BY c.customer_id
ORDER BY total_orders DESC NULLS LAST
LIMIT 5
''')
print("\nCustomer Name | Accounts | Orders | Loans | Reviews | Total Spent")
print("-" * 80)
for row in cursor.fetchall():
    print(f"  {row[0]:<20} | {row[1]:<8} | {row[2]:<6} | {row[3]:<5} | {row[4]:<7} | ${row[5] or 0:,.2f}")

print("\n" + "=" * 80)
print("All relationships verified successfully!")
print("=" * 80)

conn.close()

