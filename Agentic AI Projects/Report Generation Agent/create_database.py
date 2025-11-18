import sqlite3
import random
from datetime import datetime, timedelta

# Create database connection
conn = sqlite3.connect('retail_banking.db')
cursor = conn.cursor()

# Enable foreign key constraints
cursor.execute("PRAGMA foreign_keys = ON")

# Drop tables if they exist (for clean setup)
tables = [
    'returns', 'shipments', 'promotions', 'reviews', 'addresses',
    'credit_cards', 'payments', 'loans', 'branches', 'departments',
    'employees', 'inventory', 'suppliers', 'categories', 'order_items',
    'orders', 'transactions', 'products', 'accounts', 'customers'
]

for table in tables:
    cursor.execute(f"DROP TABLE IF EXISTS {table}")

# Create all 20 tables with proper relationships

# 1. Customers table
cursor.execute('''
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    date_of_birth DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# 2. Addresses table
cursor.execute('''
CREATE TABLE addresses (
    address_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    street_address TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    zip_code TEXT NOT NULL,
    country TEXT DEFAULT 'USA',
    address_type TEXT DEFAULT 'HOME',
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
)
''')

# 3. Branches table
cursor.execute('''
CREATE TABLE branches (
    branch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_name TEXT NOT NULL,
    branch_code TEXT UNIQUE NOT NULL,
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    phone TEXT,
    manager_name TEXT
)
''')

# 4. Departments table
cursor.execute('''
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY AUTOINCREMENT,
    department_name TEXT NOT NULL,
    branch_id INTEGER NOT NULL,
    budget DECIMAL(10, 2),
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id) ON DELETE CASCADE
)
''')

# 5. Employees table
cursor.execute('''
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    hire_date DATE NOT NULL,
    salary DECIMAL(10, 2),
    department_id INTEGER NOT NULL,
    manager_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(department_id) ON DELETE CASCADE,
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
)
''')

# 6. Accounts table
cursor.execute('''
CREATE TABLE accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    account_number TEXT UNIQUE NOT NULL,
    account_type TEXT NOT NULL,
    balance DECIMAL(10, 2) DEFAULT 0.00,
    branch_id INTEGER NOT NULL,
    opened_date DATE NOT NULL,
    status TEXT DEFAULT 'ACTIVE',
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
)
''')

# 7. Credit Cards table
cursor.execute('''
CREATE TABLE credit_cards (
    card_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    account_id INTEGER NOT NULL,
    card_number TEXT UNIQUE NOT NULL,
    card_type TEXT NOT NULL,
    expiry_date DATE NOT NULL,
    credit_limit DECIMAL(10, 2),
    current_balance DECIMAL(10, 2) DEFAULT 0.00,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE
)
''')

# 8. Transactions table
cursor.execute('''
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    employee_id INTEGER,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
)
''')

# 9. Loans table
cursor.execute('''
CREATE TABLE loans (
    loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    account_id INTEGER NOT NULL,
    loan_type TEXT NOT NULL,
    principal_amount DECIMAL(10, 2) NOT NULL,
    interest_rate DECIMAL(5, 2) NOT NULL,
    loan_date DATE NOT NULL,
    maturity_date DATE NOT NULL,
    status TEXT DEFAULT 'ACTIVE',
    employee_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
)
''')

# 10. Payments table
cursor.execute('''
CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id INTEGER NOT NULL,
    payment_amount DECIMAL(10, 2) NOT NULL,
    payment_date DATE NOT NULL,
    payment_type TEXT DEFAULT 'PRINCIPAL_AND_INTEREST',
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
)
''')

# 11. Categories table
cursor.execute('''
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL,
    description TEXT,
    parent_category_id INTEGER,
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id)
)
''')

# 12. Suppliers table
cursor.execute('''
CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_name TEXT NOT NULL,
    contact_name TEXT,
    email TEXT,
    phone TEXT,
    address TEXT
)
''')

# 13. Products table
cursor.execute('''
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    supplier_id INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    cost DECIMAL(10, 2),
    description TEXT,
    sku TEXT UNIQUE NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE CASCADE,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id) ON DELETE CASCADE
)
''')

# 14. Inventory table
cursor.execute('''
CREATE TABLE inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    branch_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0,
    reorder_level INTEGER DEFAULT 10,
    last_restocked DATE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id) ON DELETE CASCADE,
    UNIQUE(product_id, branch_id)
)
''')

# 15. Orders table
cursor.execute('''
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2) NOT NULL,
    status TEXT DEFAULT 'PENDING',
    branch_id INTEGER NOT NULL,
    employee_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
)
''')

# 16. Order Items table
cursor.execute('''
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
)
''')

# 17. Shipments table
cursor.execute('''
CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    shipment_date DATE NOT NULL,
    tracking_number TEXT UNIQUE,
    carrier TEXT,
    status TEXT DEFAULT 'IN_TRANSIT',
    estimated_delivery DATE,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
)
''')

# 18. Returns table
cursor.execute('''
CREATE TABLE returns (
    return_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    return_date DATE NOT NULL,
    return_reason TEXT,
    quantity INTEGER NOT NULL,
    refund_amount DECIMAL(10, 2),
    status TEXT DEFAULT 'PENDING',
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
)
''')

# 19. Reviews table
cursor.execute('''
CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    order_id INTEGER,
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    review_text TEXT,
    review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE SET NULL
)
''')

# 20. Promotions table
cursor.execute('''
CREATE TABLE promotions (
    promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    promotion_name TEXT NOT NULL,
    product_id INTEGER,
    category_id INTEGER,
    discount_percentage DECIMAL(5, 2),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status TEXT DEFAULT 'ACTIVE',
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE CASCADE
)
''')

print("All 20 tables created successfully!")

# Insert sample data

# Insert Customers
customers_data = [
    ('John', 'Smith', 'john.smith@email.com', '555-0101', '1985-03-15'),
    ('Sarah', 'Johnson', 'sarah.j@email.com', '555-0102', '1990-07-22'),
    ('Michael', 'Williams', 'm.williams@email.com', '555-0103', '1988-11-08'),
    ('Emily', 'Brown', 'emily.brown@email.com', '555-0104', '1992-05-30'),
    ('David', 'Jones', 'david.jones@email.com', '555-0105', '1987-09-14'),
    ('Jessica', 'Garcia', 'j.garcia@email.com', '555-0106', '1991-12-03'),
    ('Robert', 'Miller', 'r.miller@email.com', '555-0107', '1986-02-18'),
    ('Amanda', 'Davis', 'amanda.d@email.com', '555-0108', '1993-08-25'),
    ('James', 'Rodriguez', 'j.rodriguez@email.com', '555-0109', '1989-04-11'),
    ('Lisa', 'Martinez', 'lisa.m@email.com', '555-0110', '1994-06-19'),
    ('William', 'Hernandez', 'w.hernandez@email.com', '555-0111', '1985-10-27'),
    ('Jennifer', 'Lopez', 'j.lopez@email.com', '555-0112', '1990-01-05'),
    ('Richard', 'Wilson', 'r.wilson@email.com', '555-0113', '1987-07-13'),
    ('Michelle', 'Anderson', 'm.anderson@email.com', '555-0114', '1992-03-21'),
    ('Thomas', 'Taylor', 't.taylor@email.com', '555-0115', '1988-09-29'),
]

cursor.executemany('''
INSERT INTO customers (first_name, last_name, email, phone, date_of_birth)
VALUES (?, ?, ?, ?, ?)
''', customers_data)

# Insert Addresses
addresses_data = [
    (1, '123 Main St', 'New York', 'NY', '10001', 'USA', 'HOME'),
    (1, '456 Business Ave', 'New York', 'NY', '10002', 'USA', 'WORK'),
    (2, '789 Oak Blvd', 'Los Angeles', 'CA', '90001', 'USA', 'HOME'),
    (3, '321 Pine St', 'Chicago', 'IL', '60601', 'USA', 'HOME'),
    (4, '654 Elm Ave', 'Houston', 'TX', '77001', 'USA', 'HOME'),
    (5, '987 Maple Dr', 'Phoenix', 'AZ', '85001', 'USA', 'HOME'),
    (6, '147 Cedar Ln', 'Philadelphia', 'PA', '19101', 'USA', 'HOME'),
    (7, '258 Birch St', 'San Antonio', 'TX', '78201', 'USA', 'HOME'),
    (8, '369 Spruce Ave', 'San Diego', 'CA', '92101', 'USA', 'HOME'),
    (9, '741 Willow Way', 'Dallas', 'TX', '75201', 'USA', 'HOME'),
    (10, '852 Ash Blvd', 'San Jose', 'CA', '95101', 'USA', 'HOME'),
    (11, '963 Poplar St', 'Austin', 'TX', '78701', 'USA', 'HOME'),
    (12, '159 Cherry Dr', 'Jacksonville', 'FL', '32201', 'USA', 'HOME'),
    (13, '357 Walnut Ave', 'Fort Worth', 'TX', '76101', 'USA', 'HOME'),
    (14, '468 Hickory Ln', 'Columbus', 'OH', '43201', 'USA', 'HOME'),
]

cursor.executemany('''
INSERT INTO addresses (customer_id, street_address, city, state, zip_code, country, address_type)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', addresses_data)

# Insert Branches
branches_data = [
    ('Downtown Branch', 'BR001', '100 Financial Plaza', 'New York', '555-1001', 'John Manager'),
    ('Westside Branch', 'BR002', '200 Commerce St', 'Los Angeles', '555-1002', 'Jane Supervisor'),
    ('Central Branch', 'BR003', '300 Business Ave', 'Chicago', '555-1003', 'Bob Director'),
    ('North Branch', 'BR004', '400 Market St', 'Houston', '555-1004', 'Alice Manager'),
    ('South Branch', 'BR005', '500 Trade Blvd', 'Phoenix', '555-1005', 'Charlie Supervisor'),
]

cursor.executemany('''
INSERT INTO branches (branch_name, branch_code, address, city, phone, manager_name)
VALUES (?, ?, ?, ?, ?, ?)
''', branches_data)

# Insert Departments
departments_data = [
    ('Retail Banking', 1, 500000.00),
    ('Commercial Banking', 1, 750000.00),
    ('Retail Banking', 2, 450000.00),
    ('Customer Service', 2, 300000.00),
    ('Retail Banking', 3, 550000.00),
    ('Loan Department', 3, 600000.00),
    ('Retail Banking', 4, 480000.00),
    ('Investment Services', 4, 800000.00),
    ('Retail Banking', 5, 420000.00),
    ('Operations', 5, 350000.00),
]

cursor.executemany('''
INSERT INTO departments (department_name, branch_id, budget)
VALUES (?, ?, ?)
''', departments_data)

# Insert Employees
employees_data = [
    ('Robert', 'Manager', 'r.manager@bank.com', '555-2001', '2020-01-15', 75000.00, 1, None),
    ('Susan', 'Teller', 's.teller@bank.com', '555-2002', '2021-03-20', 45000.00, 1, 1),
    ('Mark', 'Advisor', 'm.advisor@bank.com', '555-2003', '2020-06-10', 65000.00, 1, 1),
    ('Patricia', 'Manager', 'p.manager@bank.com', '555-2004', '2019-02-14', 78000.00, 2, None),
    ('Daniel', 'Teller', 'd.teller@bank.com', '555-2005', '2022-05-01', 46000.00, 2, 4),
    ('Linda', 'Advisor', 'l.advisor@bank.com', '555-2006', '2021-08-22', 67000.00, 2, 4),
    ('Christopher', 'Manager', 'c.manager@bank.com', '555-2007', '2018-11-05', 80000.00, 3, None),
    ('Barbara', 'Teller', 'b.teller@bank.com', '555-2008', '2023-01-10', 47000.00, 3, 7),
    ('Matthew', 'Loan Officer', 'm.loan@bank.com', '555-2009', '2020-04-18', 70000.00, 6, 7),
    ('Nancy', 'Manager', 'n.manager@bank.com', '555-2010', '2019-07-30', 77000.00, 4, None),
    ('Anthony', 'Teller', 'a.teller@bank.com', '555-2011', '2022-09-12', 48000.00, 4, 10),
    ('Karen', 'Investment Advisor', 'k.invest@bank.com', '555-2012', '2021-12-03', 85000.00, 8, 10),
    ('Donald', 'Manager', 'd.manager@bank.com', '555-2013', '2020-03-25', 76000.00, 5, None),
    ('Betty', 'Teller', 'b.teller2@bank.com', '555-2014', '2023-06-08', 49000.00, 5, 13),
    ('Paul', 'Advisor', 'p.advisor@bank.com', '555-2015', '2022-02-14', 68000.00, 5, 13),
]

cursor.executemany('''
INSERT INTO employees (first_name, last_name, email, phone, hire_date, salary, department_id, manager_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', employees_data)

# Insert Accounts
accounts_data = [
    (1, 'ACC001234567', 'CHECKING', 5000.00, 1, '2020-01-15', 'ACTIVE'),
    (1, 'ACC001234568', 'SAVINGS', 15000.00, 1, '2020-01-15', 'ACTIVE'),
    (2, 'ACC001234569', 'CHECKING', 3200.00, 2, '2021-03-10', 'ACTIVE'),
    (2, 'ACC001234570', 'SAVINGS', 8500.00, 2, '2021-03-10', 'ACTIVE'),
    (3, 'ACC001234571', 'CHECKING', 7800.00, 3, '2019-06-20', 'ACTIVE'),
    (3, 'ACC001234572', 'SAVINGS', 22000.00, 3, '2019-06-20', 'ACTIVE'),
    (4, 'ACC001234573', 'CHECKING', 2100.00, 4, '2022-02-05', 'ACTIVE'),
    (4, 'ACC001234574', 'SAVINGS', 5000.00, 4, '2022-02-05', 'ACTIVE'),
    (5, 'ACC001234575', 'CHECKING', 9500.00, 5, '2020-08-12', 'ACTIVE'),
    (5, 'ACC001234576', 'SAVINGS', 18000.00, 5, '2020-08-12', 'ACTIVE'),
    (6, 'ACC001234577', 'CHECKING', 4500.00, 1, '2021-11-18', 'ACTIVE'),
    (7, 'ACC001234578', 'CHECKING', 6200.00, 2, '2020-05-22', 'ACTIVE'),
    (8, 'ACC001234579', 'CHECKING', 3800.00, 3, '2022-07-30', 'ACTIVE'),
    (9, 'ACC001234580', 'CHECKING', 11000.00, 4, '2019-09-14', 'ACTIVE'),
    (10, 'ACC001234581', 'CHECKING', 2900.00, 5, '2021-04-08', 'ACTIVE'),
]

cursor.executemany('''
INSERT INTO accounts (customer_id, account_number, account_type, balance, branch_id, opened_date, status)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', accounts_data)

# Insert Credit Cards
credit_cards_data = [
    (1, 1, '4532-1234-5678-9010', 'VISA', '2027-12-31', 10000.00, 2500.00),
    (1, 2, '4532-1234-5678-9011', 'MASTERCARD', '2028-06-30', 15000.00, 0.00),
    (2, 3, '4532-1234-5678-9012', 'VISA', '2027-09-30', 8000.00, 1200.00),
    (3, 5, '4532-1234-5678-9013', 'AMEX', '2028-03-31', 20000.00, 5000.00),
    (4, 7, '4532-1234-5678-9014', 'VISA', '2027-11-30', 5000.00, 800.00),
    (5, 9, '4532-1234-5678-9015', 'MASTERCARD', '2028-08-31', 12000.00, 3000.00),
    (6, 11, '4532-1234-5678-9016', 'VISA', '2027-10-31', 6000.00, 1500.00),
    (7, 12, '4532-1234-5678-9017', 'VISA', '2028-05-31', 7000.00, 2000.00),
    (8, 13, '4532-1234-5678-9018', 'MASTERCARD', '2027-07-31', 5500.00, 900.00),
    (9, 14, '4532-1234-5678-9019', 'AMEX', '2028-12-31', 25000.00, 8000.00),
]

cursor.executemany('''
INSERT INTO credit_cards (customer_id, account_id, card_number, card_type, expiry_date, credit_limit, current_balance)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', credit_cards_data)

# Insert Transactions
transactions_data = [
    (1, 'DEPOSIT', 1000.00, '2024-01-15 10:30:00', 'Initial deposit', 2),
    (1, 'WITHDRAWAL', -200.00, '2024-01-20 14:15:00', 'ATM withdrawal', None),
    (1, 'DEPOSIT', 500.00, '2024-02-01 09:00:00', 'Salary deposit', 2),
    (2, 'DEPOSIT', 2000.00, '2024-01-10 11:20:00', 'Initial deposit', 2),
    (3, 'DEPOSIT', 1500.00, '2024-01-18 13:45:00', 'Initial deposit', 5),
    (3, 'WITHDRAWAL', -300.00, '2024-02-05 16:30:00', 'Check payment', 5),
    (4, 'DEPOSIT', 800.00, '2024-02-10 10:00:00', 'Initial deposit', 8),
    (5, 'DEPOSIT', 2500.00, '2024-01-25 12:00:00', 'Initial deposit', 11),
    (5, 'WITHDRAWAL', -150.00, '2024-02-12 15:20:00', 'ATM withdrawal', None),
    (6, 'DEPOSIT', 1200.00, '2024-02-08 09:30:00', 'Initial deposit', 2),
    (7, 'DEPOSIT', 1800.00, '2024-01-30 14:00:00', 'Initial deposit', 5),
    (8, 'DEPOSIT', 950.00, '2024-02-15 11:15:00', 'Initial deposit', 8),
    (9, 'DEPOSIT', 3000.00, '2024-01-12 10:45:00', 'Initial deposit', 11),
    (10, 'DEPOSIT', 700.00, '2024-02-20 13:30:00', 'Initial deposit', 14),
    (1, 'TRANSFER', -500.00, '2024-02-18 10:00:00', 'Transfer to savings', 2),
    (2, 'TRANSFER', 500.00, '2024-02-18 10:00:00', 'Transfer from checking', 2),
]

cursor.executemany('''
INSERT INTO transactions (account_id, transaction_type, amount, transaction_date, description, employee_id)
VALUES (?, ?, ?, ?, ?, ?)
''', transactions_data)

# Insert Loans
loans_data = [
    (1, 1, 'MORTGAGE', 250000.00, 4.5, '2020-03-01', '2050-03-01', 'ACTIVE', 9),
    (2, 3, 'AUTO', 35000.00, 5.2, '2021-05-15', '2026-05-15', 'ACTIVE', 9),
    (3, 5, 'PERSONAL', 15000.00, 7.5, '2019-08-10', '2024-08-10', 'ACTIVE', 9),
    (4, 7, 'AUTO', 28000.00, 5.8, '2022-03-20', '2027-03-20', 'ACTIVE', 9),
    (5, 9, 'MORTGAGE', 180000.00, 4.2, '2020-09-05', '2050-09-05', 'ACTIVE', 9),
    (6, 11, 'PERSONAL', 10000.00, 8.0, '2021-12-01', '2024-12-01', 'ACTIVE', 9),
    (7, 12, 'AUTO', 22000.00, 6.0, '2020-06-18', '2025-06-18', 'ACTIVE', 9),
    (8, 13, 'PERSONAL', 8000.00, 7.8, '2022-08-25', '2025-08-25', 'ACTIVE', 9),
]

cursor.executemany('''
INSERT INTO loans (customer_id, account_id, loan_type, principal_amount, interest_rate, loan_date, maturity_date, status, employee_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
''', loans_data)

# Insert Payments
payments_data = [
    (1, 1200.00, '2024-01-01', 'PRINCIPAL_AND_INTEREST'),
    (1, 1200.00, '2024-02-01', 'PRINCIPAL_AND_INTEREST'),
    (2, 650.00, '2024-01-15', 'PRINCIPAL_AND_INTEREST'),
    (2, 650.00, '2024-02-15', 'PRINCIPAL_AND_INTEREST'),
    (3, 350.00, '2024-01-10', 'PRINCIPAL_AND_INTEREST'),
    (3, 350.00, '2024-02-10', 'PRINCIPAL_AND_INTEREST'),
    (4, 550.00, '2024-01-20', 'PRINCIPAL_AND_INTEREST'),
    (4, 550.00, '2024-02-20', 'PRINCIPAL_AND_INTEREST'),
    (5, 900.00, '2024-01-05', 'PRINCIPAL_AND_INTEREST'),
    (5, 900.00, '2024-02-05', 'PRINCIPAL_AND_INTEREST'),
    (6, 300.00, '2024-01-01', 'PRINCIPAL_AND_INTEREST'),
    (6, 300.00, '2024-02-01', 'PRINCIPAL_AND_INTEREST'),
    (7, 450.00, '2024-01-18', 'PRINCIPAL_AND_INTEREST'),
    (7, 450.00, '2024-02-18', 'PRINCIPAL_AND_INTEREST'),
    (8, 250.00, '2024-01-25', 'PRINCIPAL_AND_INTEREST'),
    (8, 250.00, '2024-02-25', 'PRINCIPAL_AND_INTEREST'),
]

cursor.executemany('''
INSERT INTO payments (loan_id, payment_amount, payment_date, payment_type)
VALUES (?, ?, ?, ?)
''', payments_data)

# Insert Categories
categories_data = [
    ('Electronics', 'Electronic devices and accessories', None),
    ('Clothing', 'Apparel and fashion items', None),
    ('Home & Garden', 'Home improvement and garden supplies', None),
    ('Books', 'Books and reading materials', None),
    ('Sports', 'Sports equipment and accessories', None),
    ('Smartphones', 'Mobile phones and smartphones', 1),
    ('Laptops', 'Laptop computers', 1),
    ('Men\'s Clothing', 'Clothing for men', 2),
    ('Women\'s Clothing', 'Clothing for women', 2),
    ('Furniture', 'Home furniture', 3),
]

cursor.executemany('''
INSERT INTO categories (category_name, description, parent_category_id)
VALUES (?, ?, ?)
''', categories_data)

# Insert Suppliers
suppliers_data = [
    ('TechSupply Co', 'John Supplier', 'supply1@tech.com', '555-3001', '100 Supplier St'),
    ('Fashion World', 'Sarah Supplier', 'supply2@fashion.com', '555-3002', '200 Fashion Ave'),
    ('Home Depot Pro', 'Mike Supplier', 'supply3@home.com', '555-3003', '300 Home Blvd'),
    ('Book Distributors', 'Lisa Supplier', 'supply4@book.com', '555-3004', '400 Book Lane'),
    ('Sports Direct', 'Tom Supplier', 'supply5@sports.com', '555-3005', '500 Sports Way'),
]

cursor.executemany('''
INSERT INTO suppliers (supplier_name, contact_name, email, phone, address)
VALUES (?, ?, ?, ?, ?)
''', suppliers_data)

# Insert Products
products_data = [
    ('iPhone 15 Pro', 6, 1, 999.99, 700.00, 'Latest iPhone model', 'SKU-IP15P-001'),
    ('Samsung Galaxy S24', 6, 1, 899.99, 650.00, 'Latest Samsung smartphone', 'SKU-SGS24-001'),
    ('MacBook Pro 16"', 7, 1, 2499.99, 1800.00, 'Apple MacBook Pro', 'SKU-MBP16-001'),
    ('Dell XPS 15', 7, 1, 1799.99, 1300.00, 'Dell XPS laptop', 'SKU-DXPS15-001'),
    ('Men\'s Dress Shirt', 8, 2, 49.99, 25.00, 'Classic dress shirt', 'SKU-MDS-001'),
    ('Women\'s Blazer', 9, 2, 89.99, 45.00, 'Professional blazer', 'SKU-WB-001'),
    ('Leather Sofa', 10, 3, 1299.99, 800.00, '3-seater leather sofa', 'SKU-LS-001'),
    ('Coffee Table', 10, 3, 299.99, 150.00, 'Modern coffee table', 'SKU-CT-001'),
    ('Python Programming Book', 4, 4, 49.99, 20.00, 'Learn Python programming', 'SKU-PPB-001'),
    ('Data Science Handbook', 4, 4, 59.99, 25.00, 'Comprehensive data science guide', 'SKU-DSH-001'),
    ('Basketball', 5, 5, 29.99, 15.00, 'Official size basketball', 'SKU-BB-001'),
    ('Yoga Mat', 5, 5, 24.99, 12.00, 'Premium yoga mat', 'SKU-YM-001'),
    ('Wireless Earbuds', 1, 1, 129.99, 70.00, 'Noise-cancelling earbuds', 'SKU-WE-001'),
    ('Smart Watch', 1, 1, 299.99, 180.00, 'Fitness tracking smartwatch', 'SKU-SW-001'),
    ('Running Shoes', 5, 5, 89.99, 45.00, 'Professional running shoes', 'SKU-RS-001'),
]

cursor.executemany('''
INSERT INTO products (product_name, category_id, supplier_id, price, cost, description, sku)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', products_data)

# Insert Inventory
inventory_data = [
    (1, 1, 50, 10, '2024-01-15'),
    (1, 2, 30, 10, '2024-01-20'),
    (2, 1, 40, 10, '2024-01-18'),
    (2, 3, 25, 10, '2024-01-22'),
    (3, 1, 20, 5, '2024-01-10'),
    (4, 2, 15, 5, '2024-01-12'),
    (5, 1, 100, 20, '2024-01-25'),
    (5, 4, 80, 20, '2024-01-28'),
    (6, 2, 60, 15, '2024-01-30'),
    (7, 3, 10, 3, '2024-02-01'),
    (8, 3, 25, 5, '2024-02-05'),
    (9, 1, 200, 50, '2024-01-15'),
    (10, 2, 150, 40, '2024-01-20'),
    (11, 5, 75, 20, '2024-01-18'),
    (12, 5, 100, 25, '2024-01-22'),
    (13, 1, 80, 20, '2024-02-10'),
    (14, 1, 45, 10, '2024-02-12'),
    (15, 5, 120, 30, '2024-02-15'),
]

cursor.executemany('''
INSERT INTO inventory (product_id, branch_id, quantity, reorder_level, last_restocked)
VALUES (?, ?, ?, ?, ?)
''', inventory_data)

# Insert Orders
orders_data = [
    (1, '2024-01-20 10:30:00', 1049.98, 'COMPLETED', 1, 2),
    (2, '2024-01-25 14:20:00', 899.99, 'COMPLETED', 2, 5),
    (3, '2024-02-01 11:15:00', 2499.99, 'COMPLETED', 1, 2),
    (4, '2024-02-05 09:45:00', 139.98, 'COMPLETED', 4, 11),
    (5, '2024-02-10 16:30:00', 1799.99, 'COMPLETED', 2, 5),
    (6, '2024-02-12 13:20:00', 49.99, 'COMPLETED', 1, 2),
    (7, '2024-02-15 10:00:00', 89.99, 'COMPLETED', 3, 8),
    (8, '2024-02-18 15:45:00', 1299.99, 'PENDING', 3, 8),
    (9, '2024-02-20 12:30:00', 359.98, 'COMPLETED', 4, 11),
    (10, '2024-02-22 14:15:00', 24.99, 'COMPLETED', 5, 14),
    (11, '2024-02-25 11:00:00', 129.99, 'COMPLETED', 1, 2),
    (12, '2024-02-28 09:30:00', 299.99, 'COMPLETED', 2, 5),
    (13, '2024-03-01 13:45:00', 89.99, 'COMPLETED', 5, 14),
    (14, '2024-03-03 10:20:00', 109.98, 'COMPLETED', 1, 2),
    (15, '2024-03-05 15:00:00', 49.99, 'PENDING', 2, 5),
]

cursor.executemany('''
INSERT INTO orders (customer_id, order_date, total_amount, status, branch_id, employee_id)
VALUES (?, ?, ?, ?, ?, ?)
''', orders_data)

# Insert Order Items
order_items_data = [
    (1, 1, 1, 999.99, 999.99),
    (1, 13, 1, 49.99, 49.99),
    (2, 2, 1, 899.99, 899.99),
    (3, 3, 1, 2499.99, 2499.99),
    (4, 5, 1, 49.99, 49.99),
    (4, 6, 1, 89.99, 89.99),
    (5, 4, 1, 1799.99, 1799.99),
    (6, 5, 1, 49.99, 49.99),
    (7, 6, 1, 89.99, 89.99),
    (8, 7, 1, 1299.99, 1299.99),
    (9, 8, 1, 299.99, 299.99),
    (9, 12, 1, 59.99, 59.99),
    (10, 12, 1, 24.99, 24.99),
    (11, 13, 1, 129.99, 129.99),
    (12, 14, 1, 299.99, 299.99),
    (13, 15, 1, 89.99, 89.99),
    (14, 9, 1, 49.99, 49.99),
    (14, 10, 1, 59.99, 59.99),
    (15, 9, 1, 49.99, 49.99),
]

cursor.executemany('''
INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal)
VALUES (?, ?, ?, ?, ?)
''', order_items_data)

# Insert Shipments
shipments_data = [
    (1, '2024-01-21', 'TRK001234567', 'FedEx', 'DELIVERED', '2024-01-23'),
    (2, '2024-01-26', 'TRK001234568', 'UPS', 'DELIVERED', '2024-01-28'),
    (3, '2024-02-02', 'TRK001234569', 'FedEx', 'DELIVERED', '2024-02-05'),
    (4, '2024-02-06', 'TRK001234570', 'USPS', 'DELIVERED', '2024-02-08'),
    (5, '2024-02-11', 'TRK001234571', 'UPS', 'DELIVERED', '2024-02-14'),
    (6, '2024-02-13', 'TRK001234572', 'USPS', 'DELIVERED', '2024-02-15'),
    (7, '2024-02-16', 'TRK001234573', 'FedEx', 'IN_TRANSIT', '2024-02-20'),
    (8, '2024-02-19', 'TRK001234574', 'UPS', 'PENDING', '2024-02-25'),
    (9, '2024-02-21', 'TRK001234575', 'USPS', 'DELIVERED', '2024-02-24'),
    (10, '2024-02-23', 'TRK001234576', 'FedEx', 'DELIVERED', '2024-02-26'),
    (11, '2024-02-26', 'TRK001234577', 'UPS', 'IN_TRANSIT', '2024-03-01'),
    (12, '2024-03-01', 'TRK001234578', 'FedEx', 'DELIVERED', '2024-03-03'),
    (13, '2024-03-02', 'TRK001234579', 'USPS', 'DELIVERED', '2024-03-04'),
    (14, '2024-03-04', 'TRK001234580', 'UPS', 'IN_TRANSIT', '2024-03-07'),
]

cursor.executemany('''
INSERT INTO shipments (order_id, shipment_date, tracking_number, carrier, status, estimated_delivery)
VALUES (?, ?, ?, ?, ?, ?)
''', shipments_data)

# Insert Returns
returns_data = [
    (2, 2, '2024-02-05', 'Defective product', 1, 899.99, 'APPROVED'),
    (4, 5, '2024-02-10', 'Wrong size', 1, 49.99, 'APPROVED'),
    (7, 6, '2024-02-20', 'Changed mind', 1, 89.99, 'PENDING'),
    (11, 13, '2024-03-02', 'Not as described', 1, 129.99, 'APPROVED'),
]

cursor.executemany('''
INSERT INTO returns (order_id, product_id, return_date, return_reason, quantity, refund_amount, status)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', returns_data)

# Insert Reviews
reviews_data = [
    (1, 1, 1, 5, 'Excellent phone, very satisfied!', '2024-01-25 10:00:00'),
    (1, 13, 1, 4, 'Good quality earbuds', '2024-01-25 10:05:00'),
    (2, 2, 2, 4, 'Great smartphone, fast delivery', '2024-01-30 14:00:00'),
    (3, 3, 3, 5, 'Amazing laptop, highly recommend!', '2024-02-06 11:00:00'),
    (4, 5, 4, 3, 'Shirt is okay, but could be better quality', '2024-02-09 09:00:00'),
    (5, 4, 5, 5, 'Perfect laptop for work', '2024-02-15 16:00:00'),
    (6, 5, 6, 4, 'Nice shirt, good value', '2024-02-16 13:00:00'),
    (9, 8, 9, 5, 'Beautiful coffee table', '2024-02-25 12:00:00'),
    (9, 10, 9, 4, 'Very informative book', '2024-02-25 12:05:00'),
    (10, 12, 10, 5, 'Great yoga mat, very comfortable', '2024-02-27 14:00:00'),
    (12, 14, 12, 5, 'Love this smartwatch!', '2024-03-04 10:00:00'),
    (13, 15, 13, 4, 'Comfortable running shoes', '2024-03-05 15:00:00'),
]

cursor.executemany('''
INSERT INTO reviews (customer_id, product_id, order_id, rating, review_text, review_date)
VALUES (?, ?, ?, ?, ?, ?)
''', reviews_data)

# Insert Promotions
promotions_data = [
    ('Spring Sale - Electronics', 1, None, 15.00, '2024-03-01', '2024-03-31', 'ACTIVE'),
    ('Spring Sale - Electronics', 2, None, 15.00, '2024-03-01', '2024-03-31', 'ACTIVE'),
    ('Clothing Clearance', None, 2, 25.00, '2024-02-15', '2024-03-15', 'ACTIVE'),
    ('Book Week Special', None, 4, 20.00, '2024-03-01', '2024-03-07', 'ACTIVE'),
    ('Sports Equipment Sale', None, 5, 30.00, '2024-03-10', '2024-03-20', 'ACTIVE'),
    ('Laptop Special', 3, None, 10.00, '2024-03-05', '2024-03-25', 'ACTIVE'),
    ('Home & Garden Discount', None, 3, 20.00, '2024-02-20', '2024-03-20', 'ACTIVE'),
]

cursor.executemany('''
INSERT INTO promotions (promotion_name, product_id, category_id, discount_percentage, start_date, end_date, status)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', promotions_data)

# Commit all changes
conn.commit()

# Verify data insertion
print("\nData insertion summary:")
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"{table.capitalize()}: {count} rows")

# Test relationships with a sample query
print("\n\nSample relationship query - Customer orders with products:")
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    o.order_id,
    o.order_date,
    p.product_name,
    oi.quantity,
    oi.subtotal
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
ORDER BY o.order_date DESC
LIMIT 10
''')

results = cursor.fetchall()
for row in results:
    print(f"  {row[0]} - Order #{row[1]} ({row[2]}) - {row[3]} x{row[4]} = ${row[5]}")

print("\n\nSample relationship query - Customer accounts with transactions:")
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    a.account_number,
    a.account_type,
    a.balance,
    COUNT(t.transaction_id) AS transaction_count
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
LEFT JOIN transactions t ON a.account_id = t.account_id
GROUP BY c.customer_id, a.account_id
LIMIT 10
''')

results = cursor.fetchall()
for row in results:
    print(f"  {row[0]} - {row[1]} ({row[2]}) - Balance: ${row[3]} - Transactions: {row[4]}")

print("\n\nSample relationship query - Loans with payments:")
cursor.execute('''
SELECT 
    c.first_name || ' ' || c.last_name AS customer_name,
    l.loan_type,
    l.principal_amount,
    l.interest_rate,
    COUNT(p.payment_id) AS payment_count,
    SUM(p.payment_amount) AS total_paid
FROM customers c
JOIN loans l ON c.customer_id = l.customer_id
LEFT JOIN payments p ON l.loan_id = p.loan_id
GROUP BY l.loan_id
LIMIT 10
''')

results = cursor.fetchall()
for row in results:
    print(f"  {row[0]} - {row[1]} - Principal: ${row[2]} - Rate: {row[3]}% - Payments: {row[4]} - Total Paid: ${row[5] or 0}")

print("\n\nDatabase created successfully with all relationships!")
print("Database file: retail_banking.db")

# Close connection
conn.close()

