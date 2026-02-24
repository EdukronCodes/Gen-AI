-- Project 1: Agentic AI Retail Operations Assistant
-- Sample dataset (schema + seed data) for demos, prototyping, and SQL-tool agents.
-- Designed to work in SQLite; easy to port to Postgres/MS SQL.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS supplier_scorecards;
DROP TABLE IF EXISTS sales_daily;
DROP TABLE IF EXISTS order_lines;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS stores;

-- ===== Dimensions =====

CREATE TABLE stores (
  store_id TEXT PRIMARY KEY,
  store_name TEXT NOT NULL,
  region TEXT NOT NULL,
  city TEXT NOT NULL,
  store_type TEXT NOT NULL, -- e.g. Flagship/Standard/Express
  opened_date TEXT NOT NULL
);

CREATE TABLE products (
  sku TEXT PRIMARY KEY,
  product_name TEXT NOT NULL,
  category TEXT NOT NULL,
  brand TEXT NOT NULL,
  uom TEXT NOT NULL, -- unit of measure
  unit_cost REAL NOT NULL,
  list_price REAL NOT NULL,
  active INTEGER NOT NULL CHECK (active IN (0, 1))
);

CREATE TABLE suppliers (
  supplier_id TEXT PRIMARY KEY,
  supplier_name TEXT NOT NULL,
  country TEXT NOT NULL,
  default_lead_time_days INTEGER NOT NULL,
  payment_terms TEXT NOT NULL
);

-- ===== Facts =====

CREATE TABLE inventory (
  store_id TEXT NOT NULL,
  sku TEXT NOT NULL,
  on_hand_units INTEGER NOT NULL,
  reserved_units INTEGER NOT NULL,
  reorder_point_units INTEGER NOT NULL,
  safety_stock_units INTEGER NOT NULL,
  last_updated_ts TEXT NOT NULL,
  PRIMARY KEY (store_id, sku),
  FOREIGN KEY (store_id) REFERENCES stores(store_id),
  FOREIGN KEY (sku) REFERENCES products(sku)
);

CREATE TABLE orders (
  order_id TEXT PRIMARY KEY,
  store_id TEXT NOT NULL,
  order_ts TEXT NOT NULL,
  order_status TEXT NOT NULL, -- CREATED/PICKED/SHIPPED/DELIVERED/CANCELLED/RETURNED
  channel TEXT NOT NULL,      -- POS/ONLINE/B2B
  customer_type TEXT NOT NULL, -- WALKIN/LOYALTY/CORPORATE
  FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

CREATE TABLE order_lines (
  order_id TEXT NOT NULL,
  line_no INTEGER NOT NULL,
  sku TEXT NOT NULL,
  quantity INTEGER NOT NULL,
  unit_price REAL NOT NULL,
  discount_pct REAL NOT NULL,
  PRIMARY KEY (order_id, line_no),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (sku) REFERENCES products(sku)
);

CREATE TABLE sales_daily (
  sales_date TEXT NOT NULL, -- YYYY-MM-DD
  store_id TEXT NOT NULL,
  sku TEXT NOT NULL,
  units_sold INTEGER NOT NULL,
  gross_sales REAL NOT NULL,
  net_sales REAL NOT NULL,
  PRIMARY KEY (sales_date, store_id, sku),
  FOREIGN KEY (store_id) REFERENCES stores(store_id),
  FOREIGN KEY (sku) REFERENCES products(sku)
);

CREATE TABLE supplier_scorecards (
  supplier_id TEXT NOT NULL,
  scorecard_month TEXT NOT NULL, -- YYYY-MM
  otif_pct REAL NOT NULL, -- On-Time-In-Full
  avg_lead_time_days REAL NOT NULL,
  defect_rate_pct REAL NOT NULL,
  fill_rate_pct REAL NOT NULL,
  notes TEXT,
  PRIMARY KEY (supplier_id, scorecard_month),
  FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

-- ===== Seed data =====

INSERT INTO stores (store_id, store_name, region, city, store_type, opened_date) VALUES
('S001', 'Downtown Flagship', 'North', 'Delhi', 'Flagship', '2019-06-12'),
('S002', 'Mall Central',       'North', 'Noida', 'Standard', '2020-11-03'),
('S003', 'Tech Park Express',   'West',  'Pune',  'Express',  '2021-08-19'),
('S004', 'Coastal Standard',    'South', 'Chennai','Standard','2018-02-26'),
('S005', 'Metro Express',       'East',  'Kolkata','Express', '2022-04-10');

INSERT INTO products (sku, product_name, category, brand, uom, unit_cost, list_price, active) VALUES
('SKU-1001', 'Running Shoes - Sprint',        'Footwear', 'AeroFit',  'pair', 1600, 2999, 1),
('SKU-1002', 'Casual Sneakers - Street',      'Footwear', 'UrbanWalk','pair', 1400, 2599, 1),
('SKU-2001', 'T-Shirt - Classic Cotton',      'Apparel',  'BasicCo',  'each',  220,  499, 1),
('SKU-2002', 'Jeans - Slim Fit',              'Apparel',  'DenimLab', 'each',  900, 1899, 1),
('SKU-3001', 'Smartwatch - Pulse',            'Electronics','Nova',   'each', 4200, 6999, 1),
('SKU-3002', 'Wireless Earbuds - AirLite',    'Electronics','Nova',   'each', 1800, 3499, 1),
('SKU-4001', 'Shampoo - Herbal 340ml',        'FMCG',     'Herbi',    'bottle',110,  249, 1),
('SKU-4002', 'Toothpaste - Mint 150g',        'FMCG',     'SmilePro', 'tube',   55,  129, 1),
('SKU-5001', 'Coffee Beans - Dark Roast 1kg', 'Grocery',  'Roastly',  'bag',   520,  899, 1),
('SKU-5002', 'Olive Oil - Extra Virgin 1L',   'Grocery',  'Meditra',  'bottle',650, 1099, 1);

INSERT INTO suppliers (supplier_id, supplier_name, country, default_lead_time_days, payment_terms) VALUES
('SUP-01', 'Northstar Footwear Pvt Ltd', 'IN', 10, 'Net 30'),
('SUP-02', 'DenimLab Manufacturing',    'IN', 14, 'Net 45'),
('SUP-03', 'Nova Electronics',          'CN', 21, 'Net 60'),
('SUP-04', 'Herbi FMCG Distributors',   'IN',  7, 'Net 30'),
('SUP-05', 'Meditra Foods',             'ES', 18, 'Net 45');

INSERT INTO inventory (store_id, sku, on_hand_units, reserved_units, reorder_point_units, safety_stock_units, last_updated_ts) VALUES
('S001','SKU-1001', 18,  3, 12,  8, '2026-02-24T09:10:00'),
('S001','SKU-1002',  6,  1, 10,  7, '2026-02-24T09:10:00'),
('S001','SKU-3001',  4,  2,  6,  4, '2026-02-24T09:10:00'),
('S001','SKU-3002', 11,  2, 10,  6, '2026-02-24T09:10:00'),
('S001','SKU-4001', 42,  6, 30, 20, '2026-02-24T09:10:00'),
('S001','SKU-5002',  8,  1, 12,  8, '2026-02-24T09:10:00'),

('S002','SKU-1001',  9,  2, 12,  8, '2026-02-24T09:05:00'),
('S002','SKU-2001', 55,  8, 40, 25, '2026-02-24T09:05:00'),
('S002','SKU-2002', 13,  4, 15, 10, '2026-02-24T09:05:00'),
('S002','SKU-3002',  7,  3, 10,  6, '2026-02-24T09:05:00'),
('S002','SKU-5001',  5,  0, 10,  6, '2026-02-24T09:05:00'),

('S003','SKU-1002',  3,  1, 10,  7, '2026-02-24T08:55:00'),
('S003','SKU-3001',  2,  1,  6,  4, '2026-02-24T08:55:00'),
('S003','SKU-4002', 38,  5, 25, 15, '2026-02-24T08:55:00'),
('S003','SKU-5001', 12,  2, 10,  6, '2026-02-24T08:55:00'),

('S004','SKU-2001', 31,  7, 40, 25, '2026-02-24T09:00:00'),
('S004','SKU-2002',  6,  2, 15, 10, '2026-02-24T09:00:00'),
('S004','SKU-4001', 20,  3, 30, 20, '2026-02-24T09:00:00'),
('S004','SKU-5002',  4,  0, 12,  8, '2026-02-24T09:00:00'),

('S005','SKU-1001',  5,  2, 12,  8, '2026-02-24T08:50:00'),
('S005','SKU-3002',  3,  2, 10,  6, '2026-02-24T08:50:00'),
('S005','SKU-4002', 14,  1, 25, 15, '2026-02-24T08:50:00'),
('S005','SKU-5001',  2,  0, 10,  6, '2026-02-24T08:50:00');

INSERT INTO orders (order_id, store_id, order_ts, order_status, channel, customer_type) VALUES
('O-90001','S001','2026-02-20T11:14:00','DELIVERED','ONLINE','LOYALTY'),
('O-90002','S001','2026-02-21T18:06:00','DELIVERED','POS','WALKIN'),
('O-90003','S002','2026-02-21T12:22:00','DELIVERED','POS','LOYALTY'),
('O-90004','S003','2026-02-22T09:40:00','DELIVERED','ONLINE','LOYALTY'),
('O-90005','S004','2026-02-22T16:18:00','CANCELLED','ONLINE','WALKIN'),
('O-90006','S005','2026-02-23T14:02:00','DELIVERED','POS','WALKIN'),
('O-90007','S002','2026-02-23T19:55:00','DELIVERED','ONLINE','LOYALTY'),
('O-90008','S001','2026-02-24T08:10:00','CREATED','ONLINE','LOYALTY');

INSERT INTO order_lines (order_id, line_no, sku, quantity, unit_price, discount_pct) VALUES
('O-90001',1,'SKU-3002',1,3499,10),
('O-90001',2,'SKU-4001',2, 249, 0),
('O-90002',1,'SKU-1002',1,2599, 5),
('O-90003',1,'SKU-2001',3, 499,15),
('O-90003',2,'SKU-2002',1,1899,10),
('O-90004',1,'SKU-3001',1,6999, 8),
('O-90005',1,'SKU-5002',1,1099, 0),
('O-90006',1,'SKU-5001',2, 899, 0),
('O-90006',2,'SKU-4002',1, 129, 0),
('O-90007',1,'SKU-1001',1,2999, 0),
('O-90007',2,'SKU-4001',1, 249, 0),
('O-90008',1,'SKU-1002',1,2599, 0),
('O-90008',2,'SKU-5001',1, 899, 0);

INSERT INTO sales_daily (sales_date, store_id, sku, units_sold, gross_sales, net_sales) VALUES
('2026-02-20','S001','SKU-3002',1,3499,3149.10),
('2026-02-20','S001','SKU-4001',2, 498, 498.00),
('2026-02-21','S001','SKU-1002',1,2599,2469.05),
('2026-02-21','S002','SKU-2001',3,1497,1272.45),
('2026-02-21','S002','SKU-2002',1,1899,1709.10),
('2026-02-22','S003','SKU-3001',1,6999,6439.08),
('2026-02-23','S005','SKU-5001',2,1798,1798.00),
('2026-02-23','S005','SKU-4002',1, 129, 129.00),
('2026-02-23','S002','SKU-1001',1,2999,2999.00),
('2026-02-23','S002','SKU-4001',1, 249, 249.00);

INSERT INTO supplier_scorecards (supplier_id, scorecard_month, otif_pct, avg_lead_time_days, defect_rate_pct, fill_rate_pct, notes) VALUES
('SUP-01','2026-01',93.2,10.8,1.4,96.1,'Slight delays on weekend dispatches'),
('SUP-02','2026-01',89.5,15.6,2.2,92.7,'Backlog due to capacity constraints'),
('SUP-03','2026-01',86.9,23.4,1.8,90.4,'Port congestion impacted lead times'),
('SUP-04','2026-01',95.7, 7.4,0.9,97.3,'Strong performance; stable replenishment'),
('SUP-05','2026-01',91.1,18.9,1.1,94.0,'One late delivery; otherwise consistent');

