-- Project 3: Retail Chatbot for Inventory & Order Queries (Generative AI + RAG)
-- Sample dataset (schema + seed data) for SQL-grounded answers.
-- SQLite-compatible.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS offers;
DROP TABLE IF EXISTS pricing;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS inventory_snapshot;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS stores;

CREATE TABLE stores (
  store_id TEXT PRIMARY KEY,
  store_name TEXT NOT NULL,
  region TEXT NOT NULL,
  city TEXT NOT NULL
);

CREATE TABLE products (
  sku TEXT PRIMARY KEY,
  product_name TEXT NOT NULL,
  category TEXT NOT NULL,
  brand TEXT NOT NULL,
  attributes_json TEXT -- flexible product attributes for RAG-like queries
);

CREATE TABLE suppliers (
  supplier_id TEXT PRIMARY KEY,
  supplier_name TEXT NOT NULL,
  support_email TEXT NOT NULL,
  lead_time_days INTEGER NOT NULL
);

CREATE TABLE inventory_snapshot (
  snapshot_ts TEXT NOT NULL,
  store_id TEXT NOT NULL,
  sku TEXT NOT NULL,
  on_hand_units INTEGER NOT NULL,
  reserved_units INTEGER NOT NULL,
  PRIMARY KEY (snapshot_ts, store_id, sku),
  FOREIGN KEY (store_id) REFERENCES stores(store_id),
  FOREIGN KEY (sku) REFERENCES products(sku)
);

CREATE TABLE orders (
  order_id TEXT PRIMARY KEY,
  order_ts TEXT NOT NULL,
  store_id TEXT NOT NULL,
  channel TEXT NOT NULL, -- ONLINE/POS
  status TEXT NOT NULL   -- CREATED/PICKED/SHIPPED/DELIVERED/CANCELLED/RETURNED
);

CREATE TABLE order_items (
  order_id TEXT NOT NULL,
  line_no INTEGER NOT NULL,
  sku TEXT NOT NULL,
  qty INTEGER NOT NULL,
  unit_price REAL NOT NULL,
  discount_pct REAL NOT NULL,
  PRIMARY KEY (order_id, line_no),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (sku) REFERENCES products(sku)
);

CREATE TABLE pricing (
  sku TEXT PRIMARY KEY,
  currency TEXT NOT NULL,
  list_price REAL NOT NULL,
  effective_from TEXT NOT NULL,
  effective_to TEXT
);

CREATE TABLE offers (
  offer_id TEXT PRIMARY KEY,
  offer_name TEXT NOT NULL,
  sku TEXT,
  category TEXT,
  discount_pct REAL NOT NULL,
  start_date TEXT NOT NULL,
  end_date TEXT NOT NULL,
  channel TEXT NOT NULL, -- ONLINE/POS/ALL
  terms TEXT NOT NULL
);

-- ===== Seed data =====

INSERT INTO stores (store_id, store_name, region, city) VALUES
('S001','Downtown Flagship','North','Delhi'),
('S002','Mall Central','North','Noida'),
('S003','Tech Park Express','West','Pune');

INSERT INTO products (sku, product_name, category, brand, attributes_json) VALUES
('SKU-1001','Running Shoes - Sprint','Footwear','AeroFit','{"color":["black","blue"],"sizes":[6,7,8,9,10],"material":"mesh"}'),
('SKU-1002','Casual Sneakers - Street','Footwear','UrbanWalk','{"color":["white","grey"],"sizes":[6,7,8,9],"material":"canvas"}'),
('SKU-2001','T-Shirt - Classic Cotton','Apparel','BasicCo','{"fit":"regular","fabric":"cotton","colors":["navy","white","black"]}'),
('SKU-2002','Jeans - Slim Fit','Apparel','DenimLab','{"fit":"slim","fabric":"denim","wash":["dark","mid"]}'),
('SKU-3001','Smartwatch - Pulse','Electronics','Nova','{"battery_days":7,"water_resistance":"5ATM","sensors":["HR","SpO2"]}'),
('SKU-3002','Wireless Earbuds - AirLite','Electronics','Nova','{"battery_hours":24,"noise_cancellation":true,"color":["black"]}'),
('SKU-4001','Shampoo - Herbal 340ml','FMCG','Herbi','{"type":"herbal","size_ml":340}'),
('SKU-5001','Coffee Beans - Dark Roast 1kg','Grocery','Roastly','{"roast":"dark","origin":"blend","size_g":1000}');

INSERT INTO suppliers (supplier_id, supplier_name, support_email, lead_time_days) VALUES
('SUP-03','Nova Electronics','support@nova.example',21),
('SUP-01','Northstar Footwear Pvt Ltd','help@northstar.example',10),
('SUP-04','Herbi FMCG Distributors','care@herbi.example',7),
('SUP-05','Roastly Foods','support@roastly.example',12);

INSERT INTO pricing (sku, currency, list_price, effective_from, effective_to) VALUES
('SKU-1001','INR',2999,'2026-01-01',NULL),
('SKU-1002','INR',2599,'2026-01-01',NULL),
('SKU-2001','INR', 499,'2026-01-01',NULL),
('SKU-2002','INR',1899,'2026-01-01',NULL),
('SKU-3001','INR',6999,'2026-01-01',NULL),
('SKU-3002','INR',3499,'2026-01-01',NULL),
('SKU-4001','INR', 249,'2026-01-01',NULL),
('SKU-5001','INR', 899,'2026-01-01',NULL);

INSERT INTO offers (offer_id, offer_name, sku, category, discount_pct, start_date, end_date, channel, terms) VALUES
('OFF-001','Weekend Electronics Deal',NULL,'Electronics',10,'2026-02-21','2026-02-24','ONLINE','Valid on Electronics category; max 1 unit per SKU per order.'),
('OFF-002','Footwear Flash Sale','SKU-1002',NULL,8,'2026-02-20','2026-02-26','ALL','Valid on SKU-1002 only; not combinable with other coupons.'),
('OFF-003','Essentials Combo',NULL,'FMCG',5,'2026-02-01','2026-03-01','POS','Applies to FMCG items in-store only.');

INSERT INTO inventory_snapshot (snapshot_ts, store_id, sku, on_hand_units, reserved_units) VALUES
('2026-02-24T09:00:00','S001','SKU-3002',11,2),
('2026-02-24T09:00:00','S001','SKU-1002', 6,1),
('2026-02-24T09:00:00','S001','SKU-2001',22,4),
('2026-02-24T09:00:00','S001','SKU-4001',40,6),
('2026-02-24T09:00:00','S002','SKU-3002', 7,3),
('2026-02-24T09:00:00','S002','SKU-1001', 9,2),
('2026-02-24T09:00:00','S002','SKU-2002',13,4),
('2026-02-24T09:00:00','S003','SKU-3001', 2,1),
('2026-02-24T09:00:00','S003','SKU-5001',12,2),
('2026-02-24T09:00:00','S003','SKU-1002', 3,1);

INSERT INTO orders (order_id, order_ts, store_id, channel, status) VALUES
('O-70001','2026-02-22T11:15:00','S001','ONLINE','DELIVERED'),
('O-70002','2026-02-23T14:05:00','S002','POS','DELIVERED'),
('O-70003','2026-02-24T08:12:00','S001','ONLINE','CREATED'),
('O-70004','2026-02-24T08:40:00','S003','ONLINE','SHIPPED');

INSERT INTO order_items (order_id, line_no, sku, qty, unit_price, discount_pct) VALUES
('O-70001',1,'SKU-3002',1,3499,10),
('O-70001',2,'SKU-2001',2, 499,15),
('O-70002',1,'SKU-1001',1,2999, 0),
('O-70002',2,'SKU-4001',1, 249, 5),
('O-70003',1,'SKU-1002',1,2599, 8),
('O-70003',2,'SKU-5001',1, 899, 0),
('O-70004',1,'SKU-3001',1,6999,10);

