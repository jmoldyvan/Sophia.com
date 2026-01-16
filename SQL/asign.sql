-- ============================================================================
-- Electronics Retail Company - SQL Query Assignment
-- Purpose: Analyze inventory, sales, and supplier information
-- ============================================================================

-- ============================================================================
-- PART 1: DATA RETRIEVAL FROM A SINGLE TABLE
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 1: List all products with key inventory details
-- Purpose: Provides a complete overview of product catalog with pricing and stock
-- ----------------------------------------------------------------------------
SELECT 
    product_name,          -- Name of the product
    category_id,           -- Foreign key linking to Categories table
    price,                 -- Unit price of the product
    quantity_on_hand       -- Current stock level (QOH)
FROM 
    Products
ORDER BY 
    product_name;          -- Alphabetical ordering for easier reading


-- ----------------------------------------------------------------------------
-- Query 2: Retrieve all out-of-stock products
-- Purpose: Identifies products needing restocking (zero inventory)
-- Logic: Filters products where quantity equals zero using WHERE clause
-- ----------------------------------------------------------------------------
SELECT 
    product_id,
    product_name,
    category_id,
    price,
    quantity_on_hand
FROM 
    Products
WHERE 
    quantity_on_hand = 0   -- Filter condition for out-of-stock items
ORDER BY 
    product_name;


-- ----------------------------------------------------------------------------
-- Query 3: Display products in the $100 to $500 price range
-- Purpose: Retrieves mid-range products for targeted marketing or reporting
-- Logic: Uses BETWEEN operator to filter prices within specified range (inclusive)
-- ----------------------------------------------------------------------------
SELECT 
    product_id,
    product_name,
    category_id,
    price,
    quantity_on_hand
FROM 
    Products
WHERE 
    price BETWEEN 100 AND 500   -- Inclusive range: $100 <= price <= $500
ORDER BY 
    price ASC;                   -- Sort from lowest to highest price


-- ----------------------------------------------------------------------------
-- Query 4: Show total product count grouped by category
-- Purpose: Summarizes product distribution across categories for inventory analysis
-- Logic: GROUP BY aggregates products by category; COUNT tallies each group
-- ----------------------------------------------------------------------------
SELECT 
    category_id,
    COUNT(*) AS total_products   -- Counts all products in each category
FROM 
    Products
GROUP BY 
    category_id                  -- Groups results by category
ORDER BY 
    total_products DESC;         -- Categories with most products shown first


-- ----------------------------------------------------------------------------
-- Query 5: Display average product price per category
-- Purpose: Analyzes pricing strategy across different product categories
-- Logic: GROUP BY groups products by category; AVG calculates mean price
-- ----------------------------------------------------------------------------
SELECT 
    category_id,
    COUNT(*) AS product_count,           -- Number of products in category
    ROUND(AVG(price), 2) AS avg_price    -- Average price rounded to 2 decimals
FROM 
    Products
GROUP BY 
    category_id
ORDER BY 
    avg_price DESC;                       -- Highest average price first


-- ============================================================================
-- PART 2: MULTI-TABLE QUERIES USING JOINS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 1: Join Products and Suppliers for product sourcing information
-- Purpose: Links products with their supplier contact details
-- Logic: INNER JOIN returns only products that have matching suppliers
-- ----------------------------------------------------------------------------
SELECT 
    p.product_name,
    s.supplier_name,
    s.contact_name,          -- Supplier's primary contact person
    s.contact_email,         -- Email for supplier communication
    s.contact_phone          -- Phone number for urgent orders
FROM 
    Products p
INNER JOIN 
    Suppliers s ON p.supplier_id = s.supplier_id   -- Match on supplier ID
ORDER BY 
    s.supplier_name, 
    p.product_name;


-- ----------------------------------------------------------------------------
-- Query 2: Join Products, Categories, and Orders for sales analysis
-- Purpose: Creates comprehensive sales report with product and category details
-- Logic: Multiple INNER JOINs link three tables through their foreign keys
-- ----------------------------------------------------------------------------
SELECT 
    p.product_name,
    c.category_name,
    o.quantity_sold,         -- Number of units sold in the order
    o.order_date             -- Date when the sale occurred
FROM 
    Products p
INNER JOIN 
    Categories c ON p.category_id = c.category_id    -- Link product to category
INNER JOIN 
    Orders o ON p.product_id = o.product_id          -- Link product to orders
ORDER BY 
    o.order_date DESC,       -- Most recent sales first
    c.category_name,
    p.product_name;


-- ----------------------------------------------------------------------------
-- Query 3: LEFT JOIN to list all suppliers and their products
-- Purpose: Shows all suppliers, including those not currently supplying products
-- Logic: LEFT JOIN preserves all supplier records; unmatched products show NULL
-- ----------------------------------------------------------------------------
SELECT 
    s.supplier_id,
    s.supplier_name,
    s.contact_email,
    p.product_id,
    p.product_name,
    p.price
FROM 
    Suppliers s
LEFT JOIN 
    Products p ON s.supplier_id = p.supplier_id   -- Keep all suppliers
ORDER BY 
    s.supplier_name,
    p.product_name;

-- Note: Suppliers with no products will show NULL in product columns
-- This helps identify inactive suppliers or potential new partnerships


-- ----------------------------------------------------------------------------
-- Query 4: FULL OUTER JOIN to show all products and suppliers
-- Purpose: Complete view of products and suppliers including unmatched records
-- Logic: Simulated using UNION of LEFT JOIN and RIGHT JOIN (for MySQL compatibility)
-- ----------------------------------------------------------------------------

-- Option A: Native FULL OUTER JOIN (PostgreSQL, SQL Server, Oracle)
SELECT 
    p.product_id,
    p.product_name,
    p.price,
    s.supplier_id,
    s.supplier_name,
    s.contact_email
FROM 
    Products p
FULL OUTER JOIN 
    Suppliers s ON p.supplier_id = s.supplier_id
ORDER BY 
    p.product_name,
    s.supplier_name;


-- Option B: Simulated FULL OUTER JOIN using UNION (MySQL compatible)
-- This combines LEFT JOIN and RIGHT JOIN results, removing duplicates
SELECT 
    p.product_id,
    p.product_name,
    p.price,
    s.supplier_id,
    s.supplier_name,
    s.contact_email
FROM 
    Products p
LEFT JOIN 
    Suppliers s ON p.supplier_id = s.supplier_id   -- All products, matched suppliers

UNION   -- Combines results and removes duplicates

SELECT 
    p.product_id,
    p.product_name,
    p.price,
    s.supplier_id,
    s.supplier_name,
    s.contact_email
FROM 
    Products p
RIGHT JOIN 
    Suppliers s ON p.supplier_id = s.supplier_id   -- All suppliers, matched products
ORDER BY 
    product_name,
    supplier_name;

-- Note: Products without suppliers show NULL in supplier columns
-- Suppliers without products show NULL in product columns


-- ----------------------------------------------------------------------------
-- Query 5: Find categories with more than 10 products in stock
-- Purpose: Identifies well-stocked categories for inventory management
-- Logic: GROUP BY aggregates by category; HAVING filters aggregated results
-- ----------------------------------------------------------------------------
SELECT 
    c.category_id,
    c.category_name,
    COUNT(p.product_id) AS product_count,           -- Total products in category
    SUM(p.quantity_on_hand) AS total_stock          -- Combined inventory
FROM 
    Categories c
INNER JOIN 
    Products p ON c.category_id = p.category_id
GROUP BY 
    c.category_id,
    c.category_name
HAVING 
    SUM(p.quantity_on_hand) > 10   -- Filter: only categories with >10 items in stock
ORDER BY 
    total_stock DESC;              -- Highest stock levels first

-- Note: HAVING vs WHERE distinction:
-- WHERE filters individual rows BEFORE grouping
-- HAVING filters grouped results AFTER aggregation
-- We use HAVING here because we're filtering on an aggregate function (SUM)


-- ============================================================================
-- END OF QUERIES
-- ============================================================================