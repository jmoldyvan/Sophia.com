-- ============================================================================
-- Electronics Retail Database: Advanced SQL & Schema Objects
-- ============================================================================

-- ============================================================================
-- PART 1: ADVANCED SQL FUNCTIONS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Total Sales by Product Category (Aggregate Function)
-- ----------------------------------------------------------------------------
-- Uses SUM() and GROUP BY to calculate total revenue per category

SELECT 
    c.CategoryName,
    COUNT(DISTINCT od.OrderID) AS TotalOrders,
    SUM(od.Quantity) AS TotalUnitsSold,
    SUM(od.Quantity * od.UnitPrice) AS TotalSales
FROM Categories c
JOIN Products p ON c.CategoryID = p.CategoryID
JOIN OrderDetails od ON p.ProductID = od.ProductID
GROUP BY c.CategoryID, c.CategoryName
ORDER BY TotalSales DESC;


-- ----------------------------------------------------------------------------
-- 2. Format Product Names and Supplier Contact Information (String Functions)
-- ----------------------------------------------------------------------------
-- Uses CONCAT, UPPER, SUBSTRING, and TRIM to format output

SELECT 
    UPPER(p.ProductName) AS ProductNameFormatted,
    CONCAT(
        TRIM(s.CompanyName), 
        ' | Contact: ', 
        COALESCE(CONCAT(s.ContactFirstName, ' ', s.ContactLastName), 'N/A'),
        ' | Phone: ',
        COALESCE(
            CONCAT('(', SUBSTRING(s.Phone, 1, 3), ') ', 
                   SUBSTRING(s.Phone, 4, 3), '-', 
                   SUBSTRING(s.Phone, 7, 4)),
            'N/A'
        )
    ) AS SupplierContactInfo,
    CONCAT('SKU-', LPAD(p.ProductID, 5, '0')) AS FormattedSKU
FROM Products p
JOIN Suppliers s ON p.SupplierID = s.SupplierID
ORDER BY p.ProductName;


-- ----------------------------------------------------------------------------
-- 3. Orders by Purchase Month in Descending Order (Date/Time Functions)
-- ----------------------------------------------------------------------------
-- Uses YEAR(), MONTH(), MONTHNAME(), and DATE_FORMAT for temporal analysis

SELECT 
    YEAR(o.OrderDate) AS OrderYear,
    MONTH(o.OrderDate) AS OrderMonth,
    MONTHNAME(o.OrderDate) AS MonthName,
    COUNT(o.OrderID) AS TotalOrders,
    SUM(od.Quantity * od.UnitPrice) AS MonthlyRevenue,
    DATE_FORMAT(MIN(o.OrderDate), '%Y-%m-%d') AS FirstOrderDate,
    DATE_FORMAT(MAX(o.OrderDate), '%Y-%m-%d') AS LastOrderDate
FROM Orders o
JOIN OrderDetails od ON o.OrderID = od.OrderID
GROUP BY YEAR(o.OrderDate), MONTH(o.OrderDate), MONTHNAME(o.OrderDate)
ORDER BY OrderYear DESC, OrderMonth DESC;


-- ----------------------------------------------------------------------------
-- 4. Calculate 20% Discounted Price for Most Expensive Product
-- ----------------------------------------------------------------------------
-- Uses subquery with MAX() and arithmetic calculation

SELECT 
    ProductID,
    ProductName,
    UnitPrice AS OriginalPrice,
    ROUND(UnitPrice * 0.80, 2) AS DiscountedPrice,
    ROUND(UnitPrice * 0.20, 2) AS SavingsAmount,
    '20%' AS DiscountPercentage
FROM Products
WHERE UnitPrice = (SELECT MAX(UnitPrice) FROM Products);

-- Alternative using LIMIT (if multiple products have same max price, shows only one)
SELECT 
    ProductID,
    ProductName,
    UnitPrice AS OriginalPrice,
    ROUND(UnitPrice * 0.80, 2) AS DiscountedPrice,
    ROUND(UnitPrice * 0.20, 2) AS SavingsAmount
FROM Products
ORDER BY UnitPrice DESC
LIMIT 1;


-- ============================================================================
-- PART 2: SCHEMA OBJECTS AND BUSINESS LOGIC
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. View: Top 5 Best-Selling Products
-- ----------------------------------------------------------------------------

DROP VIEW IF EXISTS vw_Top5BestSellingProducts;

CREATE VIEW vw_Top5BestSellingProducts AS
SELECT 
    p.ProductID,
    p.ProductName,
    c.CategoryName,
    SUM(od.Quantity) AS TotalQuantitySold,
    SUM(od.Quantity * od.UnitPrice) AS TotalRevenue,
    COUNT(DISTINCT od.OrderID) AS NumberOfOrders
FROM Products p
JOIN OrderDetails od ON p.ProductID = od.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
GROUP BY p.ProductID, p.ProductName, c.CategoryName
ORDER BY TotalQuantitySold DESC
LIMIT 5;

-- Query the view
SELECT * FROM vw_Top5BestSellingProducts;


-- ----------------------------------------------------------------------------
-- 2. Stored Procedure: Get Product Sales Statistics
-- ----------------------------------------------------------------------------

DROP PROCEDURE IF EXISTS sp_GetProductSalesStats;

DELIMITER //

CREATE PROCEDURE sp_GetProductSalesStats(
    IN p_ProductID INT,
    OUT p_TotalQuantitySold INT,
    OUT p_TotalRevenue DECIMAL(10,2)
)
BEGIN
    -- Initialize output parameters
    SET p_TotalQuantitySold = 0;
    SET p_TotalRevenue = 0.00;
    
    -- Check if product exists
    IF NOT EXISTS (SELECT 1 FROM Products WHERE ProductID = p_ProductID) THEN
        SIGNAL SQLSTATE '45000' 
        SET MESSAGE_TEXT = 'Product ID not found';
    ELSE
        -- Calculate total quantity sold and revenue
        SELECT 
            COALESCE(SUM(od.Quantity), 0),
            COALESCE(SUM(od.Quantity * od.UnitPrice), 0.00)
        INTO 
            p_TotalQuantitySold, 
            p_TotalRevenue
        FROM OrderDetails od
        WHERE od.ProductID = p_ProductID;
    END IF;
END //

DELIMITER ;

-- Execute the stored procedure
-- Example: Get sales stats for ProductID = 1
SET @qty_sold = 0;
SET @revenue = 0;

CALL sp_GetProductSalesStats(1, @qty_sold, @revenue);

SELECT 
    @qty_sold AS TotalQuantitySold, 
    @revenue AS TotalRevenue;


-- ----------------------------------------------------------------------------
-- 3. Trigger: Inventory Audit Log + Prevent Negative QOH
-- ----------------------------------------------------------------------------

-- First, create the Inventory Audit table
DROP TABLE IF EXISTS InventoryAudit;

CREATE TABLE InventoryAudit (
    AuditID INT AUTO_INCREMENT PRIMARY KEY,
    ProductID INT NOT NULL,
    ProductName VARCHAR(100),
    OldQOH INT,
    NewQOH INT,
    QuantityChange INT,
    ChangeType VARCHAR(20),
    ChangedBy VARCHAR(100),
    ChangeDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    Notes VARCHAR(255)
);

-- Create trigger to log QOH changes and prevent negative values
DROP TRIGGER IF EXISTS trg_InventoryAuditAndValidation;

DELIMITER //

CREATE TRIGGER trg_InventoryAuditAndValidation
BEFORE UPDATE ON Products
FOR EACH ROW
BEGIN
    -- Rule: Prevent negative QOH values
    IF NEW.QuantityOnHand < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Error: Quantity on Hand cannot be negative';
    END IF;
    
    -- Log the change only if QOH actually changed
    IF OLD.QuantityOnHand != NEW.QuantityOnHand THEN
        INSERT INTO InventoryAudit (
            ProductID,
            ProductName,
            OldQOH,
            NewQOH,
            QuantityChange,
            ChangeType,
            ChangedBy,
            Notes
        )
        VALUES (
            OLD.ProductID,
            OLD.ProductName,
            OLD.QuantityOnHand,
            NEW.QuantityOnHand,
            NEW.QuantityOnHand - OLD.QuantityOnHand,
            CASE 
                WHEN NEW.QuantityOnHand > OLD.QuantityOnHand THEN 'RESTOCK'
                ELSE 'SALE/ADJUSTMENT'
            END,
            CURRENT_USER(),
            CONCAT('QOH changed from ', OLD.QuantityOnHand, ' to ', NEW.QuantityOnHand)
        );
    END IF;
END //

DELIMITER ;

-- Test the trigger (assumes ProductID 1 exists)
-- This should work:
-- UPDATE Products SET QuantityOnHand = QuantityOnHand - 5 WHERE ProductID = 1;

-- This should fail (attempting to set negative QOH):
-- UPDATE Products SET QuantityOnHand = -10 WHERE ProductID = 1;

-- View audit log
SELECT * FROM InventoryAudit ORDER BY ChangeDate DESC;


-- ----------------------------------------------------------------------------
-- 4. Transaction Block: Update Inventory and Insert Sales Record
-- ----------------------------------------------------------------------------

DROP PROCEDURE IF EXISTS sp_ProcessSale;

DELIMITER //

CREATE PROCEDURE sp_ProcessSale(
    IN p_OrderID INT,
    IN p_ProductID INT,
    IN p_CustomerID INT,
    IN p_Quantity INT,
    OUT p_Status VARCHAR(100)
)
BEGIN
    DECLARE v_UnitPrice DECIMAL(10,2);
    DECLARE v_CurrentQOH INT;
    DECLARE v_ProductExists INT DEFAULT 0;
    
    -- Error handler for rollback
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SET p_Status = 'FAILED: Transaction rolled back due to error';
    END;
    
    -- Start transaction
    START TRANSACTION;
    
    -- Check if product exists and get current stock
    SELECT COUNT(*), QuantityOnHand, UnitPrice 
    INTO v_ProductExists, v_CurrentQOH, v_UnitPrice
    FROM Products 
    WHERE ProductID = p_ProductID
    GROUP BY QuantityOnHand, UnitPrice;
    
    -- Validate product exists
    IF v_ProductExists = 0 THEN
        ROLLBACK;
        SET p_Status = 'FAILED: Product does not exist';
    -- Validate sufficient stock
    ELSEIF v_CurrentQOH < p_Quantity THEN
        ROLLBACK;
        SET p_Status = CONCAT('FAILED: Insufficient stock. Available: ', v_CurrentQOH);
    ELSE
        -- Update inventory (reduce QOH)
        UPDATE Products 
        SET QuantityOnHand = QuantityOnHand - p_Quantity
        WHERE ProductID = p_ProductID;
        
        -- Insert order if it doesn't exist
        INSERT IGNORE INTO Orders (OrderID, CustomerID, OrderDate)
        VALUES (p_OrderID, p_CustomerID, NOW());
        
        -- Insert sales record into OrderDetails
        INSERT INTO OrderDetails (OrderID, ProductID, Quantity, UnitPrice)
        VALUES (p_OrderID, p_ProductID, p_Quantity, v_UnitPrice);
        
        -- Commit transaction
        COMMIT;
        SET p_Status = CONCAT('SUCCESS: Sale processed. New QOH: ', v_CurrentQOH - p_Quantity);
    END IF;
END //

DELIMITER ;

-- Execute the transaction procedure
SET @sale_status = '';
CALL sp_ProcessSale(1001, 1, 1, 2, @sale_status);
SELECT @sale_status AS TransactionResult;


-- ----------------------------------------------------------------------------
-- 5. Index on Frequently Queried Column
-- ----------------------------------------------------------------------------

-- Create index on OrderDate in Orders table
CREATE INDEX idx_Orders_OrderDate ON Orders(OrderDate);

-- Create composite index on OrderDetails for common join operations
CREATE INDEX idx_OrderDetails_ProductID_OrderID ON OrderDetails(ProductID, OrderID);

-- Create index on Products CategoryID for category-based queries
CREATE INDEX idx_Products_CategoryID ON Products(CategoryID);

/*
INDEX SELECTION EXPLANATION:
----------------------------------------------------------------------------

1. idx_Orders_OrderDate on Orders(OrderDate):
   - REASON: Date-based queries are extremely common in retail systems
   - USE CASES: 
     * Monthly/quarterly sales reports
     * Date range filtering (e.g., "orders from last 30 days")
     * Time-series analysis for business intelligence
   - BENEFIT: Dramatically speeds up queries with WHERE, ORDER BY, or 
     GROUP BY clauses involving OrderDate

2. idx_OrderDetails_ProductID_OrderID on OrderDetails(ProductID, OrderID):
   - REASON: This is a junction table frequently joined with both 
     Products and Orders tables
   - USE CASES:
     * Product sales lookups (which orders contain this product?)
     * Revenue calculations per product
     * Inventory and sales correlation
   - BENEFIT: Composite index covers the two most common join conditions,
     avoiding full table scans on what could be a very large table

3. idx_Products_CategoryID on Products(CategoryID):
   - REASON: Category-based filtering is common in e-commerce
   - USE CASES:
     * "Show all products in Electronics category"
     * Category sales reports
     * Inventory counts by category
   - BENEFIT: Fast category lookups without scanning entire Products table

GENERAL PRINCIPLES APPLIED:
- Indexed columns used in WHERE, JOIN, ORDER BY, and GROUP BY clauses
- Chose columns with high selectivity (many distinct values)
- Avoided over-indexing to maintain INSERT/UPDATE performance
- Prioritized read-heavy operations typical in retail reporting

*/

-- Verify indexes were created
SHOW INDEX FROM Orders;
SHOW INDEX FROM OrderDetails;
SHOW INDEX FROM Products;