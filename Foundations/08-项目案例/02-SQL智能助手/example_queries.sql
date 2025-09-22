-- SQL智能助手示例查询
-- 这个文件包含了各种类型的SQL查询示例，帮助理解数据库结构和常见查询模式

-- ================================
-- 基础查询 (Basic Queries)
-- ================================

-- 查询所有产品
SELECT * FROM products LIMIT 10;

-- 查询所有产品的名称和价格
SELECT name, price FROM products;

-- 查询特定分类的产品
SELECT p.name, p.price, c.name as category_name
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE c.name = '电子产品';

-- 查询价格在特定范围内的产品
SELECT name, price FROM products
WHERE price BETWEEN 100 AND 1000
ORDER BY price DESC;

-- ================================
-- 聚合查询 (Aggregate Queries)
-- ================================

-- 统计每个分类的产品数量
SELECT c.name as category_name, COUNT(p.id) as product_count
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
GROUP BY c.id, c.name
ORDER BY product_count DESC;

-- 计算每个分类的平均价格
SELECT c.name as category_name,
       COUNT(p.id) as product_count,
       AVG(p.price) as avg_price,
       MIN(p.price) as min_price,
       MAX(p.price) as max_price
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
GROUP BY c.id, c.name
HAVING COUNT(p.id) > 0
ORDER BY avg_price DESC;

-- 查询总销售额最高的产品
SELECT p.name,
       SUM(oi.quantity) as total_sold,
       SUM(oi.total_price) as total_revenue
FROM products p
JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name
ORDER BY total_revenue DESC
LIMIT 10;

-- ================================
-- 客户分析查询 (Customer Analysis)
-- ================================

-- 查询每个客户的订单数量和总消费金额
SELECT c.name as customer_name,
       c.city,
       COUNT(DISTINCT o.id) as order_count,
       SUM(o.total_amount) as total_spent,
       AVG(o.total_amount) as avg_order_value
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.city
HAVING COUNT(DISTINCT o.id) > 0
ORDER BY total_spent DESC
LIMIT 20;

-- 查询各城市的客户数量和消费情况
SELECT city,
       COUNT(DISTINCT c.id) as customer_count,
       COUNT(DISTINCT o.id) as order_count,
       COALESCE(SUM(o.total_amount), 0) as total_revenue
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY city
ORDER BY total_revenue DESC;

-- 查询最活跃的客户（订单数量最多）
SELECT c.name, c.email, c.city,
       COUNT(o.id) as order_count,
       SUM(o.total_amount) as total_amount,
       MIN(o.order_date) as first_order,
       MAX(o.order_date) as last_order
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.email, c.city
ORDER BY order_count DESC
LIMIT 10;

-- ================================
-- 时间序列分析 (Time Series Analysis)
-- ================================

-- 按月份统计订单数量和销售额
SELECT strftime('%Y-%m', order_date) as month,
       COUNT(*) as order_count,
       SUM(total_amount) as monthly_revenue,
       AVG(total_amount) as avg_order_value
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY strftime('%Y-%m', order_date)
ORDER BY month;

-- 按季度统计销售情况
SELECT
    CASE strftime('%m', order_date)
        WHEN '01' OR '02' OR '03' THEN 'Q1'
        WHEN '04' OR '05' OR '06' THEN 'Q2'
        WHEN '07' OR '08' OR '09' THEN 'Q3'
        ELSE 'Q4'
    END as quarter,
    COUNT(*) as order_count,
    SUM(total_amount) as revenue
FROM orders
WHERE strftime('%Y', order_date) = '2023'
GROUP BY quarter
ORDER BY quarter;

-- 查询每日销售趋势（最近30天）
SELECT DATE(order_date) as order_day,
       COUNT(*) as daily_orders,
       SUM(total_amount) as daily_revenue
FROM orders
WHERE order_date >= date('now', '-30 days')
GROUP BY DATE(order_date)
ORDER BY order_day;

-- ================================
-- 复杂关联查询 (Complex Joins)
-- ================================

-- 查询每个产品的销售详情
SELECT p.name as product_name,
       c.name as category_name,
       COUNT(DISTINCT oi.order_id) as order_count,
       SUM(oi.quantity) as total_quantity_sold,
       SUM(oi.total_price) as total_revenue,
       AVG(oi.unit_price) as avg_selling_price,
       p.price as current_price
FROM products p
LEFT JOIN categories c ON p.category_id = c.id
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name, c.name, p.price
ORDER BY total_revenue DESC;

-- 查询客户购买的产品分类偏好
SELECT c.name as customer_name,
       cat.name as preferred_category,
       COUNT(DISTINCT oi.product_id) as products_bought,
       SUM(oi.total_price) as category_spending
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
JOIN categories cat ON p.category_id = cat.id
GROUP BY c.id, c.name, cat.id, cat.name
HAVING COUNT(DISTINCT oi.product_id) >= 2
ORDER BY c.name, category_spending DESC;

-- ================================
-- 业务洞察查询 (Business Intelligence)
-- ================================

-- 查询库存预警（库存低于10的产品）
SELECT p.name,
       c.name as category,
       p.stock_quantity,
       p.price,
       COALESCE(SUM(oi.quantity), 0) as total_sold
FROM products p
LEFT JOIN categories c ON p.category_id = c.id
LEFT JOIN order_items oi ON p.id = oi.product_id
WHERE p.stock_quantity < 10
GROUP BY p.id, p.name, c.name, p.stock_quantity, p.price
ORDER BY p.stock_quantity;

-- 查询销售业绩排行榜（按分类）
SELECT c.name as category,
       COUNT(DISTINCT p.id) as product_count,
       COUNT(DISTINCT oi.order_id) as order_count,
       SUM(oi.quantity) as total_quantity,
       SUM(oi.total_price) as total_revenue,
       AVG(oi.unit_price) as avg_price
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY c.id, c.name
ORDER BY total_revenue DESC;

-- 查询订单状态分布
SELECT status,
       COUNT(*) as order_count,
       SUM(total_amount) as total_amount,
       AVG(total_amount) as avg_amount,
       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2) as percentage
FROM orders
GROUP BY status
ORDER BY order_count DESC;

-- 查询客户价值分层（RFM分析简化版）
SELECT
    customer_id,
    c.name,
    COUNT(*) as frequency,
    SUM(total_amount) as monetary,
    MAX(order_date) as recency,
    CASE
        WHEN SUM(total_amount) > 5000 THEN 'High Value'
        WHEN SUM(total_amount) > 2000 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_segment
FROM orders o
JOIN customers c ON o.customer_id = c.id
GROUP BY customer_id, c.name
ORDER BY monetary DESC;

-- ================================
-- 推荐系统相关查询 (Recommendation Queries)
-- ================================

-- 查询经常一起购买的产品组合
SELECT p1.name as product1,
       p2.name as product2,
       COUNT(*) as co_purchase_count
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
JOIN products p1 ON oi1.product_id = p1.id
JOIN products p2 ON oi2.product_id = p2.id
GROUP BY p1.id, p1.name, p2.id, p2.name
HAVING COUNT(*) >= 2
ORDER BY co_purchase_count DESC
LIMIT 20;

-- 查询客户可能感兴趣的产品（基于同类产品）
SELECT DISTINCT p.name, p.price, c.name as category
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE c.id IN (
    SELECT DISTINCT cat.id
    FROM customers cust
    JOIN orders o ON cust.id = o.customer_id
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products prod ON oi.product_id = prod.id
    JOIN categories cat ON prod.category_id = cat.id
    WHERE cust.id = 1  -- 特定客户ID
)
AND p.id NOT IN (
    SELECT DISTINCT oi.product_id
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.customer_id = 1  -- 该客户已购买的产品
)
ORDER BY p.price;

-- ================================
-- 性能分析查询 (Performance Analysis)
-- ================================

-- 查询查询执行计划（SQLite特定）
EXPLAIN QUERY PLAN
SELECT p.name, c.name, p.price
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE p.price > 1000
ORDER BY p.price DESC;

-- 检查表大小和记录数
SELECT
    'categories' as table_name,
    COUNT(*) as record_count
FROM categories
UNION ALL
SELECT
    'products' as table_name,
    COUNT(*) as record_count
FROM products
UNION ALL
SELECT
    'customers' as table_name,
    COUNT(*) as record_count
FROM customers
UNION ALL
SELECT
    'orders' as table_name,
    COUNT(*) as record_count
FROM orders
UNION ALL
SELECT
    'order_items' as table_name,
    COUNT(*) as record_count
FROM order_items;

-- ================================
-- 常见业务问题的SQL查询
-- ================================

-- 1. 今日订单数量和金额
SELECT
    COUNT(*) as today_orders,
    SUM(total_amount) as today_revenue
FROM orders
WHERE DATE(order_date) = DATE('now');

-- 2. 本月最畅销的产品
SELECT p.name,
       SUM(oi.quantity) as total_sold,
       SUM(oi.total_price) as revenue
FROM products p
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE strftime('%Y-%m', o.order_date) = strftime('%Y-%m', 'now')
GROUP BY p.id, p.name
ORDER BY total_sold DESC
LIMIT 10;

-- 3. 客户复购率分析
SELECT
    'Single Purchase' as customer_type,
    COUNT(*) as customer_count
FROM (
    SELECT customer_id, COUNT(*) as order_count
    FROM orders
    GROUP BY customer_id
    HAVING order_count = 1
)
UNION ALL
SELECT
    'Repeat Customer' as customer_type,
    COUNT(*) as customer_count
FROM (
    SELECT customer_id, COUNT(*) as order_count
    FROM orders
    GROUP BY customer_id
    HAVING order_count > 1
);

-- 4. 平均订单处理时间（如果有处理时间字段）
-- 注：这个查询假设我们有订单状态变更的时间戳

-- 5. 退款/取消订单分析
SELECT
    status,
    COUNT(*) as count,
    SUM(total_amount) as total_amount_affected,
    AVG(total_amount) as avg_order_value
FROM orders
WHERE status IN ('cancelled')
GROUP BY status;