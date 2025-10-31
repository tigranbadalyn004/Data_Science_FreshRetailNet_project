SELECT * 
FROM raw_sales_data 

-- Q1. How many stores and cities are in the dataset?
SELECT 
    COUNT(DISTINCT store_id) AS unique_stores,
    COUNT(DISTINCT city_id) AS unique_cities
FROM raw_sales_data;

--Q2. What date range does our data cover?
SELECT 
    MIN(dt) AS start_date,
    MAX(dt) AS end_date
FROM raw_sales_data;

--Q3. How many different products and product categories exist?
SELECT 
    COUNT(DISTINCT product_id) AS unique_products,
    COUNT(DISTINCT first_category_id) AS first_categories,
    COUNT(DISTINCT second_category_id) AS second_categories,
    COUNT(DISTINCT third_category_id) AS third_categories
FROM raw_sales_data;


--Q4. What is the total sales volume across all data?

SELECT 
    SUM(sale_amount) AS total_sales
FROM raw_sales_data;

--Q5. What is the average sales per hour?

select sum(sale_amount)/(count(*)*24)
FROM raw_sales_data;

--Q6. How many hours had zero sales vs non-zero sales?

SELECT 
    SUM(CASE WHEN sale_amount = 0 THEN 1 ELSE 0 END) AS zero_sales_hours,
    SUM(CASE WHEN sale_amount > 0 THEN 1 ELSE 0 END) AS non_zero_sales_hours
FROM raw_sales_data;

-- Task 2: Time Patterns

-- Task 2A: Daily Rhythms

-- Task 2: Time Patterns

-- Task 2A: Daily Rhythms

--Q7: Show total sales by hour of day. Which hours are busiest?

SELECT 
    hours_sale AS hour,
    SUM(sale_amount) AS total_sales
FROM raw_sales_data
GROUP BY hours_sale
ORDER BY total_sales DESC;

--Q8: Show total sales by day of week. Which days generate most sales?

SELECT 
    EXTRACT(DOW FROM dt) AS day_of_week,  -- 0=Sunday, 6=Saturday
    SUM(sale_amount) AS total_sales
FROM raw_sales_data
GROUP BY day_of_week
ORDER BY total_sales DESC;


--Task 2B: External Factors

--Q9: Compare average sales between holiday and non-holiday periods

SELECT 
    holiday_flag,
    AVG(sale_amount) AS avg_sales
FROM raw_sales_data
GROUP BY holiday_flag;


--Q10: Do rainy days (precipitation > 0) affect sales compared to clear days?

SELECT 
    CASE WHEN precpt > 0 THEN 'rainy' ELSE 'clear' END AS weather,
    AVG(sale_amount) AS avg_sales
FROM raw_sales_data
GROUP BY weather;


--Task 3: Building Your Own Business Views

--Task 3A: Design an Hourly Business Summary View

--CREATE VIEW hourly_business_summary AS
-- Your code here
-- Think about what columns would be most useful
-- Consider creating calculated fields that make future queries simpler


DROP VIEW IF EXISTS hourly_business_tb;

CREATE OR REPLACE VIEW hourly_business_tb AS
SELECT
    dt,
    -- Օր/ժամ
    EXTRACT(DOW FROM dt) AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM dt) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,
    EXTRACT(MONTH FROM dt) AS month,

    -- Լոկացիա և պրոդուկտ
    city_id,
    store_id,
    first_category_id,
    second_category_id,
    third_category_id,
    product_id,

    -- Վաճառքներ
    sale_amount::numeric AS sale_amount,
    CASE WHEN sale_amount::numeric > 0 THEN 1 ELSE 0 END AS sale_flag,
    CASE WHEN sale_amount::numeric > 10 THEN 1 ELSE 0 END AS high_sales_flag,

    -- Ակցիաներ և ակտիվություն
    discount::numeric AS discount,  
    CASE WHEN discount::numeric > 0 THEN 1 ELSE 0 END AS promo_flag,
    holiday_flag,
    activity_flag,

    -- Եղանակ
    CASE WHEN precpt::numeric > 0 THEN 'rainy' ELSE 'clear' END AS weather,
    precpt::numeric AS precpt,
    avg_temperature::numeric AS avg_temperature,
    avg_humidity::numeric AS avg_humidity,
    avg_wind_level::numeric AS avg_wind_level,

    -- 🟢 Stockout flag
    CASE 
        WHEN sale_amount::numeric = 0 AND activity_flag = 1 THEN 1 
        ELSE 0 
    END AS stockout_flag

FROM raw_sales_data;

SELECT 
    city_id,
    SUM(sale_amount) AS total_sales
FROM hourly_business_tb
WHERE is_weekend = 1
  AND promo_flag = 1
GROUP BY city_id
ORDER BY total_sales DESC;


select *
from hourly_business_tb


-- Task 3B: Build a Store Performance Dashboard View

-- CREATE VIEW store_performance_dashboard AS
-- Your code here  
-- Focus on metrics that would help a regional manager evaluate store performance
-- Use aggregate functions with conditional logic (CASE statements within aggregations)
-- Think about percentages and ratios that enable fair comparisons

-- --📌 steps 
-- 🔹 Operational Scale (масштаб работы)

-- total_hours → COUNT(*) (всего часов наблюдений)

-- unique_products → COUNT(DISTINCT product_id)

-- total_sales → SUM(sale_amount)

-- 🔹 Sales Performance

-- avg_sales_per_hour → AVG(sale_amount)

-- avg_sales_per_product → SUM(sale_amount) / COUNT(DISTINCT product_id)

-- 🔹 Inventory Management Effectiveness

-- stockout_hours → SUM(stockout_flag)

-- stockout_rate → AVG(stockout_flag) (процент часов без товара)

-- 🔹 Customer Engagement

-- active_hours → SUM(sale_flag)

-- engagement_rate → AVG(sale_flag) (процент активных часов)

-- avg_sales_active_hour → SUM(sale_amount) / NULLIF(SUM(sale_flag),0)

-- 🔹 Promotional Effectiveness

-- promo_sales → SUM(CASE WHEN promo_flag=1 THEN sale_amount ELSE 0 END)

-- nonpromo_sales → SUM(CASE WHEN promo_flag=0 THEN sale_amount ELSE 0 END)

-- promo_share → promo_sales / total_sales

-- 🔹 Consistency Indicators

-- sales_stddev → STDDEV(sale_amount)

-- sales_cv → STDDEV(sale_amount) / NULLIF(AVG(sale_amount),0)


CDROP VIEW IF EXISTS store_performance_dashboard_tb;

CREATE VIEW store_performance_dashboard_tb AS
SELECT
    -- Store identifiers
    city_id,
    store_id,

    -- Operational scale metrics
    COUNT(DISTINCT dt) AS total_days_recorded,         -- total number of days data is available
    COUNT(*) AS total_hours_recorded,                  -- total number of hourly records

    -- Sales performance metrics
    SUM(sale_amount) AS total_sales,
    SUM(sale_flag) AS hours_with_sales,               -- total hours with any sales
    SUM(high_sales_flag) AS hours_with_high_sales,    -- total hours with high sales
    AVG(sale_amount) AS avg_sales_per_hour,           -- average sales per hour

    -- Inventory management effectiveness
    SUM(stockout_flag) AS total_stockout_hours,       -- total hours where stockout occurred

    -- Customer engagement metrics
    AVG(sale_flag::numeric) AS engagement_rate,       -- fraction of hours with sales

    -- Promotional effectiveness
    SUM(promo_flag) AS promo_hours,                   -- number of hours with active promotion
    AVG(CASE WHEN promo_flag = 1 THEN sale_amount ELSE NULL END) AS avg_sales_during_promo, -- average sales during promo

    -- Weather impact
    SUM(CASE WHEN weather = 'rainy' THEN 1 ELSE 0 END) AS rainy_hours,
    AVG(CASE WHEN weather = 'rainy' THEN sale_amount ELSE NULL END) AS avg_sales_rainy_hours

FROM hourly_business_tb
GROUP BY city_id, store_id;


-- check the view
select *
from store_performance_dashboard_tb

-- Top 5 stores by total sales
SELECT city_id, store_id, total_sales, avg_sales_per_hour
FROM store_performance_dashboard_tb
ORDER BY total_sales DESC
LIMIT 5;

-- Bottom 5 stores by potential stockout hours
SELECT city_id, store_id, potential_stockout_hours
FROM store_performance_dashboard
ORDER BY potential_stockout_hours DESC
LIMIT 5;



-- Task 3C: Create a Product Category Intelligence View

-- CREATE VIEW category_intelligence AS
-- Your code here
-- Think about what insights would help someone decide which categories to focus on
-- Consider metrics that reveal both opportunities and challenges
-- Use conditional aggregations to analyze performance under different conditions


DROP VIEW IF EXISTS category_intelligence;

CREATE VIEW category_intelligence AS
SELECT
    first_category_id,

    -- Market presence
    COUNT(DISTINCT store_id) AS stores_count,
    SUM(sale_amount) AS total_sales,
    AVG(sale_amount) AS avg_sales_per_record,

    -- Performance characteristics
    AVG(CASE WHEN sale_flag = 1 THEN sale_amount ELSE NULL END) AS avg_sales_when_sold,
    STDDEV(sale_amount) AS sales_stddev,

    -- Inventory challenges
    SUM(CASE WHEN stockout_flag = 1 THEN 1 ELSE 0 END) AS total_stockout_hours,
    SUM(CASE WHEN stockout_flag = 1 THEN sale_amount ELSE 0 END) AS potential_lost_sales,

    -- External factor responsiveness
    SUM(CASE WHEN promo_flag = 1 THEN sale_amount ELSE 0 END) AS promo_sales,
    SUM(CASE WHEN weather = 'rainy' THEN sale_amount ELSE 0 END) AS rainy_day_sales,
    SUM(CASE WHEN holiday_flag = 1 THEN sale_amount ELSE 0 END) AS holiday_sales,

    -- Operational consistency
    AVG(sale_amount) / NULLIF(STDDEV(sale_amount),0) AS consistency_index

FROM hourly_business_tb
GROUP BY first_category_id;


--Task 3D: Design a Time-Based Business Patterns View

DROP VIEW IF EXISTS business_rhythm_patterns;

CREATE VIEW business_rhythm_patterns AS
SELECT
    -- Time structure
    day_of_week,
    month,

    -- Activity intensity
    AVG(sale_amount) AS avg_sales,
    AVG(sale_flag::numeric) AS activity_rate,    -- доля часов с продажами
    SUM(high_sales_flag) AS high_sales_hours,    -- количество часов с большими продажами

    -- Operational challenges
    SUM(stockout_flag) AS total_stockout_hours,  -- количество часов без запасов
    AVG(stockout_flag::numeric) AS stockout_rate,

    -- External context
    SUM(CASE WHEN weather = 'rainy' THEN 1 ELSE 0 END) AS rainy_hours,
    SUM(CASE WHEN promo_flag = 1 THEN 1 ELSE 0 END) AS promo_hours,
    SUM(CASE WHEN holiday_flag = 1 THEN 1 ELSE 0 END) AS holiday_hours,

    -- Resource requirements (pressure indicator)
    AVG(sale_amount + discount::numeric * 10) AS operational_pressure_index,

    -- Performance classification
    CASE 
        WHEN AVG(sale_amount) > 15 THEN 'High Activity'
        WHEN AVG(sale_amount) BETWEEN 5 AND 15 THEN 'Medium Activity'
        ELSE 'Low Activity'
    END AS activity_level

FROM hourly_business_tb
GROUP BY day_of_week, month;




-- Task 4: Inventory and Stockout Analysis
-- Task 4A: Stockout Impact

-- Question 11: What percentage of total hours experienced stockouts?

SELECT 
    100.0 * SUM(stockout_flag)::numeric / COUNT(*) AS stockout_percentage
FROM hourly_business_tb;

-- Question 12: Compare average sales between in-stock and stocked-out hours.

SELECT 
    stockout_flag,
    AVG(sale_amount) AS avg_sales
FROM hourly_business_tb
GROUP BY stockout_flag;

-- Question 13: Which 10 products have the highest stockout rates?

SELECT 
    product_id,
    100.0 * SUM(stockout_flag)::numeric / COUNT(*) AS stockout_rate_percentage
FROM hourly_business_tb
GROUP BY product_id
ORDER BY stockout_rate_percentage DESC
LIMIT 10;

-- Task 4B: Stockout Timing

-- Question 14: Do stockouts occur more at certain hours? Show stockout rate by hour.
SELECT
    hours_sale AS hour_of_day,
    100.0 * SUM(stockout_flag)::numeric / COUNT(*) AS stockout_rate_percentage
FROM hourly_business_tb
GROUP BY hours_sale
ORDER BY hours_sale;

-- Question 15: Which 5 stores have the worst stockout rates?

SELECT
    store_id,
    100.0 * SUM(stockout_flag)::numeric / COUNT(*) AS stockout_rate_percentage
FROM hourly_business_tb
GROUP BY store_id
ORDER BY stockout_rate_percentage DESC
LIMIT 5;

-- Task 5: Performance Comparison
-- Task 5A: Store Rankings

-- Question 16: Rank the top 10 stores by total sales.

SELECT
    city_id,
    store_id,
    SUM(sale_amount) AS total_sales
FROM hourly_business_tb
GROUP BY city_id, store_id
ORDER BY total_sales DESC
LIMIT 10;

-- Question 17: Which cities have the highest average sales per store?

SELECT
    city_id,
    AVG(store_total_sales) AS avg_sales_per_store
FROM (
    SELECT city_id, store_id, SUM(sale_amount) AS store_total_sales
    FROM hourly_business_tb
    GROUP BY city_id, store_id
) AS store_sales
GROUP BY city_id
ORDER BY avg_sales_per_store DESC;


-- Question 18: Using your store_performance view, identify stores with high sales but also high stockout rates.

SELECT
    city_id,
    store_id,
    SUM(sale_amount) AS total_sales,
    100.0 * SUM(stockout_flag)::numeric / COUNT(*) AS stockout_rate_percentage
FROM hourly_business_tb
GROUP BY city_id, store_id
HAVING SUM(sale_amount) > 1000  -- пример порога для "high sales"
   AND (SUM(stockout_flag)::numeric / COUNT(*)) > 0.2  -- пример порога >20%
ORDER BY total_sales DESC;


-- Task 5B: Product Analysis
-- Question 19: Which product categories generate the most total sales?
SELECT
    first_category_id,
    SUM(sale_amount) AS total_sales
FROM hourly_business_tb
GROUP BY first_category_id
ORDER BY total_sales DESC;


-- Question 20: Which individual products have the highest average hourly sales?

SELECT
    product_id,
    AVG(sale_amount) AS avg_hourly_sales
FROM hourly_business_tb
GROUP BY product_id
ORDER BY avg_hourly_sales DESC;


-- Task 6: Promotional Analysis
-- Task 6A: Discount Effectiveness

-- Question 21: Compare average sales between discounted (discount > 0) and regular price hours.
SELECT
    CASE WHEN discount > 0 THEN 'discounted' ELSE 'regular' END AS price_type,
    AVG(sale_amount) AS avg_sales
FROM hourly_business_tb
GROUP BY price_type;


-- Question 22: What discount ranges (0%, 1-10%, 11-20%, 21%+) perform best?
SELECT
    CASE 
        WHEN discount = 0 THEN '0%' 
        WHEN discount BETWEEN 0.01 AND 0.10 THEN '1-10%' 
        WHEN discount BETWEEN 0.11 AND 0.20 THEN '11-20%' 
        ELSE '21%+' 
    END AS discount_range,
    AVG(sale_amount) AS avg_sales
FROM hourly_business_tb
GROUP BY discount_range
ORDER BY avg_sales DESC;


-- Question 23: Which product categories respond best to promotions?

SELECT
    first_category_id,
    AVG(CASE WHEN promo_flag = 1 THEN sale_amount ELSE NULL END) AS avg_sales_during_promo,
    AVG(CASE WHEN promo_flag = 0 THEN sale_amount ELSE NULL END) AS avg_sales_regular,
    AVG(CASE WHEN promo_flag = 1 THEN sale_amount ELSE NULL END) - 
    AVG(CASE WHEN promo_flag = 0 THEN sale_amount ELSE NULL END) AS promo_lift
FROM hourly_business_tb
GROUP BY first_category_id
ORDER BY promo_lift DESC;

-- Task 6B: Activity Analysis
-- Question 24: How do sales compare when activity_flag = 1 vs activity_flag = 0?
SELECT
    activity_flag,
    AVG(sale_amount) AS avg_sales,
    SUM(sale_amount) AS total_sales,
    COUNT(*) AS total_hours
FROM hourly_business_tb
GROUP BY activity_flag;


-- Question 25: Do promotional activities help during stockout situations?

SELECT
    promo_flag,
    AVG(sale_amount) AS avg_sales,
    COUNT(*) AS hours_count
FROM hourly_business_tb
WHERE sale_flag = 1 AND stockout_flag = 1
GROUP BY promo_flag;



-- Task 7: Advanced Business Questions
-- Task 7A: Complex Patterns

-- Question 26: Find store-product combinations with high sales potential but frequent stockouts.
SELECT
    store_id,
    product_id,
    AVG(CASE WHEN stockout_flag = 0 THEN sale_amount ELSE NULL END) AS avg_sales_when_in_stock,
    SUM(stockout_flag) AS stockout_hours,
    COUNT(*) AS total_hours,
    (SUM(stockout_flag)::numeric / COUNT(*)) AS stockout_rate
FROM hourly_business_tb
GROUP BY store_id, product_id
HAVING AVG(CASE WHEN stockout_flag = 0 THEN sale_amount ELSE NULL END) > 5 -- high sales threshold
   AND (SUM(stockout_flag)::numeric / COUNT(*)) > 0.2 -- frequent stockout threshold
ORDER BY stockout_rate DESC, avg_sales_when_in_stock DESC
LIMIT 20;


-- Question 27: How do different weather conditions affect sales by product category?
SELECT
    first_category_id,
    CASE
        WHEN avg_temperature < 10 THEN 'cold'
        WHEN avg_temperature BETWEEN 10 AND 20 THEN 'mild'
        WHEN avg_temperature BETWEEN 21 AND 30 THEN 'warm'
        ELSE 'hot'
    END AS temp_range,
    weather,
    AVG(sale_amount) AS avg_sales,
    SUM(sale_amount) AS total_sales,
    COUNT(*) AS hours_count
FROM hourly_business_tb
GROUP BY first_category_id, temp_range, weather
ORDER BY first_category_id, temp_range;


-- Task 7B: Business Opportunities

-- Question 28: During peak hours (identify them first), which products are most likely to be out of stock?
WITH hourly_sales AS (
    SELECT
        hours_sale,
        SUM(sale_amount) AS total_sales
    FROM hourly_business_tb
    GROUP BY hours_sale
)
SELECT hours_sale
FROM hourly_sales
ORDER BY total_sales DESC
LIMIT 3;  -- топ 3 пиковых часа


WITH peak_hours AS (
    SELECT hours_sale
    FROM (
        SELECT hours_sale, SUM(sale_amount) AS total_sales
        FROM hourly_business_tb
        GROUP BY hours_sale
    ) AS hs
    ORDER BY total_sales DESC
    LIMIT 3
)
SELECT 
    product_id,
    COUNT(*) AS stockout_hours,
    SUM(stockout_flag) AS total_stockouts
FROM hourly_business_tb hbt
JOIN peak_hours ph ON hbt.hours_sale = ph.hours_sale
WHERE stockout_flag = 1
GROUP BY product_id
ORDER BY total_stockouts DESC
LIMIT 10;


-- Question 29: If a store could eliminate all stockouts, estimate the potential sales increase.

WITH avg_sales_in_stock AS (
    SELECT product_id,
           AVG(CASE WHEN stockout_flag = 0 THEN sale_amount ELSE NULL END) AS avg_in_stock_sales
    FROM hourly_business_tb
    GROUP BY product_id
)
SELECT 
    SUM(sale_amount) AS actual_sales,
    SUM(CASE WHEN stockout_flag = 1 THEN asi.avg_in_stock_sales ELSE sale_amount END) AS potential_sales,
    SUM(CASE WHEN stockout_flag = 1 THEN asi.avg_in_stock_sales ELSE sale_amount END) - SUM(sale_amount) AS potential_increase
FROM hourly_business_tb hbt
JOIN avg_sales_in_stock asi ON hbt.product_id = asi.product_id;

-- Question 30: Create a "store health score" combining sales performance and operational efficiency.

WITH store_metrics AS (
    SELECT 
        store_id,
        SUM(sale_amount) AS total_sales,
        AVG(sale_flag::numeric) AS engagement_rate,
        (SUM(stockout_flag)::numeric / COUNT(*)) AS stockout_rate,
        STDDEV(sale_amount) AS sales_stddev
    FROM hourly_business_tb
    GROUP BY store_id
)
SELECT
    store_id,
    total_sales,
    engagement_rate,
    stockout_rate,
    sales_stddev,
    -- Health score formula: higher sales + higher engagement - higher stockout - higher variability
    (total_sales * 0.5 + engagement_rate * 100 * 0.3 - stockout_rate * 100 * 0.1 - sales_stddev * 0.1) AS store_health_score
FROM store_metrics
ORDER BY store_health_score DESC
LIMIT 20;
