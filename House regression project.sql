USE house_price_regression;

SELECT * FROM house_price_regression.bp;

ALTER TABLE bp DROP data;

SELECT * from bp LIMIT 10;

SELECT count(id) as number_of_rows from bp;

SELECT DISTINCT bedrooms, count(bedrooms) as number_of_bedrooms
from bp
group by 1;

SELECT DISTINCT bathrooms, count(bathrooms) as number_of_bathrooms
from bp
group by 1;

SELECT DISTINCT floors, count(floors) as number_of_floors
from bp
group by 1;

SELECT DISTINCT bp.condition, count(bp.condition)
from bp
group by 1;

SELECT DISTINCT bp.grade, count(bp.grade)
from bp
group by 1;

SELECT *
from bp
ORDER BY price DESC
LIMIT 10;

SELECT ROUND(AVG(price))
from bp;

SELECT DISTINCT bedrooms, ROUND(AVG(price)) as average_price
from bp
group by 1; 

SELECT DISTINCT bedrooms, ROUND(AVG(sqft_living)) as average_sqft_living
from bp
group by 1; 

SELECT DISTINCT waterfront, ROUND(AVG(price)) as average_price
from bp
group by 1; 

SELECT bp.condition, AVG(bp.grade)
from bp
group by bp.condition
order by 1;

SELECT AVG(bp.condition), bp.grade
from bp
group by bp.grade
order by 2;

SELECT * from bp
WHERE (bedrooms = 3 or bedrooms = 4) and bathrooms > 3 and floors = 1 and waterfront = 0 and bp.condition > 2 and grade > 4 and price < 300000;


SELECT *
from bp 
where price > (SELECT 2*AVG(price)from bp)
ORDER by price ASC;

create view Houses_with_higher_than_double_average_price AS
SELECT *
from bp 
where price > (SELECT 2*AVG(price)from bp)
ORDER by price ASC;

select distinct bedrooms, avg(price) as p from bp
where bedrooms in (3,4)
group by 1
order by p desc;

SELECT DISTINCT zip_code, count(id) as number_of_properties
from bp
group by 1;

SELECT *
from bp
where yr_renovated > 0;

SELECT *
from (
SELECT *
from bp
ORDER BY bp.price DESC
LIMIT 11
) as p
order by p.price ASC
limit 1;