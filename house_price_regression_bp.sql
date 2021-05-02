select * from bp;

select distinct bedrooms, count(bedrooms) as b from bp
group by 1
order by bed desc;

select distinct bathrooms, count(bathrooms) as b from bp
group by 1
order by b desc;

select distinct floors, count(floors) as f from bp
group by 1
order by f desc;

select distinct grade, count(grade) as g from bp
group by 1
order by g desc;

select distinct bp.condition, count(bp.condition) as c from bp
group by 1
order by c desc;


SELECT * from bp
WHERE bedrooms = 3 and bathrooms = 4 and floors = 1 and waterfront = 0 and bp.condition > 2 and grade > 4 and price < 300000;

SELECT * from bp
WHERE bedrooms in (3,4) and bathrooms > 3 and floors = 1 and waterfront = 0 and bp.condition > 2 and grade > 4 and price < 300000;

SELECT * from bp
WHERE (bedrooms = 3 or bedrooms = 4) and bathrooms > 3 and floors = 1 and waterfront = 0 and bp.condition > 2 and grade > 4 and price < 300000;

select id, price from bp
where price > (select 2 * avg(price) from bp);

select * from bp
where price > (select 2 * avg(price) from bp)
order by price asc;

create view Houses_with_higher_than_double_average_price as
select * from bp
where price > (select 2 * avg(price) from bp)
order by price asc;

select distinct bedrooms, avg(price) as p from bp
where bedrooms in (3,4)
group by 1
order by p desc;

select distinct zip_code, count(id) from bp
group by 1;

select * from bp
where yr_renovated > 0;

select * from bp
order by price desc
limit 11;

select 
    *
from
    (select * from bp
order by price desc
limit 11) as tbl
order by price asc
limit 1;