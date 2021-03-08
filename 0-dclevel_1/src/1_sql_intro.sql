--  --  --  -- Introduction to SQL
--FYI it's a compilation of how to work
--with different commands.

--select all from a table 
SELECT * FROM people;

--select just one column
SELECT name FROM people;

-- printing 
SELECT 'Hello World'
AS result;

-- select several columns from a table 
SELECT name, birthdate
FROM people;

-- limit selected rows 
SELECT *
FROM people
LIMIT 10;

-- select in a columns just different values 
SELECT DISTINCT language
FROM films;


-- count all rows in a table
SELECT COUNT(*)
FROM people;

-- count in a particular column
SELECT COUNT(birthdate)
FROM people;

-- count different values in a particular column
SELECT COUNT(DISTINCT birthdate)
FROM people;

-- Where syntaxis:
-- = equal
-- <> not equal
-- < less than
-- > greater than
-- <= less than or equal to
-- >= greater than or equal to
SELECT title
FROM films
WHERE title = 'Metropolis'; -- string
-- and others ...

-- using logic connectors
SELECT title
FROM films
WHERE release_year > 1994 -- int, double, etc
AND release_year < 2000;

-- or/and
SELECT title
FROM films
WHERE (release_year = 1994 OR release_year = 1995)
AND (certification = 'PG' OR certification = 'R');


-- beetween connector
SELECT title
FROM films
WHERE release_year
BETWEEN 1994 AND 2000;

-- in 
SELECT name
FROM kids
WHERE age IN (2, 4, 6, 8, 10);

-- null -> missing value 
SELECT COUNT(*)
FROM people
WHERE birthdate IS NULL;

-- is not null  
SELECT name
FROM people
WHERE birthdate IS NOT NULL;


-- it's like using possible combinations 
-- begins with 
SELECT name
FROM people
WHERE name LIKE 'B%';

-- something start with and ends with 
SELECT name
FROM people
WHERE name LIKE 'B_da';

-- MATH functions in ONE column
-- AVG
SELECT AVG(budget)
FROM films;

-- MAX
SELECT MAX(budget)
FROM films;

-- SUM
SELECT SUM(budget)
FROM films;


-- calculate values
-- operators: +, -, *, and /
-- just print a value 
SELECT (4 * 3);

-- print value with title 
SELECT (4 * 3) 
AS result;

-- MATH functions in SEVERAL column (just comma separator)
SELECT MAX(budget) AS max_budget,
       MAX(duration) AS max_duration
FROM films;

-- order by 
-- ASC and DESC 
SELECT name
FROM people
ORDER BY name ASC;

-- combined order by 
SELECT title
FROM films
WHERE release_year IN (2000,2012)
ORDER BY release_year DESC

-- group by 
SELECT sex, count(*)
FROM employees
GROUP BY sex;

-- using min and group by
SELECT release_year, MIN(gross)
FROM films
GROUP BY release_year;

-- using HAVING/COUNT/GROUP BY/LIMIT/ORDER BY 
SELECT country, AVG(budget) as avg_budget, AVG(gross) as avg_gross
-- select country, average budget, 
--     and average gross
FROM films
-- from the films table
GROUP BY country
-- group by country 
HAVING COUNT(country)  > 10
-- where the country has more than 10 titles
ORDER BY country ASC
-- order by country
LIMIT 5;
-- limit to only show 5 results

-- USING JOIN 
SELECT title, imdb_score
FROM films
JOIN reviews
ON films.id = reviews.film_id
WHERE title = 'To Kill a Mockingbird';
