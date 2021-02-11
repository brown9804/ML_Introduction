-- JOINING SQL intro 
--FYI it's a compilation of how to work
--with different commands.

-- INNER Join # example 0
-- 3. Select fields with ALIASES
SELECT c.code AS country_code, name, year, inflation_rate
FROM countries AS c
  -- 1. Join to economies (alias e)
  INNER JOIN economies AS e
    -- 2. Match on code
    ON c.code = e.code;

-- INNER Join # example 1
-- 4. Select fields
SELECT c.code, c.name, c.region, p.year, p.fertility_rate
  -- 1. From countries (alias as c)
  FROM countries AS c
  -- 2. Join with populations (as p)
INNER JOIN populations as p
    -- 3. Match on country code
    ON c.code = p.country_code;

-- INNER JOIN WITH COMMON COLUMN NAME 
-- 4. Select fields
SELECT c.name as country, c.continent, l.name as language,
l.official
  -- 1. From countries (alias as c)
  FROM countries as c
  -- 2. Join to languages (as l)
    INNER JOIN languages as l
    -- 3. Match using code
      USING(code)

-- SELF JOIN 
-- 4. Select fields with aliases
SELECT p1.country_code, 
p1.size AS size2010,
p2.size AS size2015
-- 1. From populations (alias as p1)
FROM populations AS p1
  -- 2. Join to itself (alias as p2)
  INNER JOIN populations as p2
    -- 3. Match on country code
    ON p1.country_code = p2.country_code;

-- JOIN using CASE, WHERE and THEN
SELECT name, continent, code, surface_area,
    -- 1. First case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- 2. Second case
        WHEN surface_area > 350000 and surface_area < 2000000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name
        AS geosize_group
-- 5. From table
FROM countries;

-- INNER JOIN and create a column with aggregated categories
SELECT country_code, size,
    -- 1. First case
    CASE WHEN size > 50000000 THEN 'large'
        -- 2. Second case
        WHEN size > 1000000 and size < 50000000 THEN 'medium'
        -- 3. Else clause + end
        ELSE 'small' END
        -- 4. Alias name (popsize_group)
        AS popsize_group
-- 5. From table
FROM populations
-- 6. Focus on 2015
WHERE year = 2015;

-- LEFT Join with ALIAS
-- Select the city name (with alias), the country code,
-- the country name (with alias), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- From left table (with alias)
FROM cities AS c1
  -- Join to right table (with alias)
  INNER JOIN countries AS c2
    -- Match on country code
    ON c1.country_code = c2.code
-- Order by descending country code
ORDER BY code DESC;

-- LEFT Join 
-- Select country name AS country, the country's local name,
-- the language name AS language, and
-- the percent of the language spoken in the country
SELECT c.name AS country, local_name, l.name AS language, percent
-- 1. From left table (alias as c)
FROM countries AS c
  -- 2. Join to right table (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on fields
    ON c.code = l.code
-- 4. Order by descending country
ORDER BY country DESC;

-- RIGHT JOIN ~ LEFT JOIN
-- convert this code to use RIGHT JOINs instead of LEFT JOINs
/*  ---- LEFT JOIN
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
  LEFT JOIN countries
    ON cities.country_code = countries.code
  LEFT JOIN languages
    ON countries.code = languages.code
ORDER BY city, language;
*/ ---- RIGHT JOIN
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
  RIGHT JOIN countries
    ON  countries.code = languages.code
  RIGHT JOIN cities
    ON cities.country_code = countries.code
ORDER BY city, language;

-- FULL JOIN 
SELECT name AS country, code, region, basic_unit
-- 3. From countries
FROM countries
  -- 4. Join to currencies
  FULL JOIN currencies
    -- 5. Match on code
    USING (code)
-- 1. Where region is North America or null
WHERE region = 'North America' OR region IS NULL
-- 2. Order by region
ORDER BY region;

-- FULL JOIn using LIKE 
SELECT countries.name, code, languages.name AS language
-- 3. From languages
FROM languages
  -- 4. Join to countries
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS null
-- 2. Order by ascending countries.name
ORDER BY countries.name ASC;

-- FULL JOIN -- USING 
-- 7. Select fields (with aliases)
SELECT c1.name AS country, region, l.name AS language,
       basic_unit, frac_unit
-- 1. From countries (alias as c1)
FROM countries AS c1
  -- 2. Join with languages (alias as l)
  FULL JOIN languages AS l
    -- 3. Match on code
    USING (code)
  -- 4. Join with currencies (alias as c2)
  FULL JOIN currencies AS c2
    -- 5. Match on code
    USING (code)
-- 6. Where region like Melanesia and Micronesia
WHERE region LIKE 'M%esia';

-- CROSS JOIN 
-- 4. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c        
  -- 2. Join to languages (alias as l)
  CROSS JOIN languages AS l
-- 3. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';

-- LEFT JOIN, WHERE, ORDER BY, LIMIT
-- Select fields
SELECT c.name AS country, c.region, p.life_expectancy AS life_exp
-- From countries (alias as c)
FROM countries as c
  -- Join to populations (alias as p)
  LEFT JOIN populations as p
    -- Match on country code
    ON c.code = p.country_code
-- Focus on 2010
WHERE p.year  = 2010
-- Order by life_exp
ORDER BY life_exp ASC
-- Limit to 5 records
LIMIT 5;

-- UNION
-- Select fields from 2010 table
SELECT *
  -- From 2010 table
  FROM economies2010
	-- Set theory clause
	UNION ALL
-- Select fields from 2015 table
SELECT *
  -- From 2015 table
  FROM economies2015
-- Order by code and year
ORDER BY code, year ASC;

-- UNION Non duplicates
-- Select field
SELECT country_code
  -- From cities
  FROM cities 
	-- Set theory clause
	UNION 
-- Select field
SELECT code as country_code
  -- From currencies
  FROM currencies
-- Order by country_code
ORDER BY country_code;

-- UNION duplicaates
-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	UNION ALL
-- Select fields
SELECT country_code, year
  -- From populations
  FROM populations
-- Order by code, year
ORDER BY code, year;

-- Intersect
-- Select fields
SELECT code, year
  -- From economies
  FROM economies
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT country_code, year
  -- From populations
  FROM populations
-- Order by code and year
ORDER BY code, year;

-- Intersect -> countries and cities ----  returns only records appearing in both tables
-- Select fields
SELECT name
  -- From countries
  FROM countries
	-- Set theory clause
	INTERSECT
-- Select fields
SELECT name
  -- From cities
  FROM cities;

-- EXCEPT example 
-- Select field
SELECT city.name
  -- From cities
  FROM cities as city
	-- Set theory clause
	EXCEPT
-- Select field
SELECT country.capital
  -- From countries
  FROM countries as country
-- Order by result
ORDER BY name;

-- SAME ABOVE just different syntax
-- Select field
SELECT country.capital
  -- From countries
  FROM countries AS country
	-- Set theory clause
	EXCEPT
-- Select field
SELECT city.name 
  -- From cities
  FROM cities AS city
-- Order by ascending capital
ORDER BY capital ASC;

-- SELECT DISTINC
SELECT *
  FROM countries
WHERE region = 'Middle East';

-- Select DISTINCT
SELECT DISTINCT name 
  -- From languages
  FROM languages
-- Order by name
ORDER BY name;

--SUBQUERY
-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
   FROM countries 
  WHERE region='Middle East')
-- Order by name
ORDER BY name;


-- count
-- Select statement
SELECT COUNT(*)
  -- From countries
  FROM countries
-- Where continent is Oceania
WHERE continent = 'Oceania';

-- anti join 
-- 5. Select fields (with aliases)
SELECT c1.code, c1.name, c2.basic_unit AS currency
  -- 1. From countries (alias as c1)
  FROM countries AS c1
  	-- 2. Join with currencies (alias as c2)
  	INNER JOIN currencies AS c2
    -- 3. Match on code
    USING (code)
-- 4. Where continent is Oceania
WHERE continent = 'Oceania';

-- NOT IN 
-- 3. Select fields
SELECT c1.name, c1.code
  -- 4. From Countries
	FROM countries AS c1
  -- 5. Where continent is Oceania
	WHERE c1.continent = 'Oceania'
  	-- 1. And code not in
    AND code NOT IN
  	-- 2. Subquery
    (SELECT code 
    FROM currencies);

-- UNION, EXCEPT
-- Select the city name
SELECT name
  -- Alias the table where city name resides
  FROM cities AS c1
  -- Choose only records matching the result of multiple set theory clauses
  WHERE country_code IN
(
    -- Select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- Get all additional (unique) values of the field from currencies AS c2  
    UNION
    SELECT c2.code
    FROM currencies AS c2
    -- Exclude those appearing in populations AS p
    EXCEPT
    SELECT p.country_code
    FROM populations AS p
);

-- SUBQUERY inside where -- example # 0
-- Select average life_expectancy
SELECT AVG(life_expectancy)
  -- From populations
  FROM populations
-- Where year is 2015
WHERE year = 2015;

-- SUBQUERY inside where -- example # 1
-- Applying formula example
-- Select fields
SELECT *
  -- From populations
  FROM populations
-- Where life_expectancy is greater than
WHERE life_expectancy > 1.15 *
  -- 1.15 * subquery
  (SELECT AVG(life_expectancy)
  FROM populations
  WHERE year = 2015)
  AND year = 2015;

  -- SUBQUERY inside where -- example # 2
  -- 2. Select fields
SELECT city.name, city.country_code, city.urbanarea_pop
  -- 3. From cities
  FROM cities AS city
-- 4. Where city name in the field of capital cities
WHERE city.name IN
  -- 1. Subquery
  (SELECT capital
   FROM countries)
ORDER BY urbanarea_pop DESC;


-- SUBQUERY Inside SELECT -- exmaple #0
SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;


-- SUBQUERY Inside SELECT -- exmaple #1
SELECT countries.name AS country,
  (SELECT COUNT(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;


-- SUBQUERY FROM -- example #0
-- Select fields (with aliases)
SELECT code, COUNT(name) AS lang_num
  -- From languages
  FROM languages
-- Group by code
GROUP BY code;

-- SUBQUERY FROM -- example #1
-- Select fields
SELECT local_name, subquery.lang_num
  -- From countries
  FROM countries, 
  	-- Subquery (alias as subquery)
  (SELECT code, COUNT(name) AS lang_num
    FROM languages
    GROUP BY code) as subquery
  -- Where codes match
  WHERE countries.code = subquery.code
-- Order by descending number of languages
ORDER BY lang_num DESC;

-- Advanced subquery - example #0
-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries 
  	-- Join to economies
  	INNER JOIN economies
    -- Match on code
  USING (code)
-- Where year is 2015
WHERE year = 2015;

-- Advanced subquery - example #1
-- Select the maximum inflation rate as max_inf
SELECT MAX(inflation_rate) AS max_inf
  -- Subquery using FROM (alias as subquery)
FROM (
      SELECT name, continent, inflation_rate
      FROM countries
      INNER JOIN economies
      USING (code)
      WHERE year = 2015) AS subquery
-- Group by continent
GROUP BY continent;

-- Advanced subquery - example #2
-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
	-- Join to economies
	INNER JOIN economies
	-- Match on code
	ON countries.code = economies.code
  -- Where year is 2015
  WHERE year = 2015
    -- And inflation rate in subquery (alias as subquery)
    AND inflation_rate IN (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
             INNER JOIN economies
             ON countries.code = economies.code
             WHERE year = 2015) AS subquery
      -- Group by continent
        GROUP BY continent);

-- Advanced subquery - example #3
-- Select fields
SELECT code, inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
  WHERE year = 2015 AND code NOT IN
  	-- Subquery
    (SELECT code
    FROM countries
    WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic'))
-- Order by inflation rate
ORDER BY inflation_rate;

-- SQL clause are subqueries most frequently found -> WHERE

-- Complete Ex # 0
-- Select fields
SELECT DISTINCT c.name, e.total_investment, e.imports
  -- From table (with alias)
  FROM countries AS c
    -- Join with table (with alias)
    LEFT JOIN economies AS e
      -- Match on code
    ON (c.code = e.code 
      -- and code in Subquery
        AND c.code IN (
          SELECT code 
    FROM languages
    WHERE official = 'true'
    ) )
  -- Where region and year are correct
  WHERE year = 2015 AND region = 'Central America'
-- Order by field
ORDER BY c.name;


-- Complete Ex # 1
-- Select fields
SELECT region, continent, AVG(fertility_rate) AS avg_fert_rate
  -- From left table
  FROM countries AS c
    -- Join to right table
    INNER JOIN populations AS p
      -- Match on join condition
    ON c.code = p.country_code
  -- Where specific records matching some condition
  WHERE year = 2015
-- Group appropriately
GROUP BY region, continent -- aggregated each
-- Order appropriately
ORDER BY avg_fert_rate; -- sorting 


-- Complete Ex # 2
-- Select fields
SELECT name, country_code, city_proper_pop, metroarea_pop,   
      -- Calculate city_perc
    city_proper_pop / metroarea_pop * 100 AS city_perc   -- From appropriate table
  FROM cities
  -- Where 
  WHERE name IN
    -- Subquery
  (SELECT capital
   FROM countries
   WHERE (continent = 'Europe'
      OR continent LIKE '%America'))
     AND metroarea_pop IS NOT NULL
-- Order appropriately
ORDER BY city_perc DESC
-- Limit amount
LIMIT 10;




