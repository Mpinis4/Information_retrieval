CREATE TABLE IF NOT EXISTS Public."Speaker" (
	Id SERIAL PRIMARY KEY,
	member_name varchar(150),
    sitting_date varchar(50),
    parliamentary_period varchar(150),
    parliamentary_session varchar(150),
    parliamentary_sitting varchar(150),
    political_party varchar(150),
    government varchar(250),
    member_region varchar(150),
    roles varchar(250),
    member_gender varchar(10),
    speech TEXT
);

SELECT* FROM Public."Speaker";

COPY Public."Speaker"(member_name, sitting_date, parliamentary_period, parliamentary_session, parliamentary_sitting, political_party, government, member_region, roles, member_gender, speech)
FROM 'C:\temp\Greek_Parliament_Proceedings_1989_2020.csv' 
WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

DELETE FROM Public."Speaker" WHERE member_name IS NULL;

ALTER TABLE "Speaker" 
ALTER COLUMN sitting_date TYPE DATE 
USING to_date(sitting_date, 'DD-MM-YYYY');