-- The app rebuilds these tables as part of ETL. This file is a reference schema.
CREATE TABLE dim_patient_profile(profile_key INTEGER PRIMARY KEY, gender TEXT UNIQUE);
CREATE TABLE dim_age(age_key INTEGER PRIMARY KEY, age_band TEXT UNIQUE);
CREATE TABLE dim_glucose(glucose_key INTEGER PRIMARY KEY, glucose_band TEXT UNIQUE);
CREATE TABLE dim_bmi(bmi_key INTEGER PRIMARY KEY, bmi_band TEXT UNIQUE);
CREATE TABLE dim_outcome(outcome_key INTEGER PRIMARY KEY, outcome_label TEXT UNIQUE);
CREATE TABLE fact_patient(
 fact_key INTEGER PRIMARY KEY, profile_key INTEGER, age_key INTEGER, glucose_key INTEGER, bmi_key INTEGER, outcome_key INTEGER,
 glucose REAL, blood_pressure REAL, skin_thickness REAL, insulin REAL, bmi REAL, pedigree REAL, age INTEGER, record_count INTEGER DEFAULT 1,
 FOREIGN KEY(profile_key) REFERENCES dim_patient_profile(profile_key), FOREIGN KEY(age_key) REFERENCES dim_age(age_key),
 FOREIGN KEY(glucose_key) REFERENCES dim_glucose(glucose_key), FOREIGN KEY(bmi_key) REFERENCES dim_bmi(bmi_key), FOREIGN KEY(outcome_key) REFERENCES dim_outcome(outcome_key)
);
