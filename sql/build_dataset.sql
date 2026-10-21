-- Build the modeling dataset: one row per complaint x component group.
--
-- Question: when NHTSA receives a vehicle complaint, will the same
-- make / model / model year get a recall for the same component group
-- within the next 365 days?
--
-- Every feature uses only information available on the day the complaint
-- was received, so the model could be used in real time.

------------------------------------------------------------------------------
-- 1. Harmonize component names.
--    NHTSA renamed some top-level components over time (for example
--    "ENGINE AND ENGINE COOLING" became "ENGINE"), and complaints and recalls
--    do not always use the same version. Map both to one group name.
------------------------------------------------------------------------------
DROP TABLE IF EXISTS component_map;
CREATE TABLE component_map (raw_name TEXT PRIMARY KEY, comp_group TEXT);
INSERT INTO component_map VALUES
    ('ENGINE AND ENGINE COOLING',  'ENGINE'),
    ('FUEL/PROPULSION SYSTEM',     'FUEL/PROPULSION SYSTEM'),
    ('FUEL SYSTEM, GASOLINE',      'FUEL/PROPULSION SYSTEM'),
    ('FUEL SYSTEM, DIESEL',        'FUEL/PROPULSION SYSTEM'),
    ('FUEL SYSTEM, OTHER',         'FUEL/PROPULSION SYSTEM'),
    ('HYBRID PROPULSION SYSTEM',   'FUEL/PROPULSION SYSTEM'),
    ('SERVICE BRAKES, HYDRAULIC',  'SERVICE BRAKES'),
    ('SERVICE BRAKES, AIR',        'SERVICE BRAKES'),
    ('SERVICE BRAKES, ELECTRIC',   'SERVICE BRAKES'),
    ('VISIBILITY/WIPER',           'VISIBILITY');

------------------------------------------------------------------------------
-- 2. Clean complaints: vehicles only, parsed dates, component group,
--    one row per complaint (ODINO) x component group.
------------------------------------------------------------------------------
DROP TABLE IF EXISTS cmpl_clean;
CREATE TABLE cmpl_clean AS
WITH base AS (
    SELECT
        c.*,
        UPPER(TRIM(CASE WHEN instr(c.COMPDESC, ':') > 0
                        THEN substr(c.COMPDESC, 1, instr(c.COMPDESC, ':') - 1)
                        ELSE c.COMPDESC END)) AS top_comp,
        julianday(substr(c.LDATE, 1, 4) || '-' || substr(c.LDATE, 5, 2) || '-' || substr(c.LDATE, 7, 2)) AS recv_jd
    FROM complaints c
    WHERE c.PROD_TYPE = 'V'
      AND c.YEARTXT BETWEEN '1990' AND '2026'      -- drops 9999 = unknown
      AND c.LDATE BETWEEN '20150101' AND '20241231'
)
SELECT
    ODINO,
    MAKETXT, MODELTXT, CAST(YEARTXT AS INTEGER) AS model_year,
    COALESCE(m.comp_group, b.top_comp)          AS comp_group,
    MIN(LDATE)                                  AS recv_date,
    MIN(recv_jd)                                AS recv_jd,
    MAX(CRASH = 'Y')                            AS crash,
    MAX(FIRE = 'Y')                             AS fire,
    MAX(CAST(INJURED AS INTEGER))               AS injured,
    MAX(CAST(DEATHS AS INTEGER))                AS deaths,
    MAX(MEDICAL_ATTN = 'Y')                     AS medical_attn,
    MAX(VEHICLES_TOWED_YN = 'Y')                AS towed,
    MAX(POLICE_RPT_YN = 'Y')                    AS police_report,
    MAX(CAST(MILES AS INTEGER))                 AS miles,
    MAX(CAST(VEH_SPEED AS INTEGER))             AS speed,
    MAX(CMPL_TYPE)                              AS cmpl_type,
    MAX(FUEL_TYPE)                              AS fuel_type,
    MAX(CDESCR)                                 AS text
FROM base b
LEFT JOIN component_map m ON m.raw_name = b.top_comp
WHERE b.top_comp NOT IN ('UNKNOWN OR OTHER', 'EQUIPMENT', '')
GROUP BY ODINO, MAKETXT, MODELTXT, YEARTXT, COALESCE(m.comp_group, b.top_comp);

CREATE INDEX idx_cc_key ON cmpl_clean (MAKETXT, MODELTXT, model_year, comp_group);

------------------------------------------------------------------------------
-- 3. Clean recalls: vehicle recalls only, one row per campaign x vehicle x
--    component group, with the date NHTSA received the defect report.
------------------------------------------------------------------------------
DROP TABLE IF EXISTS rcl_clean;
CREATE TABLE rcl_clean AS
WITH base AS (
    SELECT
        r.*,
        UPPER(TRIM(CASE WHEN instr(r.COMPNAME, ':') > 0
                        THEN substr(r.COMPNAME, 1, instr(r.COMPNAME, ':') - 1)
                        ELSE r.COMPNAME END)) AS top_comp
    FROM recalls r
    WHERE r.RCLTYPECD = 'V' AND r.RCDATE <> ''
)
SELECT DISTINCT
    CAMPNO,
    MAKETXT, MODELTXT, CAST(YEARTXT AS INTEGER) AS model_year,
    COALESCE(m.comp_group, b.top_comp) AS comp_group,
    julianday(substr(RCDATE, 1, 4) || '-' || substr(RCDATE, 5, 2) || '-' || substr(RCDATE, 7, 2)) AS rcl_jd
FROM base b
LEFT JOIN component_map m ON m.raw_name = b.top_comp;

CREATE INDEX idx_rc_key ON rcl_clean (MAKETXT, MODELTXT, model_year, comp_group, rcl_jd);

------------------------------------------------------------------------------
-- 4. Final dataset: label + history features.
------------------------------------------------------------------------------
DROP TABLE IF EXISTS model_data;
CREATE TABLE model_data AS
WITH history AS (
    SELECT
        cc.*,
        -- How many complaints about this same vehicle + component arrived in
        -- the 90 / 365 days before this one (a "complaint spike" signal)
        COUNT(*) OVER w90  AS prior_cmpl_90d,
        COUNT(*) OVER w365 AS prior_cmpl_365d
    FROM cmpl_clean cc
    WINDOW
        w90  AS (PARTITION BY MAKETXT, MODELTXT, model_year, comp_group
                 ORDER BY recv_jd RANGE BETWEEN 90 PRECEDING AND 1 PRECEDING),
        w365 AS (PARTITION BY MAKETXT, MODELTXT, model_year, comp_group
                 ORDER BY recv_jd RANGE BETWEEN 365 PRECEDING AND 1 PRECEDING)
),
vehicle_history AS (
    -- All complaints about this vehicle (any component) in the prior 90 days
    SELECT ODINO, comp_group,
           COUNT(*) OVER (PARTITION BY MAKETXT, MODELTXT, model_year
                          ORDER BY recv_jd RANGE BETWEEN 90 PRECEDING AND 1 PRECEDING) AS prior_vehicle_cmpl_90d
    FROM cmpl_clean
)
SELECT
    h.ODINO, h.MAKETXT AS make, h.MODELTXT AS model, h.model_year, h.comp_group,
    h.recv_date,
    CAST(substr(h.recv_date, 1, 4) AS INTEGER) AS recv_year,
    CAST(substr(h.recv_date, 1, 4) AS INTEGER) - h.model_year AS vehicle_age,
    h.crash, h.fire, h.injured, h.deaths, h.medical_attn, h.towed, h.police_report,
    h.miles, h.speed, h.cmpl_type, h.fuel_type,
    h.prior_cmpl_90d, h.prior_cmpl_365d, vh.prior_vehicle_cmpl_90d,
    -- Was there already a recall for this vehicle + component in the past 2 years?
    EXISTS (
        SELECT 1 FROM rcl_clean r
        WHERE r.MAKETXT = h.MAKETXT AND r.MODELTXT = h.MODELTXT
          AND r.model_year = h.model_year AND r.comp_group = h.comp_group
          AND r.rcl_jd BETWEEN h.recv_jd - 730 AND h.recv_jd
    ) AS prior_recall_2y,
    -- LABEL: a new recall for this vehicle + component within 365 days AFTER
    -- the complaint was received
    EXISTS (
        SELECT 1 FROM rcl_clean r
        WHERE r.MAKETXT = h.MAKETXT AND r.MODELTXT = h.MODELTXT
          AND r.model_year = h.model_year AND r.comp_group = h.comp_group
          AND r.rcl_jd > h.recv_jd AND r.rcl_jd <= h.recv_jd + 365
    ) AS recall_within_1y,
    h.text
FROM history h
JOIN vehicle_history vh ON vh.ODINO = h.ODINO AND vh.comp_group = h.comp_group;
