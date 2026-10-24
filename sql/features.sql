-- Claim features: the single definition used by batch training AND the online scoring API.
-- Both paths run this exact query, so training/serving skew is ruled out by construction
-- (tests/test_features.py checks it anyway).
--
-- Entity aggregates look strictly backwards: every window frame ends one second before the
-- claim was reported, so a claim never sees itself or anything that happened after it.
-- The query reads from a relation named `claims`; callers register that view first.
-- Plain DuckDB SQL: on Databricks the same logic runs as Spark SQL over Delta tables.

WITH base AS (
    SELECT
        *,
        repair_estimate / nullif(vehicle_value, 0) AS repair_to_value
    FROM claims
)
SELECT
    claim_id,
    month,
    report_ts,

    -- what happened
    loss_type,
    province,
    ln(1 + total_claimed)                                         AS log_total_claimed,
    ln(1 + injury_reserve)                                        AS log_injury_reserve,
    ln(1 + repair_estimate)                                       AS log_repair_estimate,
    repair_to_value,
    vehicle_value,
    vehicle_age,
    CAST(total_loss AS INTEGER)                                   AS total_loss,
    loss_hour,
    CASE WHEN loss_hour >= 19 OR loss_hour < 5 THEN 1 ELSE 0 END  AS night_loss,
    n_occupants,
    n_injured,
    n_injured / greatest(n_occupants, 1)                          AS injured_share,
    CAST(soft_tissue_only AS INTEGER)                             AS soft_tissue_only,
    CAST(police_report AS INTEGER)                                AS police_report,
    witnesses,
    CASE WHEN rep_id IS NOT NULL THEN 1 ELSE 0 END                AS represented,
    tow_distance_km,
    storage_days,
    tow_fee,

    -- who is claiming
    policy_tenure_days,
    report_delay_days,
    claimant_age,
    prior_claims_3y,

    -- network: how often these identifiers and providers showed up before this claim
    count(*) OVER phone_365                                       AS phone_prior_claims_365d,
    count(*) OVER address_365                                     AS address_prior_claims_365d,
    CASE WHEN body_shop_id IS NULL THEN 0
         ELSE count(*) OVER shop_90 END                           AS shop_prior_claims_90d,
    CASE WHEN body_shop_id IS NULL THEN NULL
         ELSE avg(repair_to_value) OVER shop_180 END              AS shop_avg_repair_ratio_180d,
    CASE WHEN clinic_id IS NULL THEN 0
         ELSE count(*) OVER clinic_90 END                         AS clinic_prior_claims_90d,
    CASE WHEN rep_id IS NULL OR clinic_id IS NULL THEN 0
         ELSE count(*) OVER rep_clinic_365 END                    AS rep_clinic_prior_claims_365d,
    CASE WHEN tow_company_id IS NULL THEN 0
         ELSE count(*) OVER tow_90 END                            AS tow_prior_claims_90d,
    CASE WHEN tow_company_id IS NULL THEN NULL
         ELSE avg(storage_days) OVER tow_180 END                  AS tow_avg_storage_days_180d
FROM base
WINDOW
    phone_365      AS (PARTITION BY phone_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING),
    address_365    AS (PARTITION BY address_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING),
    shop_90        AS (PARTITION BY body_shop_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING),
    shop_180       AS (PARTITION BY body_shop_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 180 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING),
    clinic_90      AS (PARTITION BY clinic_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING),
    rep_clinic_365 AS (PARTITION BY rep_id, clinic_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING),
    tow_90         AS (PARTITION BY tow_company_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING),
    tow_180        AS (PARTITION BY tow_company_id ORDER BY report_ts
                       RANGE BETWEEN INTERVAL 180 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING)
ORDER BY report_ts, claim_id
