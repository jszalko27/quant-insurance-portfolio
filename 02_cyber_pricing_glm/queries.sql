-- name: book_by_year
-- Frequency, severity and loss ratio by accident year
SELECT p.year,
       COUNT(*)                                   AS policies,
       ROUND(SUM(p.exposure), 0)                  AS exposure_years,
       COALESCE(SUM(c.n_claims), 0)               AS claims,
       ROUND(1.0 * SUM(c.n_claims) / SUM(p.exposure), 4) AS frequency,
       ROUND(SUM(c.incurred) / SUM(c.n_claims), 0) AS severity,
       ROUND(SUM(p.premium), 0)                   AS premium,
       ROUND(COALESCE(SUM(c.incurred), 0) / SUM(p.premium), 3) AS loss_ratio
FROM policies p
LEFT JOIN (SELECT policy_id, COUNT(*) AS n_claims, SUM(amount) AS incurred
           FROM claims GROUP BY policy_id) c USING (policy_id)
GROUP BY p.year ORDER BY p.year;

-- name: loss_ratio_by_industry
SELECT p.industry,
       COUNT(*) AS policies,
       ROUND(1.0 * COALESCE(SUM(c.n_claims), 0) / SUM(p.exposure), 4) AS frequency,
       ROUND(COALESCE(SUM(c.incurred), 0) / SUM(p.premium), 3) AS loss_ratio,
       ROUND(100.0 * SUM(p.premium) / SUM(SUM(p.premium)) OVER (), 1) AS pct_of_premium
FROM policies p
LEFT JOIN (SELECT policy_id, COUNT(*) AS n_claims, SUM(amount) AS incurred
           FROM claims GROUP BY policy_id) c USING (policy_id)
GROUP BY p.industry ORDER BY loss_ratio DESC;

-- name: loss_ratio_by_controls
-- The current rating plan gives no credit for controls: does the loss ratio show it should?
SELECT (p.mfa + p.edr + p.backups + p.patching) AS controls_in_place,
       COUNT(*) AS policies,
       ROUND(1.0 * COALESCE(SUM(c.n_claims), 0) / SUM(p.exposure), 4) AS frequency,
       ROUND(COALESCE(SUM(c.incurred), 0) / SUM(p.premium), 3) AS loss_ratio
FROM policies p
LEFT JOIN (SELECT policy_id, COUNT(*) AS n_claims, SUM(amount) AS incurred
           FROM claims GROUP BY policy_id) c USING (policy_id)
GROUP BY controls_in_place ORDER BY controls_in_place;

-- name: loss_ratio_by_revenue_band
SELECT CASE WHEN revenue < 10e6 THEN '1: <$10M'
            WHEN revenue < 50e6 THEN '2: $10M-$50M'
            WHEN revenue < 250e6 THEN '3: $50M-$250M'
            ELSE '4: $250M-$1B' END AS revenue_band,
       COUNT(*) AS policies,
       ROUND(1.0 * COALESCE(SUM(c.n_claims), 0) / SUM(p.exposure), 4) AS frequency,
       ROUND(COALESCE(SUM(c.incurred), 0) / NULLIF(SUM(c.n_claims), 0), 0) AS severity,
       ROUND(COALESCE(SUM(c.incurred), 0) / SUM(p.premium), 3) AS loss_ratio
FROM policies p
LEFT JOIN (SELECT policy_id, COUNT(*) AS n_claims, SUM(amount) AS incurred
           FROM claims GROUP BY policy_id) c USING (policy_id)
GROUP BY revenue_band ORDER BY revenue_band;

-- name: peril_mix
SELECT peril,
       COUNT(*) AS claims,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_claims,
       ROUND(AVG(amount), 0) AS avg_claim,
       ROUND(100.0 * SUM(amount) / SUM(SUM(amount)) OVER (), 1) AS pct_of_dollars
FROM claims GROUP BY peril ORDER BY pct_of_dollars DESC;
