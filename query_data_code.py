"""
Command-line entry point that mirrors the logic from `code.ipynb`.

The script runs the Snowflake query that assembles the health-score dataset,
applies the same filtering/cleaning steps, and prints the resulting DataFrame.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from snowflake_conn import get_connection

# Display options match the notebook so terminal output stays readable.
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.precision", 10)


QUERY_CALLS = """
WITH 
m AS (
    SELECT
        customer_id,
        month_ending,
        total_mrr,
        is_lost,
        MAX(CASE WHEN is_new THEN month_ending END) OVER (PARTITION BY customer_id) AS last_new_month
    FROM prod.finance.mrr_by_location
    QUALIFY month_ending >= last_new_month
    ),

lifetime_stats AS (
    SELECT
        customer_id,
        SUM(total_mrr) AS lifetime_value,
        COUNT(*) AS lifetime_months,
        min(month_ending) AS first_month
    FROM m
    GROUP BY customer_id
),


starting_mrr AS (
    SELECT
        month_ending,
        mrr.customer_id,
        total_mrr
    FROM prod.finance.mrr_by_location mrr
    join lifetime_stats life on mrr.CUSTOMER_ID = life.CUSTOMER_ID and life.first_month = mrr.month_ending
),

deduped_monthly_bundles AS (
    SELECT
        month_ending,
        finance_id,
        product_name
    FROM prod.product.churn_by_product cp
    join lifetime_stats life on cp.finance_id = life.customer_id and life.first_month = cp.month_ending
    WHERE product_family = 'Core Products'
    
),

starting_bundle AS (
    SELECT
        month_ending,
        finance_id,
        product_name
    FROM deduped_monthly_bundles
    QUALIFY MIN(month_ending) OVER (PARTITION BY finance_id) = month_ending
),

current_bundle AS (
    SELECT
        month_ending,
        finance_id,
        product_name
    FROM deduped_monthly_bundles
    QUALIFY MAX(month_ending) OVER (PARTITION BY finance_id) = month_ending
),

churn_list AS (
    SELECT
        customer_id,
        month_ending,
        is_lost
    FROM m
    WHERE is_lost = TRUE)
    
,

support_case_week AS (
    SELECT
        sup.created_at,
        sup.interaction_number,
        sup.location_id,
        sup.billing_start_date,
        COALESCE(sup.primary, sup.reason_topic) AS support_reason,
        sup.subject,
        FLOOR(DATEDIFF(day, sup.billing_start_date::date, sup.created_at::date) / 7) AS week
    FROM prod.support.support_interaction_v3 AS sup
    LEFT JOIN prod.sales.account_v2 AS acc
        ON sup.location_id = acc.location_id
    WHERE sup.location_id IS NOT NULL
),

weekly_case_count AS (
    SELECT
        location_id,
        COUNT_IF(week = 1) AS week_1_cases,
        COUNT_IF(week = 2) AS week_2_cases,
        COUNT_IF(week = 3) AS week_3_cases,
        COUNT_IF(week = 4) AS week_4_cases,
        COUNT_IF(week = 5) AS week_5_cases,
        COUNT_IF(week = 6) AS week_6_cases,
        COUNT_IF(week = 7) AS week_7_cases,
        COUNT_IF(week = 8) AS week_8_cases,
        COUNT_IF(week = 9) AS week_9_cases,
        COUNT_IF(week = 10) AS week_10_cases,
        COUNT_IF(week = 11) AS week_11_cases,
        COUNT_IF(week = 12) AS week_12_cases
    FROM support_case_week
    GROUP BY location_id
),

support_reasons_pivot AS (
    SELECT
        location_id,
        SUM(CASE WHEN support_reason ILIKE '%Billing%' THEN 1 ELSE 0 END) AS reason_billing,
        SUM(CASE WHEN support_reason ILIKE '%Phone System%' THEN 1 ELSE 0 END) AS reason_phone_system,
        SUM(CASE WHEN support_reason ILIKE '%Data Sync%' THEN 1 ELSE 0 END) AS reason_data_sync
    FROM support_case_week
    WHERE week IN (1, 2, 3, 4)
    GROUP BY location_id
),


stage AS (
    SELECT
        l.location_id,
        a.finance_number AS customer_id,
        l.report_month,
        MIN(l.report_month) OVER (PARTITION BY a.finance_number) AS first_usage_month,
        m.last_new_month,
        l.inbound_sms_count,
        l.outbound_sms_count,
        l.inbound_call_count,
        l.outbound_call_count,
        l.missed_call_sms_count,
        l.missed_text_sms_count,
        l.automated_sms_sent_count,
        l.has_appointment_reminders,
        l.has_missed_call_text,
        l.has_missed_text_text
    FROM prod.product.location_feature_access_and_usage_by_month AS l
    JOIN prod.sales.account_v2 AS a
        ON l.location_id = a.location_id
    JOIN m
        ON a.finance_number = m.customer_id
),

first_month_usage AS (
    SELECT DISTINCT *
    FROM stage
    WHERE report_month = first_usage_month
      AND DATEDIFF('month', DATE_TRUNC('month', last_new_month), DATE_TRUNC('month', first_usage_month)) <= 1
),

integrations AS (
    SELECT
        location_id,
        integrations
    FROM staging.weave_integrations.stg_location_integrations
),

base AS (
    SELECT
        account_id,
        business_segment_by_hierarchy
    FROM prod.sales.opportunity_stages_timeline
    GROUP BY 1, 2
),

business_segment AS (
    SELECT
        location_id,
        case
            when number_of_locations_in_hierarchy_active = 1 then 'Single' when
               number_of_locations_in_hierarchy_active < 10
                then 'Multi'
            when number_of_locations_in_hierarchy_active < 50 then 'Mid Market' when
                number_of_locations_in_hierarchy_active >= 50
                then 'Enterprise'
        end as business_segment_by_hierarchy_active,
        case
            when number_of_locations_in_hierarchy_total = 1 then 'Single' when
               number_of_locations_in_hierarchy_total < 10
                then 'Multi'
            when number_of_locations_in_hierarchy_total < 50 then 'Mid Market' when
                number_of_locations_in_hierarchy_total >= 50
                then 'Enterprise'
        end as business_segment_by_hierarchy_total
    FROM prod.sales.account_v2 
    where location_id IS NOT NULL
),
swat as (select 
location_id,
case_id,
case_status,
created_date,
cancel_summary,
cancellation_reason,
cancellation_subreason
from prod.cs.swat_case
qualify  MAX(created_date) OVER (PARTITION BY location_id) = created_date
)

SELECT
    loc.slug,
    loc.finance_id,
    loc.location_id,
    loc.location_name,
    bs.business_segment_by_hierarchy_active,
    bs.business_segment_by_hierarchy_total,
    loc.core_industry,
    loc.core_sub_industry,
    loc.specialty,
    acc.is_parent_account,
    loc.is_satellite,
    loc.location_type,
    acc.billing_start_date,
    smrr.total_mrr::NUMBER(38, 2) AS starting_mrr,
    life.lifetime_value,
    life.first_month AS first_mrr_month,
    COALESCE(int.integrations, loc.practice_management_software) AS practice_management_software,
    int.integrations,
    acc.onboarding_handoff AS onboarding_handoff_date,
    sbun.product_name AS starting_bundle,
    cbun.product_name AS end_bundle,
    CASE WHEN churn.is_lost THEN 'Churn' ELSE 'Retain' END AS churn_or_retain,
    churn.month_ending AS churn_month,
    CASE WHEN churn.is_lost THEN life.lifetime_months - 1 ELSE life.lifetime_months END AS lifetime_months,
    -- ✅ Weekly support case counts
    week.week_1_cases,
    week.week_2_cases,
    week.week_3_cases,
    week.week_4_cases,
    week.week_5_cases,
    week.week_6_cases,
    week.week_7_cases,
    week.week_8_cases,
    week.week_9_cases,
    week.week_10_cases,
    week.week_11_cases,
    week.week_12_cases,
    reasons_pivot.reason_billing,
    reasons_pivot.reason_phone_system,
    reasons_pivot.reason_data_sync,
    -- ✅ First-month usage metrics
    usage.report_month AS first_usage_report_month,
    usage.inbound_sms_count AS first_inbound_sms_count,
    usage.outbound_sms_count AS first_outbound_sms_count,
    usage.inbound_call_count AS first_inbound_call_count,
    usage.outbound_call_count AS first_outbound_call_count,
    usage.missed_call_sms_count AS first_missed_call_sms_count,
    usage.missed_text_sms_count AS first_missed_text_sms_count,
    usage.automated_sms_sent_count AS first_automated_sms_count,
    usage.has_appointment_reminders AS first_appointment_reminders_access,
    usage.has_missed_call_text AS first_missed_call_text,
    usage.has_missed_text_text AS first_missed_text_text,
    sw.case_id as swat_case_id,
    sw.case_status as swat_case_status,
    sw.created_date as swat_case_created_date,
    sw.cancel_summary as swat_cancel_summary,
    sw.cancellation_reason as swat_cancellation_reason,
    sw.cancellation_subreason as swat_cancellation_subreason
FROM prod.common.location AS loc
LEFT JOIN prod.sales.account_v2 AS acc
    ON acc.slug = loc.slug
LEFT JOIN starting_mrr AS smrr
    ON smrr.customer_id = loc.finance_id
LEFT JOIN lifetime_stats AS life
    ON loc.finance_id = life.customer_id
LEFT JOIN starting_bundle AS sbun
    ON sbun.finance_id = loc.finance_id
LEFT JOIN current_bundle AS cbun
    ON cbun.finance_id = loc.finance_id
LEFT JOIN churn_list AS churn
    ON churn.customer_id = loc.finance_id
LEFT JOIN weekly_case_count AS week
    ON loc.location_id = week.location_id
LEFT JOIN integrations AS int
    ON int.location_id = loc.location_id
LEFT JOIN support_reasons_pivot AS reasons_pivot
    ON loc.location_id = reasons_pivot.location_id
LEFT JOIN first_month_usage AS usage
    ON loc.location_id = usage.location_id
LEFT JOIN business_segment AS bs
    ON loc.location_id = bs.location_id
LEFT JOIN swat AS sw
    ON loc.location_id = sw.location_id
   AND COALESCE(churn.is_lost, FALSE)
"""


def run_query(query: str) -> pd.DataFrame:
    """Execute SQL against Snowflake and return the data as a DataFrame."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)
    finally:
        cursor.close()
        conn.close()


def _case_insensitive_lookup(
    df: pd.DataFrame, targets: Iterable[str]
) -> Mapping[str, str]:
    """
    Build a mapping from requested column names to actual DataFrame columns.

    Handles the notebook's title-cased column references while Snowflake
    returns uppercase names by default.
    """
    index = {col.lower(): col for col in df.columns}
    resolved = {}
    for target in targets:
        actual = index.get(target.lower())
        if actual is not None:
            resolved[target] = actual
    return resolved


def build_final_dataframe(calls_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the notebook filtering/cleaning steps."""
    churn_label = _case_insensitive_lookup(calls_df, ["CHURN_OR_RETAIN"])[
        "CHURN_OR_RETAIN"
    ]
    lifetime_months_col = _case_insensitive_lookup(
        calls_df, ["LIFETIME_MONTHS"]
    )["LIFETIME_MONTHS"]

    retain_mask = (calls_df[churn_label] == "Retain") & (
        calls_df[lifetime_months_col] > 12
    )
    churn_mask = (calls_df[churn_label] == "Churn") & (
        calls_df[lifetime_months_col].between(1, 6)
    )

    filtered = pd.concat(
        [calls_df.loc[retain_mask], calls_df.loc[churn_mask]],
        ignore_index=True,
    )

    first_mrr_map = _case_insensitive_lookup(filtered, ["FIRST_MRR_MONTH"])
    if first_mrr_map:
        first_mrr_col = first_mrr_map["FIRST_MRR_MONTH"]
        filtered[first_mrr_col] = pd.to_datetime(
            filtered[first_mrr_col], errors="coerce"
        )
        filtered = filtered[
            filtered[first_mrr_col] >= datetime(2024, 1, 1)
        ]

    required_cols = _case_insensitive_lookup(
        filtered, ["SLUG", "FINANCE_ID", "LOCATION_ID", "LOCATION_NAME"]
    ).values()
    if required_cols:
        filtered = filtered.dropna(subset=list(required_cols))

    filtered = filtered.dropna(axis=1, how="all")

    week_cols = _case_insensitive_lookup(
        filtered, [f"WEEK_{i}_CASES" for i in range(1, 13)]
    ).values()
    if week_cols:
        filtered = filtered.dropna(subset=list(week_cols), how="any")

    usage_cols_required = _case_insensitive_lookup(
        filtered,
        [
            "FIRST_INBOUND_SMS_COUNT",
            "FIRST_OUTBOUND_SMS_COUNT",
            "FIRST_INBOUND_CALL_COUNT",
            "FIRST_OUTBOUND_CALL_COUNT",
        ],
    ).values()
    if usage_cols_required:
        filtered = filtered.dropna(subset=list(usage_cols_required), how="any")

    return filtered


def main() -> None:
    calls_df = run_query(QUERY_CALLS)
    final_df = build_final_dataframe(calls_df)
    churn_col_map = _case_insensitive_lookup(final_df, ["CHURN_OR_RETAIN"])
    churn_col = churn_col_map.get("CHURN_OR_RETAIN")
    churn_df = final_df
    if churn_col:
        churn_df = final_df[final_df[churn_col] == "Churn"].copy()

    base_dir = Path(__file__).resolve().parent
    churn_path = base_dir / "0-6 month churned customer since 2024.csv"
    combined_path = base_dir / "0-6 month churn and 12+ month retain since 2024.csv"

    churn_df.to_csv(churn_path, index=False)
    final_df.to_csv(combined_path, index=False)
    print(final_df)
    print(
        f"\nSaved {len(churn_df):,} churn rows to {churn_path}"
        f"\nSaved {len(final_df):,} total rows (churn + 12+ month retain) to {combined_path}"
    )


if __name__ == "__main__":
    main()
