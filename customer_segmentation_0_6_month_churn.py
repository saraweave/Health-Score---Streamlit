"""
Segment 0-6 month churned customers into persona-style groups using OpenAI embeddings
and GPT-driven summaries. This module mirrors the workflow from
customer_segmentation_reasoning.ipynb and adapts it to the data stored in
`0-6 month churned customer since 2024.csv`.

Running the script will:
1. load and clean the churn dataset
2. create embeddings for a blended job/reason signal
3. cluster customers with KMeans
4. optionally generate GPT-written summaries for each cluster
5. persist the labeled dataset and a PCA visualisation for manual review
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ---------------------------- Configuration --------------------------------- #

DEFAULT_DATA_PATH = Path("0-6 month churned customer since 2024.csv")
DEFAULT_OUTPUT_PATH = Path("0_6_month_churn_reason_segments_2024.csv")
DEFAULT_PLOT_PATH = Path("0_6_month_churn_cluster_plot.png")

RANDOM_STATE = 42
JOB_EMBED_DIM = 12
REASON_EMBED_DIM = 24
DEFAULT_SEGMENTS = 3
EMBEDDING_MODEL = "text-embedding-3-small"
SUMMARY_MODEL = "gpt-4o-mini"

TEXT_FEATURES = [
    "CORE_INDUSTRY",
    "PRACTICE_MANAGEMENT_SOFTWARE",
    "INTEGRATIONS",
    "STARTING_BUNDLE",
    "LIFETIME_MONTHS",
]

SUMMARY_COLUMNS = [
    "CORE_INDUSTRY",
    "LIFETIME_MONTHS",
    "PRACTICE_MANAGEMENT_SOFTWARE",
    "STARTING_BUNDLE",
    "SWAT_CANCEL_SUMMARY",
]


# ------------------------------- Utilities ---------------------------------- #


def load_api_key() -> str:
    """Get the OpenAI API key from top_secret module or environment variable."""
    try:
        from top_secret import my_sk  # type: ignore
    except ImportError:
        my_sk = os.getenv("OPENAI_API_KEY")

    if not my_sk:
        raise RuntimeError(
            "OpenAI API key not provided. Supply top_secret.my_sk or set OPENAI_API_KEY."
        )
    return my_sk


def read_csv_to_dataframe(path: Path) -> pd.DataFrame:
    """Load CSV data using the csv module to avoid pandas parser segfaults."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []

    return pd.DataFrame(rows, columns=columns)


def normalise_text_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure text columns exist and replace blank values with 'Unknown'."""
    for column in columns:
        if column not in df.columns:
            df[column] = "Unknown"
        else:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
            )
    return df


def compose_feature_strings(df: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """Generate concatenated feature strings for embeddings."""
    normalised = normalise_text_features(df.copy(), features)
    return normalised.astype(str).agg(" ".join, axis=1).tolist()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table for GPT consumption."""
    if df.empty:
        return "| No data available |\n| ------------------ |"

    header = f"| {' | '.join(df.columns)} |\n"
    separator = f"| {' | '.join('-' * len(col) for col in df.columns)} |\n"
    rows = [
        f"| {' | '.join(str(cell) if cell == cell else '' for cell in row)} |"
        for row in df.itertuples(index=False, name=None)
    ]
    return header + separator + "\n".join(rows)


def build_summary_prompt(markdown_table: str) -> str:
    """Compose the GPT prompt for summarising a cluster."""
    return f"""You are a data scientist specialising in customer segmentation for churn analysis.
Below is a list of 0-6 month churned Weave customers. Use it to craft a single predominant profile. I really need to know all the detail weave product features, such as form, auto reminder or data sync.

Please include:
1. Core Industry: top 2 industries that best represent the table.
2. Practice Management Software: I need all the detail pms name(or state 'No integration info').
3. Lifetime in Months: describe the most common tenure buckets.
4. Bundles: highlight up to 5 starting bundles that dominate.
5. Key issue features: list of the product/feature, such as Form, auto reminder, appointment reminder or data sync with their integration issues
6. key onboarding issue: The pain points of onboarding and customer support. 
### Survey Responses
{markdown_table}

### Instructions
- Keep each section to one short sentence or bullet.
- Use clear, concise language under 100 words total.
- Return only the five items above.
"""


def generate_segment_summary(client: OpenAI, df_segment: pd.DataFrame) -> str:
    """Request a GPT summary for a given cluster."""
    markdown_table = dataframe_to_markdown(df_segment)
    prompt = build_summary_prompt(markdown_table)

    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def compute_embeddings(client: OpenAI, df: pd.DataFrame) -> pd.DataFrame:
    """Generate concatenated job and reason embeddings for clustering."""
    if df.empty:
        raise ValueError("No rows available for embedding generation.")

    job_inputs = compose_feature_strings(df, TEXT_FEATURES)
    reason_inputs = df["SWAT_CANCEL_SUMMARY"].astype(str).tolist()

    job_response = client.embeddings.create(
        input=job_inputs,
        model=EMBEDDING_MODEL,
        dimensions=JOB_EMBED_DIM,
    )
    reason_response = client.embeddings.create(
        input=reason_inputs,
        model=EMBEDDING_MODEL,
        dimensions=REASON_EMBED_DIM,
    )

    embeddings = [
        np.concatenate([job.embedding, reason.embedding])
        for job, reason in zip(job_response.data, reason_response.data, strict=True)
    ]

    column_names = [
        f"job_embedding_{i+1}" for i in range(JOB_EMBED_DIM)
    ] + [f"reason_embedding_{i+1}" for i in range(REASON_EMBED_DIM)]

    return pd.DataFrame(embeddings, columns=column_names)


def cluster_embeddings(embeddings: pd.DataFrame, n_clusters: int) -> tuple[KMeans, np.ndarray]:
    """Cluster embeddings with KMeans."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if len(embeddings) < n_clusters:
        n_clusters = len(embeddings)

    if n_clusters == 0:
        raise ValueError("No embeddings available for clustering.")

    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = model.fit_predict(embeddings)
    return model, labels


def save_cluster_plot(
    embeddings: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    title: str = "0-6 Month Churn Clusters",
) -> None:
    """Reduce embeddings to 2D and save a scatter plot."""
    if len(embeddings) <= 1:
        return

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            components[mask, 0],
            components[mask, 1],
            label=f"Segment {label + 1}",
        )

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the churn dataset CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to store the labeled segments CSV.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Location to write the PCA scatter plot PNG.",
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=DEFAULT_SEGMENTS,
        help="Number of clusters to compute (auto-adjusts if needed).",
    )
    parser.add_argument(
        "--skip-gpt",
        action="store_true",
        help="Skip GPT summaries (only produces clustering results).",
    )
    return parser.parse_args()


# ------------------------------- Entry Point -------------------------------- #


def main() -> None:
    args = parse_args()

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    plt.style.use("seaborn-v0_8")

    df_raw = read_csv_to_dataframe(args.data_path)
    if df_raw.empty:
        raise RuntimeError(f"No data found in {args.data_path}")

    df_filtered = df_raw.copy()
    df_filtered["SWAT_CANCEL_SUMMARY"] = (
        df_filtered["SWAT_CANCEL_SUMMARY"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
    )
    df_filtered = df_filtered[df_filtered["SWAT_CANCEL_SUMMARY"].notna()].copy()

    if df_filtered.empty:
        raise RuntimeError("All rows are missing SWAT cancel summaries; nothing to cluster.")

    embeddings = compute_embeddings(client, df_filtered)
    _, labels = cluster_embeddings(embeddings, args.num_segments)

    df_labeled = df_filtered.assign(reason_segment=labels + 1)
    df_labeled.to_csv(args.output_path, index=False)

    df_augmented = df_raw.copy()
    df_augmented["reason_segment"] = pd.NA
    df_augmented.loc[df_labeled.index, "reason_segment"] = df_labeled["reason_segment"].to_numpy()
    augmented_output_path = args.data_path.with_name("0-6 month churn customer with segment.csv")
    df_augmented.to_csv(augmented_output_path, index=False)

    if args.plot_path:
        save_cluster_plot(embeddings, labels, args.plot_path)

    print(f"Wrote {len(df_labeled)} labeled rows to {args.output_path}")
    print(f"Saved augmented dataset with reason_segment to {augmented_output_path}")
    if args.plot_path:
        print(f"Saved PCA cluster visualisation to {args.plot_path}")

    if args.skip_gpt:
        return

    print("\nSegment summaries\n" + "-" * 80)
    segment_info_rows: list[dict[str, object]] = []
    for segment in sorted(df_labeled["reason_segment"].unique()):
        segment_rows = df_labeled[df_labeled["reason_segment"] == segment][SUMMARY_COLUMNS]
        summary = generate_segment_summary(client, segment_rows)
        segment_info_rows.append(
            {
                "segment": int(segment),
                "size": int(len(segment_rows)),
                "summary": summary,
            }
        )
        print(f"Segment {segment} | Size: {len(segment_rows)}")
        print(summary)
        print("-" * 80)

    if segment_info_rows:
        segment_info_path = args.output_path.with_name(f"{args.output_path.stem}_segment_info.csv")
        pd.DataFrame(segment_info_rows).to_csv(segment_info_path, index=False)
        print(f"Saved segment summaries to {segment_info_path}")


if __name__ == "__main__":
    main()
