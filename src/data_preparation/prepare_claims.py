import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

TRAIN_FILE = os.path.join(RAW_DIR, "train.tsv")
DEV_FILE = os.path.join(RAW_DIR, "dev.tsv")
TEST_FILE = os.path.join(RAW_DIR, "test.tsv")

CLEANED_CLAIMS_FILE = os.path.join(PROCESSED_DIR, "pubhealth_cleaned.csv")
SAMPLE_CLAIMS_FILE = os.path.join(PROCESSED_DIR, "claims_sample_100.csv")


def prepare_claims():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_FILE, sep="\t")
    dev = pd.read_csv(DEV_FILE, sep="\t")
    test = pd.read_csv(TEST_FILE, sep="\t")

    df = pd.concat([train, dev, test], ignore_index=True)

    required_columns = [
        "claim_id",
        "claim",
        "label",
        "explanation",
        "main_text",
        "sources",
        "date_published",
        "subjects"
    ]

    df = df[required_columns]

    df = df.dropna(subset=["claim_id", "claim", "label", "main_text"])
    df = df.drop_duplicates(subset=["claim_id"])
    df = df.drop_duplicates(subset=["claim"])

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["claim"] = df["claim"].astype(str).str.strip()
    df["main_text"] = df["main_text"].astype(str).str.strip()

    df = df[df["claim"].str.len() > 20]
    df = df[df["main_text"].str.len() > 100]

    df = df.reset_index(drop=True)

    df.to_csv(CLEANED_CLAIMS_FILE, index=False)

    sample_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(25, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    sample_df.to_csv(SAMPLE_CLAIMS_FILE, index=False)

    print("Cleaned claims:", df.shape) # Cleaned claims: (12251, 8)
    print("Sample claims:", sample_df.shape) # Sample claims: (100, 7)
    print("\nLabel distribution:")
    print(sample_df["label"].value_counts())


if __name__ == "__main__":
    prepare_claims()