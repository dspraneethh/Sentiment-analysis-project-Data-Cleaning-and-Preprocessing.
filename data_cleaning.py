import pandas as pd

# --------------------------------------------------
# Step 1: Load the dataset
# --------------------------------------------------
df = pd.read_csv("Sentiment_dataset.csv")

print("\nDataset loaded successfully!")
print("Columns found:", list(df.columns))

# --------------------------------------------------
# Step 2: Auto-detect text and sentiment columns
# --------------------------------------------------
# Common possible names
text_candidates = ["text", "sentence", "review", "comment", "content"]
sentiment_candidates = ["sentiment", "label", "polarity", "class"]

text_col = None
sentiment_col = None

for col in df.columns:
    if col.lower() in text_candidates:
        text_col = col
    if col.lower() in sentiment_candidates:
        sentiment_col = col

if text_col is None or sentiment_col is None:
    raise ValueError(
        "Could not automatically detect text or sentiment columns.\n"
        f"Available columns: {list(df.columns)}"
    )

print(f"Using TEXT column: '{text_col}'")
print(f"Using SENTIMENT column: '{sentiment_col}'")

# --------------------------------------------------
# Step 3: Initial inspection
# --------------------------------------------------
print("\nInitial dataset info:")
print(df.info())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# Step 4: Remove rows with missing text or sentiment
# --------------------------------------------------
df = df.dropna(subset=[text_col, sentiment_col])

# --------------------------------------------------
# Step 5: Remove duplicate rows
# --------------------------------------------------
df = df.drop_duplicates()

# --------------------------------------------------
# Step 6: Standardize sentiment values
# --------------------------------------------------
df[sentiment_col] = df[sentiment_col].astype(str).str.lower().str.strip()

# --------------------------------------------------
# Step 7: Create text length feature
# --------------------------------------------------
df["text_length"] = df[text_col].astype(str).apply(len)

# --------------------------------------------------
# Step 8: Final validation
# --------------------------------------------------
print("\nAfter cleaning:")
print(df.info())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# Step 9: Save cleaned dataset
# --------------------------------------------------
df.to_csv("cleaned_sentiment_dataset.csv", index=False)

print("\n✅ Data cleaning completed successfully!")
print("Cleaned file saved as: cleaned_sentiment_dataset.csv")
