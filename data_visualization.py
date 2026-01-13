import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Step 1: Load cleaned dataset
# --------------------------------------------------
df = pd.read_csv("cleaned_sentiment_dataset.csv")

print("Dataset loaded for visualization")
print(df.head())

# --------------------------------------------------
# Step 2: Bar Chart - Sentiment Distribution
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# --------------------------------------------------
# Step 3: Line Chart - Text Length Trend
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(df["text_length"])
plt.title("Text Length Trend")
plt.xlabel("Index")
plt.ylabel("Text Length")
plt.tight_layout()
plt.savefig("text_length_trend.png")
plt.show()

# --------------------------------------------------
# Step 4: Scatter Plot - Text Length vs Sentiment
# --------------------------------------------------
sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

df["sentiment_encoded"] = df["Sentiment"].map(sentiment_map)

plt.figure(figsize=(6, 4))
plt.scatter(df["text_length"], df["sentiment_encoded"], alpha=0.6)
plt.title("Text Length vs Sentiment")
plt.xlabel("Text Length")
plt.ylabel("Sentiment (Encoded)")
plt.yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
plt.tight_layout()
plt.savefig("text_length_vs_sentiment.png")
plt.show()

print("✅ Data visualization completed successfully!")
