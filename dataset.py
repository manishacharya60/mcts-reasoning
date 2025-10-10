from datasets import load_dataset
import pandas as pd

# Method 1: Load and convert to pandas DataFrame
print("=== Method 1: Using pandas DataFrame ===")
ds = load_dataset("HuggingFaceH4/MATH-500")
df = ds['test'].to_pandas()

# Alternative: View specific columns in full
# print("\n=== Viewing specific columns completely ===")
# for idx, row in df.head(3).iterrows():
#     print(f"\n--- Row {idx} ---")
#     print(f"Problem: {row['problem']}")
#     print(f"Solution: {row['solution']}")
#     print(f"Answer: {row['answer']}")
#     print(f"Subject: {row['subject']}")
#     print(f"Level: {row['level']}")
#     print(f"ID: {row['unique_id']}")

# Alternative: Examine one row completely
print("\n=== Single Row Complete View ===")
row = df.iloc[485]
print("Problem:", row['problem'])
print("Solution:", row['solution'])
print("Answer:", row['answer'])
print("Subject:", row['subject'])
print("Level:", row['level'])
print("Unique ID:", row['unique_id'])

# print(df['level'].unique())
# print(df['subject'].unique())