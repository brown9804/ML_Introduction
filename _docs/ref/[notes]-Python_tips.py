import pandas as pd

getting_sample = pd.read_csv("./file_.csv")
df = pd.DataFrame(getting_sample)
print(df.head(5).to_csv("./getting_sample.csv"))
