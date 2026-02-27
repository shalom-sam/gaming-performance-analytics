from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    encoder = LabelEncoder()
    df["Rank"] = encoder.fit_transform(df["Rank"])
    df = df.drop("Player_ID", axis=1)
    return df