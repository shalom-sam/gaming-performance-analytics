import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocessing import preprocess


df = pd.read_csv(r"C:\Users\Shalo\OneDrive\Desktop\final project\player_stats.csv")



X = df.drop(["Rank", "Player_ID"], axis=1)
y = df["Rank"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))