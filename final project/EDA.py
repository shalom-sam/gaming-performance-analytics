import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Shalo\OneDrive\Desktop\final project\player_stats.csv")

print(df.head())
print(df.describe())

# KD Ratio 
sns.histplot(df["KD_Ratio"], kde=True)
plt.title("KD Ratio Distribution")
plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()





from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    encoder = LabelEncoder()
    df["Rank"] = encoder.fit_transform(df["Rank"])
    df = df.drop("Player_ID", axis=1)
    return df




import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\Users\Shalo\OneDrive\Desktop\final project\player_stats.csv")


if "Player_ID" in df.columns:
    df = df.drop("Player_ID", axis=1)


encoder = LabelEncoder()
df["Rank"] = encoder.fit_transform(df["Rank"])


X = df.drop("Rank", axis=1)
y = df["Rank"]


X = X.apply(pd.to_numeric)
X = X.astype("float32")
y = y.astype("float32")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.dtypes)  # must show float32 only








#graph
from model_training import model 
importance = model.feature_importances_
features = X.columns

import matplotlib.pyplot as plt

plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Shalo\OneDrive\Desktop\final project\player_stats.csv")

# Accuracy vs Damage
sns.scatterplot(x="Accuracy", y="Damage", hue="Rank", data=df)
plt.title("Accuracy vs Damage")
plt.show()

# Matches vs Kills
sns.scatterplot(x="Matches", y="Kills", hue="Rank", data=df)
plt.title("Matches vs Kills")
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Neural Network
model_dl = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(5, activation="softmax")   # 5 ranks/classes
])

# Compile
model_dl.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model_dl.fit(X_train, y_train, epochs=20)





import numpy as np

new_player = np.array([[150, 500, 250, 2.0, 28, 20, 60000]], dtype="float32")

prediction = model_dl.predict(new_player)

print(prediction)



predicted_class = prediction.argmax()

print("Predicted Rank:", predicted_class)
