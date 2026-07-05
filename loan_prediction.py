import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
print(df.columns)
# Fill missing values
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert categorical columns to numbers
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
print("Loan Status Values:",df["Loan_Status"].unique())

# Features and Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

print("Prediction:", prediction[0])
if prediction[0] == 0:
    print("Loan Approved")
else:
    print("Loan Not Approved")