import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Select input features
X = data[["Hours_Studied", "Attendance", "Previous_Score", "Assignments"]]

# Select target
y = data["Final_Score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the ML model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Calculate performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== STUDENT PERFORMANCE PREDICTION =====")
print("Model trained successfully!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Get student details
print("\nEnter student details:")

hours = float(input("Hours studied: "))
attendance = float(input("Attendance percentage: "))
previous_score = float(input("Previous score: "))
assignments = float(input("Number of assignments completed: "))

# Create input data
student = pd.DataFrame(
    [[hours, attendance, previous_score, assignments]],
    columns=[
        "Hours_Studied",
        "Attendance",
        "Previous_Score",
        "Assignments"
    ]
)

# Predict final score
predicted_score = model.predict(student)[0]

# Display result
print("\n===== PREDICTION RESULT =====")
print("Predicted Final Score:", round(predicted_score, 2))