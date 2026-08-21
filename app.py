import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

data = pd.read_csv("data/student_data.csv")


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

X = data[
    [
        "Hours_Studied",
        "Attendance",
        "Previous_Score",
        "Assignments"
    ]
]

y = data["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🎓 Student Performance Prediction System")

st.write(
    "An ML-based system that predicts a student's "
    "final score using academic performance factors."
)


# -------------------------------------------------
# MODEL METRICS
# -------------------------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )

with col3:
    st.metric(
        "Students in Dataset",
        len(data)
    )


# -------------------------------------------------
# STUDENT INPUT
# -------------------------------------------------

st.subheader("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:

    hours = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )


with col2:

    previous_score = st.number_input(
        "Previous Score",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    assignments = st.number_input(
        "Assignments Completed",
        min_value=0.0,
        max_value=20.0,
        value=8.0,
        step=1.0
    )


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button(
    "🔮 Predict Final Score",
    use_container_width=True
):

    student = pd.DataFrame(
        [
            [
                hours,
                attendance,
                previous_score,
                assignments
            ]
        ],
        columns=[
            "Hours_Studied",
            "Attendance",
            "Previous_Score",
            "Assignments"
        ]
    )

    prediction = model.predict(student)[0]

    prediction = max(
        0,
        min(100, prediction)
    )

    st.subheader("🎯 Prediction Result")

    st.success(
        f"Predicted Final Score: {prediction:.2f} / 100"
    )


    # Performance category

    if prediction >= 85:

        st.info(
            "🌟 Performance Level: Excellent"
        )

    elif prediction >= 70:

        st.info(
            "👍 Performance Level: Good"
        )

    elif prediction >= 50:

        st.warning(
            "📚 Performance Level: Average"
        )

    else:

        st.error(
            "⚠️ Performance Level: Needs Improvement"
        )


# -------------------------------------------------
# ACTUAL VS PREDICTED GRAPH
# -------------------------------------------------

st.subheader(
    "📈 Actual vs Predicted Scores"
)

fig, ax = plt.subplots()

ax.scatter(
    y_test,
    predictions
)

ax.set_xlabel(
    "Actual Final Score"
)

ax.set_ylabel(
    "Predicted Final Score"
)

ax.set_title(
    "Actual vs Predicted"
)

ax.grid(True)

st.pyplot(fig)


# -------------------------------------------------
# DATASET
# -------------------------------------------------

st.subheader(
    "📋 Dataset Preview"
)

st.dataframe(
    data,
    use_container_width=True
)