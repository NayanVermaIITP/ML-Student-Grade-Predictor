
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# student data
data = {
    "study_hours": [1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,1,3,5,6,7,2,4,6,8,9],
    "attendance": [45,50,55,60,65,70,75,90,48,58,62,68,72,80,95,40,56,70,78,85,52,64,74,92,96],
    "previous_marks": [30,35,40,45,50,55,60,75,32,42,46,52,58,65,80,25,38,55,62,70,34,48,60,78,85],
    "assignment_score": [20,25,30,40,45,50,60,80,22,32,38,48,55,65,85,15,28,52,60,72,25,42,58,82,90],
    "sleep_hours": [5,6,6,7,7,8,7,8,5,6,7,7,8,8,9,5,6,7,8,7,6,7,8,9,8],
    "final_marks": [25,30,38,45,50,60,68,85,28,36,43,52,62,72,90,20,35,55,65,78,32,47,64,88,95]
}

df = pd.DataFrame(data)

# check data
print(df.head())
print(df.shape)
print(df.isnull().sum())

# clean data
df = df.drop_duplicates().copy()
df = df.fillna(df.median(numeric_only=True))

# select columns for clustering
x = df[[
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "sleep_hours"
]]

# scale the values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# make 3 groups of students
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["group"] = model.fit_predict(x_scaled)

# check average marks in each group
group_marks = df.groupby("group")["final_marks"].mean()
print(group_marks)

# show the groups
for group in sorted(df["group"].unique()):
    students = df[df["group"] == group]
    plt.scatter(
        students["study_hours"],
        students["final_marks"],
        label=f"Group {group}"
    )

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Student Performance")
plt.legend()
plt.show()

# enter a new student's details
study = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
previous = float(input("Previous marks: "))
assignment = float(input("Assignment score: "))
sleep = float(input("Sleep hours per day: "))

student = pd.DataFrame([[
    study, attendance, previous, assignment, sleep
]], columns=x.columns)

# find the group and estimate the result
student_scaled = scaler.transform(student)
group = model.predict(student_scaled)[0]
marks = group_marks.loc[group]

print("\nEstimated marks:", round(marks, 2))

if marks >= 40:
    print("Result: Pass")
else:
    print("Result: Fail")
