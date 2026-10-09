# ML-Student-Grade-Predictor

# Student Performance Analysis Using K-Means

A simple machine learning project that uses K-Means clustering to group students based on their study habits and academic performance. It estimates a student's marks and pass/fail result using the average marks of their assigned group.

## Features

- Sample dataset of 25 students
- Data understanding and cleaning
- Data handling using Pandas and NumPy
- Feature scaling using StandardScaler
- Student grouping using K-Means clustering
- Data visualization using Matplotlib
- User input for student details
- Estimated marks and pass/fail result

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## How to Run

**1. Install the required libraries**

```bash
pip install pandas numpy matplotlib scikit-learn
```

**2. Run the Python file**

```bash
python student_prediction.py
```

**3. Enter student details**

Provide study hours, attendance percentage, previous marks, assignment score, and daily sleep hours.

The program will assign the student to a group and display estimated marks and a pass/fail result.

## How It Works

1. Creates a sample dataset of students.
2. Cleans and scales the data.
3. Uses K-Means to identify three student groups.
4. Calculates the average marks for each group.
5. Assigns a new student to a group.
6. Estimates marks using that group's average and displays the result.

## Important Note

This project demonstrates unsupervised learning for educational purposes. K-Means identifies groups rather than directly predicting marks. The marks and pass/fail result are estimates based on a small dummy dataset and should not be treated as reliable academic predictions.

## Author

Nayan Verma
