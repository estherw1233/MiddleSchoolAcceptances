# MiddleSchoolAcceptances

In this final project, we will explore a dataset provided by the New York City Department of Education. Of particular interest is whether characteristics of NYC middle schools predict admission to one of 8 highly selective public high schools (Stuyvesant, Bronx High School of Science, etc.) in New York (from now on called HSPHS). Admission to these schools is contingent on applying AND scoring sufficiently highly on the Specialized High Schools Admissions Test (SHSAT), an independently produced and anonymously graded standardized test.

The Dataset: The dataset (‘middleSchoolData.csv’) contains data from all 594 NYC middle schools, including 485 public schools and 109 charter schools (in the last 109 rows) from a randomly picked year in the past 5 years. Each row of the dataset represents a particular school, so the unit of analysis is “school”.

Here is what the columns represent:
A - B: NYC DOE school code and name, respectively
C: Number of applications to HSPHS originating from this school
D: Number of applicants to HSPHS accepted from this school
E: Per student spending, in $
F: Average class size
G-K: Self-described ethnic identity of the student body
L-Q: Average rating of “school climate” factors as perceived by the students, e.g. trust
R: Percentage of students who have been evaluated as disabled
S: Percentage of students living in households below the poverty line
T: Percentage of ESL students
U: School size (Number of students in the entire school)
V: Average student achievement on a state-wide standardized test
W-X: Proportion of students exceeding state-wide expectations in reading and math

This dataset is comprehensive, but some data is missing. If data in a cell is missing, you have to handle (clean or impute) it, in order to do the analyses. Sometimes, data is missing systematically. For instance, data for columns E and F is missing for all charter schools.
