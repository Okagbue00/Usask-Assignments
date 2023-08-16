import sys

# process the command line
if len(sys.argv) != 2:
    print('Usage: python', sys.argv[0], '<filename>')
    sys.exit()

# read the term file
term_text = open(sys.argv[1], 'r')

# this reads the number of days for the exam
num_days = int(term_text.readline())

# read the classes
classes = {}

for roster_line in term_text.readlines():
    names = roster_line.split()

    # Add the class details
    classes[names[0]] = set(names[1:])

# define the CSP

# X - set of variables - set of classes that will hold exam for each day
X = [set() for _ in range(num_days)]

# D (shared across all X variable)
D = list(classes.keys())

# C - set of constraints

C = [set() for _ in range(num_days)]

# Loop through each day of the exam
for day in range(len(X)):
    i = 0
    while i < len(D):
        # Get the instructor
        instructor = D[i]

        # Check if constraint is satisfied
        if len(C[day].intersection(classes[instructor])) == 0:
            X[day].add(D[i])
            C[day] = C[day].union(classes[instructor])

            D = D[:i] + D[i + 1:]

        # Move to next instructor
        else:
            i += 1

# Print the exam schedule
print('Exam schedule', X)




