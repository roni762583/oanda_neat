import sys

if len(sys.argv) > 1:
    if sys.argv[1] == 'test':
        print("Running test_genome.py logic")
        # Execute code from test_genome.py
        with open('test_genome.py', 'r') as file:
            exec(file.read())
    else:
        print("Running multi-test.py logic")
        # Execute code from multi-test.py
        with open('multi-test.py', 'r') as file:
            exec(file.read())
else:
    print("Running multi-test.py logic")
    # Execute code from multi-test.py by default
    with open('multi-test.py', 'r') as file:
        exec(file.read())
