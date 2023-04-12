echo "TEST_HPC = False" > test_constants.py
echo "TEST_NONHPC = True" >> test_constants.py
python3 -m unittest unitary/*.py
