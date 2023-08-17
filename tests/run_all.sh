echo "TEST_HPC = True" > test_constants.py
echo "TEST_NONHPC = True" >> test_constants.py
python3 -m unittest unit/*.py
