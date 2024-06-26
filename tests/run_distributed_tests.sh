# Run the tests and return exit code 0 if all tests pass, 1 if any fail
for test in \
    "pytest --runmpi tests/test_distributed.py::test_squared_edges_are_same_dist_split_over_dataset" \
    "pytest --runmpi tests/test_distributed.py::test_squared_edges_are_same_dist_split_over_out_dim" \
    "pytest --runmpi tests/test_distributed.py::test_stochastic_edges_are_same_dist_split_over_out_dim" \
    "pytest --runmpi tests/test_distributed.py::test_distributed_basis"
do
    echo "Running $test"
    $test
    if [ $? -ne 0 ]; then
        exit 1
    fi
done
exit 0