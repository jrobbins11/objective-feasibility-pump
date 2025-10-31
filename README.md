## Objective Feasibility Pump

Implementation of the objective feasibility pump using HiGHS for LP solutions.

### References:
Achterberg, Tobias, and Timo Berthold. "Improving the feasibility pump." Discrete Optimization 4.1 (2007): 77-86.
Fischetti, Matteo, Fred Glover, and Andrea Lodi. "The feasibility pump." Mathematical Programming 104.1 (2005): 91-104.

### Examples:
C++ example:
```
cmake -DOFP_MAKE_EXAMPLE=ON -S . -B build
cmake --build build
./build/examples/test_feas_pump
```

Python example:
```
pip install .
python3 examples/test_feas_pump.py
```