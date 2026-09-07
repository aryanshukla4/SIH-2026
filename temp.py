import time, highspy
h = highspy.Highs()
h.readModel('tests/data/netlib/80bau3b.mps')
t0 = time.time()
h.run()
print('status:', h.getModelStatus())
print('objective:', h.getInfo().objective_function_value)
print('solve_time_seconds:', time.time() - t0)
