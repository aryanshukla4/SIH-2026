import time, highspy
h = highspy.Highs()
h.readModel(r'D:\git hub projects\SIH 2026\tests\data\netlib\shell.mps')
t0 = time.time()
h.run()
print('status:', h.getModelStatus())
print('objective:', h.getInfo().objective_function_value)
print('solve_time_seconds:', time.time() - t0)
