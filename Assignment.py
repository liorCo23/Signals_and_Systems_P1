import matplotlib.pyplot as plt
import numpy as np

# creating the time vector n=-1000:1:1000.
# start value is -1000 and the stop value is 1001 (1001 not included).
n = np.arange(-1000, 1001)

# creating the Window signal as required.
# np.abs(n) < 100 make sure that the condition is only for what between -99 and 99 (99 and -99 included)
a = np.where(np.abs(n) < 100, 1, 0)

# ----Plotting the signal and the axis----

# plotting the signal, the higher zorder is means that the line is above the lower zorder lines
plt.plot(n, a, zorder=2)

# plotting x-axis and y-axis with lower zorder than the signal
plt.axhline(y=0, color='black', linewidth=1.15, zorder=0)
plt.axvline(x=0, color='black', linewidth=1.15, zorder=0)

# labels and title
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('Window Signal')
plt.grid(True)
plt.show()
