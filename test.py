import numpy as np
import matplotlib.pyplot as plt

f = plt.figure(0)
g = plt.figure(1)
ax = f.add_subplot(1,1,1, label=0)
xs = np.linspace(0, 2 * np.pi, 100)
image, = ax.plot(xs, np.sin(xs), 'r-')
bx = g.add_subplot(1,1,1)
bx.plot(xs, np.cos(xs), 'g-')
image.set_data(xs, np.tan(xs))
plt.draw()
plt.show()