import matplotlib.pyplot as plt
import numpy as np
import astropy.wcs as wcs

fig, ax = plt.subplots(nrows=2, ncols=2, dpi=150)
data = np.array([x for x in range(0, 100)])



ax[0][0].set_xlabel("data")
ax[0][0].set_ylabel("data")
ax[0][0].grid()

ax[0][1].plot(data, data**2)
ax[0][1].set_xlabel("data")
ax[0][1].set_ylabel("data ** 2")
ax[0][1].grid()


ax[1][0].plot(data, data**3)
ax[1][0].set_xlabel("data")
ax[1][0].set_ylabel("data ** 3")
ax[1][0].grid()


ax[1][1].plot(data, data**4)
ax[1][1].set_xlabel("data")
ax[1][1].set_ylabel("data ** 4")
ax[1][1].grid()

left=0.09
bottom=0.086
right=0.988
top=0.967
wspace=0.32
hspace=0.24
plt.tight_layout()
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
plt.show()