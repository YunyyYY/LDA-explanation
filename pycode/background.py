import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

data = np.random.random((20,20))
img = plt.imshow(data, interpolation='nearest')

# viridisBig = cm.get_cmap('RdPu', 512)
# viridisBig = cm.get_cmap('Blues', 512)
viridisBig = cm.get_cmap('Oranges', 512)
newcmp = ListedColormap(viridisBig(np.linspace(0, 0.4, 256)))

# img.set_cmap('Blues')
img.set_cmap(newcmp)
plt.axis('off')
plt.savefig("background/3.png", bbox_inches='tight')
