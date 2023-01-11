import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, path_fig):

    plt.gcf().set_size_inches(8.5, 2)
    plt.subplot(131)
    plt.imshow(cm / np.sum(cm, axis=0),)
    plt.colorbar()
    plt.ylabel('Annotations')
    plt.xlabel('Predictions')

    plt.subplot(132)
    plt.imshow(np.log2(cm / np.sum(cm, axis=0)))# cmap='gray_r'
    plt.colorbar()
    plt.title('Log scaled')
    plt.savefig(path_fig)
    plt.show()