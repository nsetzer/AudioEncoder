
import os
import sys
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve

def generate_roc(cfg, model, x, y, pngFile):
    yp = model.predict(x)

    genres = cfg.getGenres()
    n_classes = len(cfg.getGenres())

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], yp[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        np.asarray(y).ravel(),
        np.asarray(yp).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'seagreen', 'darkorchid',
                    'firebrick', 'orangered', 'sienna',
                    'goldenrod', 'steelblue', 'hotpink',
                    'forestgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC %s (area = %.2f)' % (genres[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s' % cfg.featureRecipe)
    plt.legend(loc="lower right")
    plt.savefig(pngFile)

def plot_confusion(cfg, cm_frame, cm_file):
    dir = cfg.plotDirectory()
    os.makedirs(dir, exist_ok=True)
    run_id = cfg.uid()
    ctime = cfg.ctime_s()
    filename_fmt = "%s-%s-%%s_cm.png" % (ctime, run_id)
    cm_frame.plot('Frame Confusion Matrix',
                  os.path.join(dir, filename_fmt % "frame"))
    cm_file.plot('File Confusion Matrix',
                 os.path.join(dir, filename_fmt % "file"))