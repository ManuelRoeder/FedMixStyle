import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def min_max_norm_tsne(x):
    tsne = TSNE(n_components=2, random_state=42)

    data = tsne.fit_transform(x)
    data_max, data_min = np.max(data, 0), np.min(data, 0)
    d = (data - data_min) / (data_max - data_min)
    return d


def plot_centroids(centroids, num_classes, include_mean = False):
    num_domains = int(len(centroids) / num_classes)
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")
    label_list = list()
    for _ in range(num_domains):
        label_list.extend(list(range(1, num_classes + 1)))
    np_centroids = np.asarray(centroids)
    transformed_data = min_max_norm_tsne(np_centroids)
    df = pd.DataFrame({'tsne_1': transformed_data[:, 0], 'tsne_2': transformed_data[:, 1], 'label': label_list})
    _, ax = plt.subplots(1)
    plt.figure(figsize=(8, 8), dpi=300)

    # split array
    df_split = np.array_split(df, num_domains)
    # cycle through domains and create scatterplots
    marker_list = ["o", "v", "x"]
    for idx, split in enumerate(df_split):
        if include_mean and idx == 0:
            # mean centroids
            sns.scatterplot(
                x="tsne_1", y="tsne_2",
                # hue="label",
                # palette=sns.color_palette("hls", nr_classes),
                c="red",
                data=split,
                legend=False,
                alpha=1.0,
                # from http://mirrors.ibiblio.org/CTAN/fonts/stix/doc/stix.pdf
                marker=marker_list[idx],
                s=100)
        else:
            sns.scatterplot(
                x="tsne_1", y="tsne_2",
                hue="label",
                palette=sns.color_palette("hls", num_classes),
                data=split,
                legend=False,
                alpha=0.5,
                # from http://mirrors.ibiblio.org/CTAN/fonts/stix/doc/stix.pdf
                marker=marker_list[idx],
                s=350)

    lim = (transformed_data.min() - 0.1, transformed_data.max() + 0.1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')

    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
