import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans

def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """
    closest = np.empty([image.shape[0], image.shape[1]])
    closest[:] = 9e+10
    closest_centroid = image.copy()
    for centroid in centroids:
        dist = np.sqrt(np.sum((image - centroid)**2, axis=2))
        condition = np.where(closest > dist) 
        closest[condition] = dist[condition]
        closest_centroid[condition] = centroid
    return closest_centroid

def main(args):

    # Setup
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'vscikit-learn_orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'vscikit-learn_orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image2D = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(image2D)
    image_clustered = update_image(image, kmeans.cluster_centers_)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'vscikit-learn_updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    args = parser.parse_args()
    main(args)
