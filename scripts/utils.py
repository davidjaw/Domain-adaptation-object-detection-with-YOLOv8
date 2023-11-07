import numpy as np
import os
import cv2


def load_setup(yaml_path='config.yaml'):
    import yaml
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # determine whether os is windows or linux
    if os.name == 'nt':
        config = config['windows']
    else:
        config = config['linux']
    return config


def parse_setup(config: dict, add_worker=False):
    s = ''
    for key in config:
        s += f'--{key} {config[key]} '
    if add_worker:
        s += f'--workers {min(os.cpu_count(), 8) if os.name != "nt" else 3} '
    return s


def visualize_img(img: np.ndarray, label: [[int | float]]):
    # the label is in COCO format
    if len(label) > 0:
        for label in label:
            c, x, y, w, h = label
            x = int(x * img.shape[1])
            y = int(y * img.shape[0])
            w = int(w * img.shape[1])
            h = int(h * img.shape[0])
            # visualize the label with class num
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
            cv2.putText(img, str(int(c)), (int(x - w / 2), int(y - h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def visualize(img: np.ndarray, labels: [[int | float]], ratio=1):
    # Create a color palette for different classes
    color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Set thickness and font size relative to image dimensions
    thickness = int(max(img.shape) * 0.003) + 1
    font_scale = img.shape[1] / 1000.0

    if len(labels) > 0:
        for label in labels:
            c, x, y, w, h = label
            x = int(x * img.shape[1])
            y = int(y * img.shape[0])
            w = int(w * img.shape[1])
            h = int(h * img.shape[0])

            # visualize the label with class num
            color = color_palette[int(c) % len(color_palette)]
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, thickness)

            # Calculate size of text and create a semi-transparent rectangle behind it
            text = str(int(c))
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            overlay = img.copy()
            cv2.rectangle(overlay, (int(x - w / 2), int(y - h / 2) - text_height - 10),
                          (int(x - w / 2) + text_width, int(y - h / 2)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

            # Adjust text position slightly above the box
            text_pos = (int(x - w / 2), int(y - h / 2) - 10)
            cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    if ratio != 1:
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def restore_coord(img_shape: [int], x: float, y: float, w: float, h: float) -> (int, int, int, int):
    """
    Restore the coordinate of the object to the original image
    Args:
        img_shape: (h, w) or (h, w, c)
        x: float in [0, 1] corresponding to the x-center of the object
        y: float in [0, 1] corresponding to the y-center of the object
        w: float in [0, 1] corresponding to the width of the object
        h: float in [0, 1] corresponding to the height of the object
    Returns:
        (x, y, w, h) in respect to the original image coordinate/size
    """
    x = int(x * img_shape[1])
    y = int(y * img_shape[0])
    w = int(w * img_shape[1])
    h = int(h * img_shape[0])
    return x, y, w, h


def get_img_label_path(base_path) -> ([str], [str]):
    image_dir = os.path.join(base_path, "images")
    label_dir = os.path.join(base_path, "labels")
    img_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    img_paths = [os.path.join(image_dir, img_file) for img_file in img_files]
    label_paths = [os.path.join(label_dir, label_file) for label_file in label_files]
    return img_paths, label_paths


def crop_object(img: np.ndarray, x, y, w, h, target_size=None) -> np.ndarray:
    x, y, w, h = restore_coord(img.shape, x, y, w, h)
    img = img[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img


def get_object_mask(img_shape, x, y, w, h) -> np.ndarray:
    """
    Get the mask of the object (white) and the background (black)
    Returns:
        mask: np.ndarray of shape same as the input image with dtype np.uint8
    """
    x, y, w, h = restore_coord(img_shape, x, y, w, h)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask[y - h // 2:y + h // 2, x - w // 2:x + w // 2] = 255
    return mask


def tsne_2d(features, labels, save_path, pca_n=100, max_num=2500, name_list=None, show_legend=True, fn=None, title=None):
    """
    Perform PCA followed by t-SNE clustering on features and plot the results in 2D.
    Note that max_num limits the number of features plotted, or else it'll pop error.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    print(f'Performing PCA + t-SNE on {features.shape[0]} features...')
    print(f'- Processing PCA with {pca_n} components...', end='')
    # Reduce dimensionality using PCA
    if features.shape[0] < pca_n:
        print(f'Warning: features.shape[0] ({features.shape[0]}) < pca_n ({pca_n}),'
              f'Setting pca_n to features.shape[0] // 2 ({features.shape[0] // 2})')
        pca_n = features.shape[0] // 2
    pca = PCA(n_components=pca_n)
    reduced_features = pca.fit_transform(features)
    print('Done.')

    # pop warning if exceeding max_num
    if reduced_features.shape[0] > max_num:
        print(f'Warning: features.shape[0] ({reduced_features.shape[0]}) > max_num ({max_num}),'
              f'only plotting the first {max_num} features to avoid memory error.')
        indices = np.random.choice(reduced_features.shape[0], max_num, replace=False)
        reduced_features = reduced_features[indices]
        labels = labels[indices]

    # Determine unique number of classes and create color palette
    num_classes = len(np.unique(labels))
    palette = sns.color_palette("husl", num_classes)

    if name_list:
        if len(name_list) != num_classes:
            raise ValueError(f'len(name_list) ({len(name_list)}) != num_classes ({num_classes})')
    else:
        name_list = [f'Class {i}' for i in range(num_classes)]

    print(f'- Processing t-SNE with {num_classes} classes...', end='')
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(reduced_features[:max_num, :])
    labels_array = labels[:max_num]
    print('Done.')

    fig, ax = plt.subplots(figsize=(11, 10), dpi=100)
    # Plot t-SNE with the Set2 color palette
    for i in range(num_classes):
        plt.scatter(transformed_data[labels_array == i, 0], transformed_data[labels_array == i, 1], label=name_list[i],
                    color=palette[i], alpha=0.75)

    if show_legend:
        plt.legend()
    if title is None:
        title = f't-SNE Clustering ({os.path.basename(save_path)})'
    plt.title(title)
    if fn is None:
        fn = f'{os.path.basename(save_path)}-PCA-tSNE'
    plt.savefig(f'{save_path}/{fn}.png')
    plt.clf()
    plt.close()


def tsne_2d_on_dir(dir_name, pca_dim=100, name_list=None):
    """
    Perform t-SNE clustering on features saved in the provided directory.
    Args:
    - dir_name (str): Directory from which features should be loaded.
    Returns:
    - None. Displays the t-SNE plot.
    """
    # Load features and labels
    features_list = []
    labels = []

    for file_name in os.listdir(dir_name):
        if file_name.endswith('.npy'):
            file_path = os.path.join(dir_name, file_name)
            features = np.load(file_path)
            batch_size = features.shape[0]
            class_idx, _ = file_name.split('_')
            features_list.append(features)
            labels += [int(class_idx)] * batch_size

    # Convert to numpy arrays
    features_array = np.vstack(features_list)
    labels_array = np.array(labels)
    if len(features_array.shape) > 2:
        features_array = features_array.reshape(features_array.shape[0], -1)
    tsne_2d(features_array, labels_array, dir_name, pca_n=pca_dim, name_list=name_list)


def tsne_clustering_3d(dir_name):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D
    """
    Perform t-SNE clustering on features saved in the provided directory.

    Args:
    - dir_name (str): Directory from which features should be loaded.

    Returns:
    - None. Displays the t-SNE plot in 3D.
    """
    # Load features and labels
    features_list = []
    labels = []

    for file_name in os.listdir(dir_name):
        if file_name.endswith('.npy'):
            file_path = os.path.join(dir_name, file_name)
            features = np.load(file_path)
            batch_size = features.shape[0]
            class_idx, _ = file_name.split('_')
            features_list.append(features)
            labels += [int(class_idx)] * batch_size

    # Convert to numpy arrays
    features_array = np.vstack(features_list)
    labels_array = np.array(labels)

    # Determine unique number of classes and create color palette
    num_classes = len(np.unique(labels_array))
    palette = sns.color_palette("Set2", num_classes)

    # Perform t-SNE with 3 components
    tsne = TSNE(n_components=3, random_state=42)
    transformed_data = tsne.fit_transform(features_array[:2800, :])
    labels_array = labels_array[:2800]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot t-SNE with the Set2 color palette
    for i in range(num_classes):
        ax.scatter(transformed_data[labels_array == i, 0],
                   transformed_data[labels_array == i, 1],
                   transformed_data[labels_array == i, 2],
                   label=f"Class {i}",
                   color=palette[i])

    ax.legend()
    ax.set_title(f't-SNE Clustering in 3D ({os.path.basename(dir_name)})')
    plt.savefig(f'{os.path.dirname(dir_name)}/{os.path.basename(dir_name)}-tSNE-3D.png')
    plt.clf()
    plt.close()


