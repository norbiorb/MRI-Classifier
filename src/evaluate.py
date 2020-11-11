import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import cv2

from matplotlib import pyplot as plt
import seaborn as sns

from cnn_helper import preprocess_images
from gradcam_keras import GradCAM as GradCAMKeras

img_width = 224
img_height = 224

p_labels = ['AX', 'COR', 'SAG']
s_labels = ['DWI', 'T1', 'T1KM', 'T2']
long_p_labels = ['axial', 'coronal', 'sagittal']
long_s_labels = ['diffusion-\nweighted', 'T1-\nweighted', 'T1 with\ncontrast', 'T2-\nweighted']


def transform_labels(y_true, labels):
    le = LabelEncoder()
    le.fit(labels)
    return le.transform(y_true)


def get_predictions(model, df, train_classes, images_dir, rescale=True):
    x_col = 'image-name'
    y_col = 'y_true'
    # create data generator with rescaling only
    if rescale:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )
    # create data generator with resnet preprocessing only
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        )

    predictions = predict(model, df, images_dir, x_col, y_col, train_classes, datagen)
    df['predictions'] = pd.Series(list(predictions), index=df.index)
    df['y_pred'] = [np.argmax(pred) for pred in predictions]

    return df


def predict(model, df, images_dir, x_col, y_col, train_classes, datagen):
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=images_dir,
        x_col=x_col,
        y_col=y_col,
        class_mode='sparse',
        classes=train_classes,
        shuffle=False,
        target_size=(224, 224)
    )

    predictions = model.predict(
        generator,
        callbacks=None,
        max_queue_size=10,
        workers=-1,
        use_multiprocessing=False,
        verbose=1)

    return predictions


def print_metrics(df, labels):
    # plot confustion matrix
    true_labels = transform_labels(df['y_true'], labels)
    # print('Confusion Matrix')
    # plot_confusion_matrix(true_labels, df['y_pred'], labels=labels)

    # plot metrics
    metrics_df = get_metrics(true_labels, df['y_pred'])
    print('\n')
    print('Metrics')
    metrics_df

    return metrics_df


def get_metrics(y_true, y_pred):
    # Macro / Micro Driven Metrics

    metrics_dict = {}

    for avg in ['macro', 'micro']:
        met_name = 'precision' + ('_' + avg)
        res = metrics.precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics_dict[met_name] = res

        met_name = 'f1' + ('_' + avg)
        res = metrics.f1_score(y_true, y_pred, average=avg, zero_division=0)
        metrics_dict[met_name] = res

        met_name = 'recall' + ('_' + avg)
        res = metrics.recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics_dict[met_name] = res

    metrics_dict['accuracy'] = metrics.accuracy_score(y_true, y_pred)

    metrics_df = pd.DataFrame.from_records(metrics_dict, index=[0])

    return metrics_df


def plot_cm(
        df,
        labels,
        long_labels,
        title,
        label_type,
        fmt='.0f'
):
    l1 = df.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    """
    fig.suptitle(
        f'A                                      {label_type} confusion matrices', 
        fontsize=72,
        x=-0.0,
        y=1.05,
        horizontalalignment='left'
    )
    """
    plt1 = plot_cm_heatmap(df, labels, long_labels, ax,
                           f'{label_type.capitalize()} Confusion Matrix\n{title} ({l1} images)', fmt)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{label_type}_confusionmatrix_{title}.png', dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight')
    plt.show()
    return


def plot_cm_heatmap(df, labels, long_labels, ax, title, fmt):
    fontsize = 48
    labels_cat = pd.Series(labels).astype('category')
    true_labels = transform_labels(df['y_true'], labels)

    conf_mat = metrics.confusion_matrix(true_labels, df['y_pred'], labels=labels_cat.index)
    df_conf = pd.DataFrame(conf_mat, columns=labels_cat, index=labels_cat)

    ax = sns.heatmap(df_conf, annot=True, annot_kws={"size": fontsize}, fmt=fmt, ax=ax, cmap='Blues', cbar=False,
                     linewidths=0.5, linecolor='black', square=True)
    sns.set_style("darkgrid")

    fontdict = {
        'fontsize': fontsize,
        'fontweight': 'regular',
        'verticalalignment': 'center',
        'horizontalalignment': 'center'
    }
    ax.set_yticklabels(long_labels, verticalalignment='center', fontdict=fontdict)
    ax.set_xticklabels(long_labels, verticalalignment='center', fontdict=fontdict)

    ax.tick_params(axis='y', which='major', pad=100)
    ax.tick_params(axis='x', which='major', pad=60)

    ax.set_title(title, fontsize=fontsize, pad=80)

    ax.set_ylabel('Actual', fontsize=fontsize, labelpad=40)
    ax.set_xlabel('Predicted', fontsize=fontsize, labelpad=40)

    return ax


def gradcam_keras_output(
        model,
        im_path,
        actual,
        predicted,
        classifier_layer_names,
        last_conv_layer_name='conv5_block3_out'
):
    cam = GradCAMKeras(model, last_conv_layer_name)
    resized, im, orig = preprocess_images(im_path)
    # rescale image
    # im /= 255
    heatmap = cam.compute_heatmap_by_array(im, classifier_layer_names)

    heatmap = cv2.resize(heatmap, (resized.shape[0], resized.shape[1]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, resized, alpha=0.5)

    return output


def get_heatmaps(
        img_list,
        model,
        y_true,
        y_pred,
        classifier_layer_names,
        last_conv_layer_name='conv5_block3_out'
):
    heatmaps = []
    for i, img_path in enumerate(img_list):
        actual = y_true[i]
        predicted = y_pred[i]
        heatmap = gradcam_keras_output(
            model,
            img_path,
            actual,
            predicted,
            classifier_layer_names,
            last_conv_layer_name=last_conv_layer_name
        )
        heatmaps.append(heatmap)

    return heatmaps


def plot_gradcam_image(heatmaps, labels=['axial', 'coronal', 'sagittal'], title='Perspective'):
    n = len(heatmaps)

    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    fig.suptitle(title, fontsize=24)

    for i, heatmap in enumerate(heatmaps):
        ax = axes[i]
        ax = create_ax(ax, labels[i])
        ax.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

    # ax.set_title()
    # plt.rcParams["axes.grid"] = False
    plt.tight_layout()
    plt.savefig(f'{title}_gradcam.png', dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.show()

    return


def create_ax(ax, label):
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.lines = []
    ax.set_xlabel(label)
    ax.set_xlabel(label, fontsize=24, labelpad=10)

    return ax