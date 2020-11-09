import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import cv2

from matplotlib import pyplot as plt
import seaborn as sns

import os
from pathlib import Path

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from cnn_helper import preprocess_images
from gradcam_keras import GradCAM as GradCAMKeras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

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


def transform_perspective(y_true):
    return transform_labels(y_true, p_labels)


def transform_sequence(y_true):
    return transform_labels(y_true, s_labels)


def map_sequence_predictions(preds):
    pred_map = {0: 'DWI', 1: 'T1', 2: 'T1KM', 3: 'T2'}
    return pred_map.get(np.argmax(preds))


def map_perspective_predictions(preds):
    pred_map = {0: 'AX', 1: 'COR', 2: 'SAG'}
    return pred_map.get(np.argmax(preds))


def create_df(selected_dir, label_type, get_perspective_label, get_sequence_label):
    image_list = list(Path(selected_dir).glob('*.jpg'))

    df = pd.DataFrame([str(x) for x in image_list], columns=['image-path'])
    df['image-name'] = df['image-path'].apply(lambda x: x.split('/')[-1])

    if label_type == 'sequence':
        df['y_true'] = df['image-name'].apply(get_sequence_label)
    elif label_type == 'perspective':
        df['y_true'] = df['image-name'].apply(get_perspective_label)
    else:
        df['y_true_p'] = df['image-name'].apply(get_perspective_label)
        df['y_true_s'] = df['image-name'].apply(get_sequence_label)

    return df


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

    return


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


def plot_confusion_matrix(
        y_true,
        y_pred,
        labels,
        title='Confusion Matrix',
        save_path='Confusion_Matrix.png',
        fmt='.0f'
):
    """function to plot confusion matrix heatmap"""

    labels_cat = pd.Series(labels).astype('category')

    conf_mat = metrics.confusion_matrix(y_true, y_pred, labels=labels_cat.index)
    df_conf = pd.DataFrame(conf_mat, columns=labels_cat, index=labels_cat)

    fig, ax = plt.subplots(figsize=(10, 9))
    ano = True if df_conf.shape[0] < 100 else False
    plt.suptitle(title, fontsize=16)
    sns.heatmap(df_conf, annot=ano, fmt=fmt, ax=ax, cmap='Blues', cbar=False,
                linewidths=.1, linecolor='black', square=True)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(save_path, facecolor='white', edgecolor='none')
    plt.show()

    return


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


def evaluate_df(df, predictions, get_label, map_predictions):
    df['predictions'] = pd.Series(list(predictions), index=df.index)
    df['y_true'] = df['image_name'].apply(get_label)
    df['y_pred'] = df['predictions'].apply(map_predictions)
    # df['correct'] = df.apply(lambda r: 1 if r.y_true == r.pred else 0, axis=1)
    return df


def predict_and_save(model, images_dir, get_label, save_path):
    predictions_df = model.predict(images_dir, get_label)
    predictions_drop_df = predictions_df.drop('image_path', axis=1)
    predictions_drop_df.to_csv(save_path)
    return predictions_df


def plot_gradcam_img(original, output, heatmap, title='GradCam'):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    # images are in cv2 format 'BGR' but pyplot uses 'RGB'
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax3.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

    # add titles
    fig.suptitle(title)
    ax1.set_title('Original')
    ax2.set_title('GradCAM Output')
    ax3.set_title('Heatmap')
    plt.tight_layout()
    plt.show()

    return


def plot_gradcam_keras_output(output, title='GradCam'):
    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
    # images are in cv2 format 'BGR' but pyplot uses 'RGB'
    ax1.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    # add titles
    fig.suptitle(title)
    ax1.set_title('GradCAM Output')
    plt.tight_layout()
    plt.show()

    return


def plot_gradcam_keras(
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
    image_name = os.path.split(im_path)[-1]
    title = f'GradCAM \n{image_name} \nactual: {actual} predicted: {predicted}'

    plot_gradcam_img(orig, output, heatmap, title)
    return


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

    #plot_gradcam_img_list(heatmaps)
    return heatmaps


def plot_heatmaps(heatmaps, names, y_true, y_pred):
    N = len(heatmaps)
    columns = 5
    rows = N // columns + 1
    #plt.gcf().set_size_inches(5*rows, 5*columns)
    for i, heatmap in enumerate(heatmaps):
        heatmap = cv2.resize(heatmap, (224, 224))
        #plt.subplot(rows, columns, i + 1)

        #plt.imshow(heatmap)

        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
        # images are in cv2 format 'BGR' but pyplot uses 'RGB'
        ax1.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        # add titles
        ax1.set_title('GradCAM Output')
        fig.suptitle(f'Image {names[i]} class {y_true[i]} , predicted {y_pred[i]}')

        plt.tight_layout()
    plt.show()


def plot_gradcam_img_list(model, images_dir, image_names, y_true, y_pred, classifier_layer_names):
    columns = 1
    rows = len(image_names)

    cam = GradCAM(model)

    for i, image_name in enumerate(image_names):
        im_path = os.path.join(images_dir, image_name)
        actual = y_true[i]
        predicted = y_pred[i]

        resized, im, orig = preprocess_images(im_path)
        heatmap = cam.compute_heatmap_by_array(im, classifier_layer_names, eps=1e-100)
        heatmap = cv2.resize(heatmap, (resized.shape[0], resized.shape[1]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, resized, alpha=0.5)

        image_name = os.path.split(im_path)[-1]
        title = f'GradCAM \n{image_name} \nactual: {actual} precicted: {predicted}'

        plt.subplot(rows, columns, i + 1)
        plt.title(title)
        plt.axis('off')


def plot_gradcam_image(heatmaps, labels=['axial', 'coronal', 'sagittal'], title='Perspective'):
    n = len(heatmaps)

    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
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