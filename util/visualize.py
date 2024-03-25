import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Subset
from util.util import get_output
import copy
import itertools
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)


def my_confusion_matrix(y_true, y_pred,args):
    N = args.num_classes
    cm = np.zeros((N,N))
    for n in range(y_true.shape[0]):
        cm[int(y_true[n]), int(y_pred[n])] += 1
    return cm 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          mode=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)
        cm[np.where(np.isnan(cm))[0]] = np.NaN

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    if  mode == 'predict_vc_noise':
       plt.ylabel('Noise label')
       plt.xlabel('Predicted label')
    if mode == 'predict_vs_true':
       plt.ylabel('True label')
       plt.xlabel('Predicted label')
    if mode == 'noise_vs_true':
       plt.ylabel('True label')
       plt.xlabel('Noise label')
 

def visual_non_iid(args, class_mat, category_names, save_path):
    """
    Parameters
    ----------
    class_mat : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    df = pd.DataFrame(data = class_mat, columns = category_names)
    df.to_csv(save_path+'non_iid_stat.csv')
    
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    title = r'p=%.2f, $\alpha_{Dir}=%.2f$'%(args.non_iid_prob_class,args.alpha_dirichlet)
    labels = [i for i in range(args.num_users)]
    data_cum = class_mat.cumsum(axis=1)
    category_colors = plt.get_cmap('tab20c')(  #tab20c
        np.linspace(0.15, 0.85, class_mat.shape[1]))   # RdYlGn 0.15, 0.85
    plt.axes().get_xaxis().set_visible(True)
    plt.title(title, fontsize=15)
    plt.ylabel("Client", fontsize=15)
    plt.xlabel("Class distribution", fontsize=15)
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = class_mat[:, i]
        starts = data_cum[:, i] - widths
        rects = plt.barh(labels, widths, left=starts, height=0.8,
                        label=colname, color=color)
        
    plt.legend(ncol=1,loc=(1.01,0.659), fontsize=10)
    plt.savefig(save_path+'non_iid_fig',dpi=300, bbox_inches = "tight")
    #plt.show()

def visual_non_iid_dots(args, class_mat, save_path):
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 22))
    # Set the axes limits
    # ax.set_xlim(0, 4)
    # ax.set_ylim(0, 50)
    classes = np.arange(0,args.num_classes)
    for idx in range(class_mat.shape[0]):
        
        # Create a scatter plot of the red dots
        y = np.ones(len(classes))*idx
        sc = ax.scatter(x=classes, y=y, c='red', s=(class_mat[idx]).astype(int))
        sc = ax.scatter(x=classes, y=y, c='red', s=(class_mat[idx]).astype(int))
        ax.hlines(y=idx,xmin=-1,xmax=class_mat.shape[1],color="blue",linewidth=0.5,zorder=0,linestyle='--')
    for c in classes:
        ax.vlines(x=c,ymin=-1,ymax=class_mat.shape[0],color="blue",linewidth=0.5,zorder=0,linestyle='--')

    # Add axis labels
    ax.set_xlabel('class id')
    ax.set_ylabel('client id')

    # Add tick labels
    ax.set_xticks(np.arange(class_mat.shape[1]))
    ax.set_yticks(np.arange(class_mat.shape[0]))
    ax.margins(x=0,y=0)
    # Set the title of the figure
    title = r'p=%.2f, $\alpha_{Dir}=%.2f$'%(args.non_iid_prob_class,args.alpha_dirichlet)
    ax.set_title(title)

    # fig.tight_layout()
    # Show the figure
    handles, labels = sc.legend_elements(prop="sizes", alpha=0.5, color = 'red')
    legend2 = ax.legend(handles, labels, loc="center left", title="# of samples", bbox_to_anchor=(1, 0.5),labelspacing = 1.0)
    plt.savefig(save_path+'non_iid_dots_fig',dpi=300, bbox_inches = "tight")

def visual_cnf_mat(args, idx=None, noise_level = None, true_labels=None, noise_labels=None, save_path=None, sub_path=None):
        class_names = np.arange(0,args.num_classes)
        # if not test:
        # plt.figure(figsize=(10,8))
        # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False)
        # noise_labels = np.array([])
        # for _,label in loader:
        #     noise_labels = np.append(noise_labels, label)
            
        # output, _ = get_output(loader, net.to(args.device), args, False, None)
        # label_output = np.argmax(output, axis=1)
        # conf_matrix = my_confusion_matrix(noise_labels, label_output,args)
        # plot_confusion_matrix(conf_matrix, classes=class_names, normalize=True, title='Client %d - %.4f'%(idx, noise_level),mode='predict_vc_noise')
        # savepath = save_path+'conf_matrix/'+'train/noise_label/client_'+ str(idx) +'/'
        # if not os.path.exists(savepath):
        #         os.makedirs(savepath)
        # savepath = savepath + sub_path + '_conf_matrix_plot'
        # plt.savefig(savepath,dpi=300, bbox_inches = "tight")
        
        # plt.figure(figsize=(10,8))
        # conf_matrix = my_confusion_matrix(true_labels, label_output,args)
        # plot_confusion_matrix(conf_matrix, classes=class_names, normalize=True, title='Client %d - %.4f'%(idx, noise_level),mode='predict_vs_true')
        # savepath = save_path+'conf_matrix/'+'train/true_label/client_'+ str(idx) +'/'
        # if not os.path.exists(savepath):
        #         os.makedirs(savepath)
        # savepath = savepath + sub_path + '_conf_matrix_plot'
        # plt.savefig(savepath,dpi=300, bbox_inches = "tight")

        plt.figure(figsize=(10,8))
        conf_matrix = my_confusion_matrix(true_labels, noise_labels,args)
        plot_confusion_matrix(conf_matrix, classes=class_names, normalize=True, title='Client %d - %.4f'%(idx, noise_level),mode='noise_vs_true')
        savepath = save_path+'conf_matrix/'+'true_vs_noise/client_'+ str(idx) +'/'
        if not os.path.exists(savepath):
                os.makedirs(savepath)
        savepath = savepath + sub_path + '_conf_matrix_plot'
        plt.savefig(savepath,dpi=300, bbox_inches = "tight")
       
        # else:
        #     plt.figure(figsize=(10,8))
        #     y_test_true = np.array(dataset.targets)
        #     conf_matrix = my_confusion_matrix(y_test_true, test_output,args)
        #     plot_confusion_matrix(conf_matrix, classes=class_names, normalize=True, title='Test Data', mode = 'predict_vs_true')
        #     savepath = save_path+'conf_matrix/' + 'test/'
        #     if not os.path.exists(savepath):
        #             os.makedirs(savepath)
        #     savepath = savepath + sub_path + '_test_conf_matrix_plot'
        #     plt.savefig(savepath,dpi=300, bbox_inches = "tight")
