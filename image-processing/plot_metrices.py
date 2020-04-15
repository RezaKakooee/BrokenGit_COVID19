"""
==============================================================================
============ A binary classifeir for Two-Class imageset with DeepNN ========== 
=========================== by ABIZ: MP, TK, RK ==============================
================================ December 2019 ===============================
==============================================================================
"""

### ======================================================================= ###
###### plot_metrics.py
### ======================================================================= ###

#%% ======================================================================= ###
###### Import packages
### ======================================================================= ###
import os
import numpy as np
import matplotlib.pyplot as plt
from metadata import Classes_Metadata

#%% ======================================================================= ###
###### Define plot metrics class
class plot_histories(object):
    def __init__(self, 
                 histories_dir = Classes_Metadata().histories_dir,
                 pretrained_model_name = Classes_Metadata().histories_dir):
        
        self.pretrained_model_name = pretrained_model_name
        self.histories_dir = histories_dir
        
        self.metadata = Classes_Metadata()
        self.initial_epochs = self.metadata.INITIAL_EPOCHS
        self.NUM_TOTAL_EPOCHS = self.metadata.TOTAL_EPOCHS
        self.NUM_SPLITS = self.metadata.n_split
        
    def plotting(self):
        histories_dir = self.histories_dir
        initial_epochs = self.initial_epochs
        NUM_TOTAL_EPOCHS = self.NUM_TOTAL_EPOCHS
        NUM_SPLITS = self.NUM_SPLITS
        
        histories = tuple(os.listdir(histories_dir))
        
        recall_list = []
        precision_list = []
        val_recall_list = []
        val_precision_list = []
        plt.figure(figsize=(10, 10))
        for i in range(NUM_SPLITS):
            historyb_name = histories[2*i]
            historyt_name = histories[2*i+1]
            
            historyb_path = os.path.join(histories_dir, historyb_name)
            historyt_path = os.path.join(histories_dir, historyt_name)
            
            history_base = np.load(historyb_path, allow_pickle=True)
            history_tune = np.load(historyt_path, allow_pickle=True)
            
            history_base = history_base.item()
            history_tune = history_tune.item()
            
            historyb = dict()
            historyt = dict()
            
            keysb = history_base.keys()
            keyst = history_tune.keys()
            
            for keyb in keysb:
                if "val_precision" in keyb:
                    historyb['val_precision'] = history_base[keyb]
                    val_precision = history_base[keyb]
                elif "precision" in keyb:
                    historyb['precision'] = history_base[keyb]
                    precision = history_base[keyb]
                    
                if "val_recall" in keyb:
                    historyb['val_recall'] = history_base[keyb]
                    val_recall = history_base[keyb]
                elif "recall" in keyb:
                    historyb['recall'] = history_base[keyb]
                    recall = history_base[keyb]
                    
                    
            for keyt in keyst:
                if "val_precision" in keyt:
                    historyt['val_precision'] = history_tune[keyt]
                    val_precision += history_tune[keyt]
                elif "precision" in keyt:
                    historyt['precision'] = history_tune[keyt]
                    precision += history_tune[keyt]
                    
                if "val_recall" in keyt:
                    historyt['val_recall'] = history_tune[keyt]
                    val_recall += history_tune[keyt]
                elif "recall" in keyt:
                    historyt['recall'] = history_tune[keyt]
                    recall += history_tune[keyt]
        
            f1_score = 2 * np.array(precision) * np.array(recall) / (np.array(precision) + np.array(recall))
            val_f1_score = 2 * np.array(val_precision) * np.array(val_recall) / (np.array(val_precision) + np.array(val_recall))
            
            f1_score = np.nan_to_num(f1_score)
            val_f1_score = np.nan_to_num(val_f1_score)
            
            # plot recall and percision for train and validation
            plt.subplot(4, 1, i+1)
            epochs = np.arange(1, NUM_TOTAL_EPOCHS+1)
            plt.plot(epochs, precision, label='Training Precision', marker='d')
            plt.plot(epochs, val_precision, label='Validation Precision', marker='x')
            plt.plot(epochs, recall, label='Training Recall', marker='o')
            plt.plot(epochs, val_recall, label='Validation Recall', marker='P')
            plt.plot(epochs, f1_score, label='Training F1 Score', marker='+', linestyle='dashed')
            plt.plot(epochs, val_f1_score, label='Validation F1 Score', marker='*', linestyle='dashed')
            
            plt.ylim([0.0, 1])
            plt.plot([initial_epochs,initial_epochs],
                      plt.ylim(), label='Start Fine Tuning')
            plt.legend(loc='upper left')
            plt.xticks(np.arange(1, NUM_TOTAL_EPOCHS+1))
            
            plt.title("Split {}".format(i+1))
            
            if i == 0:
                plt.arrow(1, 0.2, 4, 0, length_includes_head=True,
                    head_width=0.1, head_length=0.3)
                plt.arrow(4.7, 0.2, -3.7, 0, length_includes_head=True,
                    head_width=0.1, head_length=0.3)
                
                plt.arrow(6, 0.2, 4, 0, length_includes_head=True,
                    head_width=0.1, head_length=0.3)
                plt.arrow(9.7, 0.2, -3.7, 0, length_includes_head=True,
                    head_width=0.1, head_length=0.3)
            
                plt.text(2.2, 0.25, 'Base Model')
                plt.text(7.2, 0.25, 'Fine Tune Model')
        
            precision_list.append(precision)
            recall_list.append(recall)
            val_precision_list.append(val_precision)
            val_recall_list.append(val_recall)

    
        # average
        precision_avg = np.mean(precision_list, axis=0)
        recall_avg = np.mean(recall_list, axis=0)
        
        val_precision_avg = np.mean(val_precision_list, axis=0)
        val_recall_avg = np.mean(val_recall_list, axis=0)
        
        
        f1_score_avg = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)
        val_f1_score_avg = 2 * val_precision_avg * val_recall_avg / (val_precision_avg + val_recall_avg)
        
        f1_score_avg = np.nan_to_num(f1_score_avg)
        val_f1_score_avg = np.nan_to_num(val_f1_score_avg)
        
        plt.subplot(4, 1, 4)
        epochs = np.arange(1, NUM_TOTAL_EPOCHS+1)
        plt.plot(epochs, precision_avg, label='Training Average Precision', marker='d')
        plt.plot(epochs, val_precision_avg, label='Validation Average Precision', marker='x')
        plt.plot(epochs, recall_avg, label='Training Average Recall', marker='o')
        plt.plot(epochs, val_recall_avg, label='Validation Average Recall', marker='P')
        plt.plot(epochs, f1_score_avg, label='Training Average F1 Score', marker='+', linestyle='dashed')
        plt.plot(epochs, val_f1_score_avg, label='Validation Average F1 Score', marker='*', linestyle='dashed')
        
        plt.ylim([0.0, 1])
        plt.plot([initial_epochs,initial_epochs],
                  plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper left')
        plt.xticks(np.arange(1, NUM_TOTAL_EPOCHS+1))
        plt.title("Average Metrices Over {} Splits".format(i+1))
        if i == 2: plt.xlabel('Epoch')
    
#       plt.subplots_adjust(top=1.2)
        plt.show()

# histories_dir = Classes_Metadata().histories_dir #os.path.join('D:\HSLU\HSLU_Selfie\exported-files\InceptionV3\histories-nparray\models-cv')
# plot_histories(histories_dir=histories_dir).plotting()