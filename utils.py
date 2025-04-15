import networkx as nx
import re
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold


import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split as split



def split_dataset_category(data_path, ratio, seed):
    from sklearn.model_selection import train_test_split
    df_view_angles = pd.read_csv(data_path)
    train, test = train_test_split(df_view_angles, test_size=ratio, stratify=df_view_angles['category'], random_state=seed)
    training_samples = train['id'].values
    test_samples = test['id'].values
    return training_samples, test_samples


def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys


def post_processing_voting(df, dataset):
    results_df = pd.DataFrame(columns=df.columns)
    for test_sample in np.unique(df['test_sample']):
        g = dataset[test_sample]['g']
        artery_branches = [g.nodes()[k]['data'].vessel_class for k in g.nodes()]
        sub_df = df[df['test_sample']==test_sample]
        data_row = {"test_sample": test_sample, "template_sample": "",
                    "category": np.unique(sub_df['category'])[0],
                    "n": len(artery_branches)}
        matched = 0
        for artery_branch in artery_branches:
            # print(f"[x] sample {test_sample}, branch {artery_branch}")
            sub_series = sub_df[artery_branch].dropna()
            if sub_series.shape[0] > 0:
                unique, counts = np.unique(sub_series, return_counts=True)
                label = unique[np.argmax(counts)]
                data_row[artery_branch] = label
                if artery_branch == label:
                    matched += 1

        data_row['matched'] = matched
        results_df = results_df.append(data_row, ignore_index=True)

    return results_df


def evaluate_main_branches(df, dataset, print_result=True):
    columns = []
    columns.extend(["test_sample", "category"])

    branch_mapping = {}
    for sub_branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
        for main_branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
            if sub_branch_name.startswith(main_branch_name):
                branch_mapping[sub_branch_name] = main_branch_name
    # print(branch_mapping)

    for main_branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
        columns.append(f"{main_branch_name}_matched")
        columns.append(f"{main_branch_name}_unmatched")

    result_df = pd.DataFrame(columns=columns)

    for test_sample in np.unique(df['test_sample']):
        g = dataset[test_sample]['g']
        sub_artery_branches = [g.nodes()[k]['data'].vessel_class for k in g.nodes()]
        sub_df = df[df['test_sample'] == test_sample]

        # init data row
        data_row = {"test_sample": test_sample, "category": np.unique(sub_df['category'])[0]}

        for branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
            data_row[f'{branch_name}_matched'] = 0
            data_row[f'{branch_name}_unmatched'] = 0

        # assign value
        for sub_branch_name in sub_artery_branches:
            # print(f"[x] sample {test_sample}, branch {sub_branch_name}")
            sub_series = sub_df[sub_branch_name].dropna()
            if sub_series.shape[0] > 0:
                unique, counts = np.unique(sub_series, return_counts=True)
                label = unique[np.argmax(counts)]
                # data_row[artery_branch] = label
                mapped_main_branch = branch_mapping[sub_branch_name] # map sub label to main branch label defined in Artery.MAIN_BRANCH_CATEGORY
                if sub_branch_name == label:
                    data_row[f'{mapped_main_branch}_matched'] = data_row[f'{mapped_main_branch}_matched']+1
                else:
                    data_row[f'{mapped_main_branch}_unmatched'] = data_row[f'{mapped_main_branch}_unmatched'] + 1

        result_df = result_df.append(data_row, ignore_index=True)

    for main_branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
        result_df[f"{main_branch_name}_total"] = result_df[f"{main_branch_name}_matched"]+result_df[f"{main_branch_name}_unmatched"]

    if print_result:
        for main_branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
            if result_df[f'{main_branch_name}_total'].sum() == 0:
                print("{}, total = {}, matched = {}, acc = {}".format(main_branch_name,
                                                                      result_df[f'{main_branch_name}_total'].sum(),
                                                                      result_df[f'{main_branch_name}_matched'].sum(), 0.))
            else:
                print("{}, total = {}, matched = {}, acc = {}".format(main_branch_name,
                                        result_df[f'{main_branch_name}_total'].sum(),
                                        result_df[f'{main_branch_name}_matched'].sum(),
                                        result_df[f'{main_branch_name}_matched'].sum()/result_df[f'{main_branch_name}_total'].sum()))
    return result_df


def evaluate_main_branches_sklearn(df, dataset, method="macro", return_results=False):

    def convert_name_to_label(artery_names):
        lbls = []
        for artery_name in artery_names:
            for i in range(len(["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"])):
                if artery_name == ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"][i]:
                    lbls.append(i)
        return lbls

    branch_mapping = {}
    for sub_branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
        for main_branch_name in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
            if sub_branch_name.startswith(main_branch_name):
                branch_mapping[sub_branch_name] = main_branch_name

    gts = []
    preds = []

    for test_sample in np.unique(df['test_sample']):
        g = dataset[test_sample]['g']
        sub_artery_branches = [g.nodes()[k]['data'].vessel_class for k in g.nodes()]
        # all_arteries_gt.extend(sub_artery_branches)

        sub_df = df[df['test_sample'] == test_sample]

        for sub_branch_name in sub_artery_branches:
            # print(f"[x] sample {test_sample}, branch {sub_branch_name}")
            sub_series = sub_df[sub_branch_name].dropna()
            if sub_series.shape[0] > 0:
                unique, counts = np.unique(sub_series, return_counts=True)
                label = unique[np.argmax(counts)]
                # data_row[artery_branch] = label
                mapped_main_branch_gt = branch_mapping[sub_branch_name] # map sub label to main branch label defined in Artery.MAIN_BRANCH_CATEGORY
                mapped_main_branch_pred = branch_mapping[label]
                gts.append(mapped_main_branch_gt)
                preds.append(mapped_main_branch_pred)
    

    gts = convert_name_to_label(gts)
    preds = convert_name_to_label(preds)

    cm = metrics.confusion_matrix(gts, preds)
    acc = metrics.accuracy_score(gts, preds)
    precision = metrics.precision_score(gts, preds, average=method)
    recall = metrics.recall_score(gts, preds, average=method)
    f1_score = metrics.f1_score(gts, preds, average=method)

    try:
        clf_report = metrics.classification_report(gts, preds, target_names=["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"], output_dict=True)
    except:
        clf_report = metrics.classification_report([0,1,2,3,4], [0,0,0,0,0], target_names=["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"], output_dict=True)
    
    if return_results:
        return cm, clf_report, acc, precision, recall, f1_score, gts, preds
    else:
        return cm, clf_report, acc, precision, recall, f1_score
    


def get_category(sample_name):
    for category in ["RAO_CAU", "LAO_CAU", "AP_CAU", "RAO_CRA", "LAO_CRA", "AP_CRA"]:
        if sample_name.rfind(category) != -1:
            return category
    return ""
