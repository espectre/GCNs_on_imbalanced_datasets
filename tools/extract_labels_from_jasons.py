#coding:utf-8 
import sys
import pickle
import numpy as np
import lmdb
import csv
from shutil import copyfile
import argparse
import os
from pathlib import Path
import json as js


def read_csv(path, keys):
    data = {}
    for item in keys:
        data[item] = []
    with open(path, mode='r') as csv_file:
        raw_data = csv.DictReader(csv_file)
        for row in raw_data:
            for item in keys:
                data[item].append(row[item])
    
    return data


def rename_files_with_face_id(src_path_prefix, dst_path_prefix, body_data, face_data, keys):
    # keys = [key of body name, key of face name, linking key]
    body_names = np.array(body_data[keys[0]])
    body_links = np.array(body_data[keys[2]])
    face_names = np.array(face_data[keys[1]])
    face_links = np.array(face_data[keys[2]])

    for idx,item in enumerate(body_names):
        body_fn = os.path.join(src_path_prefix,item+'.jpg')
        if os.path.exists(body_fn):
            face_idx = np.where(face_links == body_links[idx])
            if len(face_idx) == 1:
                target_fn = os.path.join(dst_path_prefix,str(face_names[face_idx][0])+'.jpg')
                print(target_fn)
                copyfile(body_fn, target_fn)
            else:
                print("!! No corresponding face_id info for body: ", item)
        print("!!!! No body img: ", item)


def rename_body_imgs(body_csv,face_csv,src_path_prefix,dst_path_prefix):
    face_data = read_csv(face_csv, ['face_id','record_id'])
    body_data = read_csv(body_csv, ['person_id','record_id'])
    rename_files_with_face_id(src_path_prefix,dst_path_prefix,body_data,face_data,['person_id','face_id','record_id'])


def copy_selected_imgs(src, dst, unique_face_ids):
    no_file_ids = []
    for face_id in unique_face_ids:
        src_fn = os.path.join(src, str(face_id)+'.jpg')
        dst_fn = os.path.join(dst, str(face_id)+'.jpg')
        if os.path.exists(src_fn):
            copyfile(src_fn, dst_fn)
        else:
            no_file_ids.append(face_id)
    
    return np.array(no_file_ids)


def load_pickle(args):
    data = pickle.load(open(args.pkl_path,'rb'))
    return data

    # if args.unique_list:
    #     unique_list_arr = np.load(args.unique_list)
    # else:
    #     # list_data = []
    #     unique_list = []
    #     for i in data:
    #         for j in i:
    #             if j not in unique_list:
    #                 unique_list.append(j)
    #             # list_data.append(j)
    #     unique_list_arr = np.array(unique_list)

    # return data, unique_list_arr


def read_jason(path):
    with open(path, mode='r') as js_file:
        raw_data = js.load(js_file)
    
        return raw_data

def get_label_result_list(raw_data):
    data = raw_data['review_result']
    if len(data[0]) == 0:
        data = raw_data['anno_result']
    
    data = data[0]
    data = data['data']

    return data


def get_exception(data):
    exceptions = data['exception']
    if len(exceptions) == 0:
        return None
    else:
        return exceptions[0]

def get_labels(data):
    results = data['result']
    labels = []
    labels_status = []
    for i in range(len(results)):
        result = results[i]
        label = result['label']
        label_status = result['labelStatus']
        labels.append(label)
        labels_status.append(label_status)

    return labels, labels_status

def get_sort(data):
    sorts = data['sort']
    if len(sorts) == 0:
        return None
    else:
        return sorts[0]

def extract_info(path):
    raw_data = read_jason(path)
    data = get_label_result_list(raw_data)
    exception = get_exception(data)
    sort = get_sort(data)
    labels, labels_status = get_labels(data)

    return labels, labels_status, exception, sort


def extract_labels(id_data,label_path,output_path):
    label_dir = Path(label_path)

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    labeled_index = []
    labels = []
    labels_status = []
    exceptions = []
    sorts = []
    num_exceptions = 0
    num_sorts = 0

    print("Extracting label info ...")
    for i in range(len(id_data)):

        # query_id = id_data[i][0]
        # result_ids = id_data[i][1:]

        label_file = label_dir.joinpath(f"task-{i:06d}.json")

        if os.path.exists(label_file):
            label, label_status, exception, sort = extract_info(label_file)
            if exception is not None:
                num_exceptions += 1
            if sort is not None:
                num_sorts += 1
            labels.append(label)
            labels_status.append(label_status)
            exceptions.append(exception)
            sorts.append(sort)
            labeled_index.append(i)
        else:
            labels.append(None)
            labels_status.append(None)
            exceptions.append(None)
            sorts.append(None)

    print("Saving info files ...")
    labels_file = output_dir.joinpath('labels.pkl')
    with open(labels_file, 'wb') as f:
        pickle.dump(labels, f)
    
    labels_status_file = output_dir.joinpath('labels_status.pkl')
    with open(labels_status_file, 'wb') as f:
        pickle.dump(labels_status, f)

    exceptions_file = output_dir.joinpath('exceptions.pkl')
    with open(exceptions_file, 'wb') as f:
        pickle.dump(exceptions, f)

    sorts_file = output_dir.joinpath('sorts.pkl')
    with open(sorts_file, 'wb') as f:
        pickle.dump(sorts, f)

    labeled_index_file = output_dir.joinpath('labeled_index.pkl')
    with open(labeled_index_file, 'wb') as f:
        pickle.dump(labeled_index, f)
    #with open(output_dir.joinpath("index.txt"), 'wb') as f:
    #    f.write("\n".join(labeled_index))
    
    print("Total queries: ",len(id_data))
    print("No. of labeled queries: ",len(labeled_index))
    print("No. of queries with exceptions: ",num_exceptions)
    print("No. of queries with sorts: ",num_sorts)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract info from returned jason files with label info')
    # parser.add_argument("--body_csv",
    #                     type=str,
    #                     default='hgy_damo_body_20200627.csv',
    #                     help="csv file storing body structure info")
    # parser.add_argument("--face_csv",
    #                     type=str,
    #                     default='hgy_damo_face_20200627.csv',
    #                     help="csv file storing face structure info")
    parser.add_argument("--pkl_path",
                        type=str,
                        default='./out_hardpos.pkl',
                        help="pickle file storing the list of face_ids to label")
    # parser.add_argument("--unique_list",
    #                     type=str,
    #                     default='unique_list.npy',
    #                     help="npy file storing the list of unique face_ids")
    # parser.add_argument("--src_prefix",
    #                     type=str,
    #                     default='./person',
    #                     help="the folder storing original body imgs")
    # parser.add_argument("--dst_prefix",
    #                     type=str,
    #                     default='./person_renamed',
    #                     help="the folder storing renamed body imgs")
    # parser.add_argument("--src_face_dir",
    #                     type=str,
    #                     default='./face',
    #                     help="the folder storing source face imgs")
    # parser.add_argument("--src_body_dir",
    #                     type=str,
    #                     default='./body',
    #                     help="the folder storing source body imgs")
    # parser.add_argument("--dst_face_dir",
    #                     type=str,
    #                     default='./face',
    #                     help="the folder storing selected face imgs")
    # parser.add_argument("--dst_body_dir",
    #                     type=str,
    #                     default='./body',
    #                     help="the folder storing selected body imgs")
    parser.add_argument("--label_path",
                        type=str,
                        default='raw_label_data',
                        help="the folder storing returned jason files with label info")
    parser.add_argument("--output_path",
                        type=str,
                        default='labels',
                        help="the folder storing label info")                 
    # parser.add_argument("--process_body_imgs",
    #                     action='store_true',
    #                     help="whether to rename body imgs with face_ids")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # if args.process_body_imgs:
    #     print("Processing body images ... ")
    #     rename_body_imgs(args.body_csv,args.face_csv,args.src_prefix,args.dst_prefix)

    # clusters_list, unique_face_ids = load_pickle(args)
    clusters_list = load_pickle(args)

    # print("Copying selected files to target folders ...")
    # no_files_face = copy_selected_imgs(args.src_face_dir, args.dst_face_dir, unique_face_ids)
    # print("-------------------")
    # print("No face imgs: ", len(no_files_face))
    # print(no_files_face)
    # no_files_body = copy_selected_imgs(args.src_body_dir, args.dst_body_dir, unique_face_ids)
    # print("-------------------")
    # print("No body imgs: ", len(no_files_body))
    # print(no_files_body)

    # np.save("no_files_face",no_files_face)
    # np.save("no_files_body",no_files_body)

    # # Generate jason and index files
    # generate_jasons(clusters_list,args.output_path,args.oss_prefix,no_files_face,no_files_body)

    extract_labels(clusters_list,args.label_path,args.output_path)

    print("!!!Done!!!")
