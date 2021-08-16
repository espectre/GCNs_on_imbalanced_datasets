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
import time

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

def parse_str_feature_to_arr(raw_line):
    raw_line = raw_line[1: -1]
    raw_line = raw_line.split(',')
    raw_line = [float(x) for x in raw_line]
    return np.array(raw_line, dtype=np.float32)

def str2time_seconds(time_str,format='%d/%m/%Y %H:%M:%S.%f'):
    try:
        time.strptime(time_str,format)
    except:
        format='%d/%m/%Y %H:%M:%S'
    return time.mktime(time.strptime(time_str,format))

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


# keys = ['feature_data','shot_time','device_id']
def extract_info_from_csv_by_face_id(face_csv,body_csv,keys,unique_face_ids,output_path):
    face_keys = ['face_id', 'record_id']
    body_keys = ['person_id', 'record_id', 'feature_data'] 
    face_keys.extend(keys)
    face_data = read_csv(face_csv, face_keys)
    body_data = read_csv(body_csv, body_keys)

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # format feature data and shot time
    for key,value in face_data.items():
        if key == 'feature_data':
            for i in range(len(value)):
                face_data[key][i] = parse_str_feature_to_arr(value[i])
        elif key == 'shot_time':
            for i in range(len(value)):
                face_data[key][i] = str2time_seconds(value[i])

    # initialize dicts for selected samples
    selected_data = {}
    for item in keys:
        selected_data[item] = []
    selected_body_feat = []
    nodeid2faceid = {}
    faceid2nodeid = {}
    face_data_face_id = np.array(face_data['face_id'])
    body_data_record_id = np.array(body_data['record_id'])
    # select the ones by unique_face_ids
    for i in range(len(unique_face_ids)):
        nodeid2faceid[i] = unique_face_ids[i]
        faceid2nodeid[unique_face_ids[i]] = i
        index_ = np.where(face_data_face_id == nodeid2faceid[i])
        index_ = index_[0]
        if len(index_) == 1:
            for item in keys:
                selected_data[item].append(face_data[item][index_[0]])
            # get corresponding body features
            record_id_ = face_data['record_id'][index_[0]]
            index_body = np.where(body_data_record_id == record_id_)
            index_body = index_body[0]

            body_feat = None
            if len(index_body) == 1:
                body_feat = parse_str_feature_to_arr(body_data['feature_data'][index_body[0]])
            selected_body_feat.append(body_feat)
        else:
            for item in keys:
                selected_data[item].append(None)
            selected_body_feat.append(None)
            print("Did not find the data for face_id: ", nodeid2faceid[i])

    # Save the above data to files
    for item in keys:
        print("Saving: ", item)
        fn = output_dir.joinpath(item+'.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(selected_data[item], f)

    print("Saving nodeid2faceid.dict")
    fn = output_dir.joinpath('nodeid2faceid.dict')
    with open(fn, 'wb') as f:
        pickle.dump(nodeid2faceid, f)

    print("Saving faceid2nodeid.dict")
    fn = output_dir.joinpath('faceid2nodeid.dict')
    with open(fn, 'wb') as f:
        pickle.dump(faceid2nodeid, f)

    print("Saving body_feature_data.pkl")
    fn = output_dir.joinpath('body_feature_data.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(selected_body_feat, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data to label')
    parser.add_argument("--body_csv",
                        type=str,
                        default='hgy_damo_body_20200627.csv',
                        help="csv file storing body structure info")
    parser.add_argument("--face_csv",
                        type=str,
                        default='hgy_damo_face_20200627.csv',
                        help="csv file storing face structure info")
    parser.add_argument("--unique_list",
                        type=str,
                        default='unique_list.npy',
                        help="npy file storing the list of unique face_ids")
    parser.add_argument("--output_path",
                        type=str,
                        default='selected_samples',
                        help="the folder storing selected samples")                 
    # parser.add_argument("--process_body_imgs",
    #                     action='store_true',
    #                     help="whether to rename body imgs with face_ids")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # clusters_list = load_pickle(args)

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

    unique_face_ids = np.load(args.unique_list)
    keys = ['feature_data','shot_time','device_id']
    extract_info_from_csv_by_face_id(args.face_csv,args.body_csv,keys,unique_face_ids,args.output_path)

    print("!!!Done!!!")
