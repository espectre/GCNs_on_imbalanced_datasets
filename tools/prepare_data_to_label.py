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
import json


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

    if args.unique_list:
        unique_list_arr = np.load(args.unique_list)
    else:
        # list_data = []
        unique_list = []
        for i in data:
            for j in i:
                if j not in unique_list:
                    unique_list.append(j)
                # list_data.append(j)
        unique_list_arr = np.array(unique_list)

    return data, unique_list_arr


def generate_jasons(data,output_path,oss_prefix,no_files_face,no_files_body):
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    data_index = []
    no_additional_list_query = []
    no_additional_list_result = []

    print("Generating jason files ...")
    for i in range(len(data)):

        query_id = data[i][0]
        result_ids = data[i][1:]

        if query_id not in no_files_face:
            url_id = os.path.join(oss_prefix,'face/'+str(query_id)+'.jpg')
            url_id_body = os.path.join(oss_prefix,'body/'+str(query_id)+'.jpg')
            if query_id in no_files_body:
                no_additional_list_query.append(query_id)

            # resultlist = [os.path.join(oss_prefix,'face/'+str(item)+'.jpg' for item in result_ids]
            # resultlist_body = [os.path.join(oss_prefix,'body/'+str(item)+'.jpg' for item in result_ids]
            
            resultlist = []
            resultlist_body = []
            for item in result_ids:
                resultlist.append(os.path.join(oss_prefix,'face/'+str(item)+'.jpg'))
                resultlist_body.append(os.path.join(oss_prefix,'body/'+str(item)+'.jpg'))
                if item in no_files_body:
                    no_additional_list_result.append(item)

            fname = f"task-{i:06d}.json"
            data_index.append('data/'+fname)
            with open(output_dir.joinpath(fname), 'w') as f:
                json.dump({
                    "additional_info": {
                        "anno_info":"2"
                    },
                    "default_data": {
                        "data": {
                            "url_id":url_id,
                            "url_additional":[{"url_id":url_id_body}],
                            # "result":[dict(url_id=r,additional=[dict(url_id=additional) for additional in resultlist_body],label="4") for r in resultlist]
                            "result":[dict(url_id=r,additional_info=[dict(url_id=resultlist_body[idx])],label="4") for idx,r in enumerate(resultlist)]
                        }
                    },
                    "anno_result": [

                    ],    
                    "review_result": [

                    ],
                    "gt_data": {

                    },
                    "anno_info": [

                    ]
                }, f)

    print("Generating the index file ...")
    with open(output_dir.joinpath("../index").resolve(), 'w') as f:
        f.write("\n".join(data_index))
    
    print("No. of queries without body images: ",len(no_additional_list_query))
    print("No. of candidates without body images: ",len(no_additional_list_result))
    np.save("no_addition_list_query",np.array(no_additional_list_query))
    np.save("no_addition_list_result",np.array(no_additional_list_result))


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
    parser.add_argument("--pkl_path",
                        type=str,
                        default='./out_hardpos.pkl',
                        help="pickle file storing the list of face_ids to label")
    parser.add_argument("--unique_list",
                        type=str,
                        default='unique_list.npy',
                        help="npy file storing the list of unique face_ids")
    parser.add_argument("--src_prefix",
                        type=str,
                        default='./person',
                        help="the folder storing original body imgs")
    parser.add_argument("--dst_prefix",
                        type=str,
                        default='./person_renamed',
                        help="the folder storing renamed body imgs")
    parser.add_argument("--src_face_dir",
                        type=str,
                        default='./face',
                        help="the folder storing source face imgs")
    parser.add_argument("--src_body_dir",
                        type=str,
                        default='./body',
                        help="the folder storing source body imgs")
    parser.add_argument("--dst_face_dir",
                        type=str,
                        default='./face',
                        help="the folder storing selected face imgs")
    parser.add_argument("--dst_body_dir",
                        type=str,
                        default='./body',
                        help="the folder storing selected body imgs")
    parser.add_argument("--oss_prefix",
                        type=str,
                        default='dabiao/zhiyuan/hgy/hardsamples',
                        help="the oss folder storing the images")
    parser.add_argument("--output_path",
                        type=str,
                        default='hardsamples',
                        help="the folder storing task files")                 
    parser.add_argument("--process_body_imgs",
                        action='store_true',
                        help="whether to rename body imgs with face_ids")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.process_body_imgs:
        print("Processing body images ... ")
        rename_body_imgs(args.body_csv,args.face_csv,args.src_prefix,args.dst_prefix)

    clusters_list, unique_face_ids = load_pickle(args)

    print("Copying selected files to target folders ...")
    no_files_face = copy_selected_imgs(args.src_face_dir, args.dst_face_dir, unique_face_ids)
    print("-------------------")
    print("No face imgs: ", len(no_files_face))
    print(no_files_face)
    no_files_body = copy_selected_imgs(args.src_body_dir, args.dst_body_dir, unique_face_ids)
    print("-------------------")
    print("No body imgs: ", len(no_files_body))
    print(no_files_body)

    np.save("no_files_face",no_files_face)
    np.save("no_files_body",no_files_body)

    # Generate jason and index files
    generate_jasons(clusters_list,args.output_path,args.oss_prefix,no_files_face,no_files_body)

    print("!!!Done!!!")
