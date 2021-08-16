import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_data_path', type=str, metavar='PATH',
                        default='/mnt/disk/gcn_cluster/20200811_clean/mark0824/first_dir')
    parser.add_argument('--oss_prefix', type=str, metavar='PATH',
                        default='spectre/hp/first_dir/')
    parser.add_argument('--out_data_path', type=str, metavar='PATH',
                        default='./out')
    args = parser.parse_args()
    return args


# 读取指定路径下所有文件夹，每个文件夹内有不定数量的图片，且有一张图片的名称与文件夹名称相同为主图，其他图片为副图及副图辅助图
# 根据上述结构生成打标系统所需json
def main():
    args = parse_args()
    i = 0
    for home, dirs, files in os.walk(args.local_data_path):
        # print(home,dirs,files)
        # print("----------------",i)
        if i == 0:
            i += 1
            continue
        pic2json(home, files, args.oss_prefix, args.out_data_path)
        i += 1
    #     if i>=1:
    #         break


def pic2json(home, files, oss_prefix, out_data_path):
    # print('home',home,'files',files)
    dir_name = home.rsplit("/", 1)[1]
    main_url = dir_name
    # if '0_'+main_url not in files:
    #     return
    per_dic = create_default_json(main_url, files, oss_prefix)
    output_path = out_data_path + 'recognize_1_n/data/'
    if not os.path.exists(out_data_path):
        os.makedirs(output_path)
    output_json = output_path + main_url + '.json'
    with open(output_json, 'a') as f:  # 追加模式按行写入文件
        f.write(per_dic)

    index_path = out_data_path + 'recognize_1_n/index'
    with open(index_path, 'a') as f:  # 追加模式按行写入文件
        f.write('data/'+main_url+'.json\n')


def create_default_json(main_url, url_list, oss_prefix):
    """
    返回一个满足json格式的字典
    :param main_url:
    :param url_list:
    :param oss_prefix:
    :return:
    """
    dic = {
            "additional_info":{
                "anno_info": "1"
            },
            "default_data": {
               "data": {
                   "url_id": 'zhutu',
                    "result": []
                }
            },
            "anno_result": [],
            "review_result": [],
            "gt_data": {},
            "anno_info": []
    }

    info_ = dic['additional_info']
    id_ = dic['default_data']['data']['url_id']
    for url in url_list:
        if main_url + '.jpg' == url:
            continue

        if 'big' in url:
            continue

        url_result = {}
        url_result["url_id"] = oss_prefix+main_url+'/'+url
#         print(url_result["url_id"])
        url_result["label"] = "1"
#         url_result["additional_info"] = [{'url_id':oss_prefix+url.replace('samll','big')}]
        dic["default_data"]["data"]["result"].append(url_result)
        '''
        try:
            if url.find("shga-ryzp") == -1:
                dic["default_data"]["data"]["url_id"]=url
        except:
            continue
        '''
    # 主图赋值
    # print(url)
    dic["default_data"]["data"]["url_id"] = oss_prefix+main_url+'/'+main_url+'.jpg'
    return json.dumps(dic)


if __name__ == '__main__':
    main()
