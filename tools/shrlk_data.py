import os
import numpy as np
from utils.misc import TextColors, l2norm, dump2npy
# from utils import (load_data, dump_data, mkdir_if_no_exists, Timer)

class SHRLK_Dataset():

    def __init__(self,
                 name,
                 start_index=0,
                 doc_amount=3,
                 prefix='data',
                 dim=512,
                 normalize=False,
                 sort=False,
                 sort_type='id_wise',
                 clean=True,
                 verbose=True):
        self.name = name
        self.dtype = np.float32
        self.dim = dim
        self.normalize = normalize
        self.sort = sort
        self.sort_type = sort_type
        self.clean_repeating_samples = clean

        if not os.path.exists(prefix):
            raise FileNotFoundError(
                'folder({}) does not exist.'.format(prefix))

        self.prefix = prefix
        self.doc_prefix = os.path.join(prefix, name)

        if doc_amount > 1:
            pic_ids = []
            features = []
            labels = []

            for i in range(start_index,doc_amount):
                curr_doc_path = self.doc_prefix + str(i) + '.npz'
                #curr_doc_path = '/home/zhiyuan.zfy/data/renlianku_sh/valset_gn_ft_0' + str(i) + '.npz'
                print("Loading: ", curr_doc_path)
                curr_pic_ids, curr_features, curr_labels = self.read_npz_files(curr_doc_path)
                pic_ids.append(curr_pic_ids)
                features.append(curr_features)
                labels.append(curr_labels)
        
            self.pic_ids, self.features, self.labels = self.merge_dataset(pic_ids,features,labels)
        else:
            curr_doc_path = self.doc_prefix+'.npz'
            self.pic_ids, self.features, self.labels = self.read_npz_files(curr_doc_path)

        if self.clean_repeating_samples:
            self.remove_duplicated_samples()
        
        self.lb2idxs, self.idx2lb = self.count_label(self.labels,
                                                  verbose=verbose)
        self.inst_num = len(self.idx2lb)
        self.cls_num = len(self.lb2idxs)
    
        if self.normalize:
            self.features = l2norm(self.features)

        if self.sort:
            print("Sort Enabled! Type: "+self.sort_type)
            self.sort_dataset()
        else:
            print("Shuffle Enabled!")
            self.shuffle_dataset()

    def info(self):
        print('name:{}{}{}\ninst_num:{}\ncls_num:{}\ndim:{}\n'
              'feat_path:{}\nnormalization:{}{}{}\ndtype:{}'.format(
                  TextColors.OKGREEN, self.name, TextColors.ENDC,
                  self.inst_num, self.cls_num, self.dim, self.doc_prefix,
                  TextColors.FATAL, self.normalize, TextColors.ENDC,
                  self.dtype))

    def count_label(self, label, verbose=True):
        lb2idxs = {}
        idx2lb = {}
        for idx, x in enumerate(label):
            lb = x
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

        inst_num = len(idx2lb)
        cls_num = len(lb2idxs)
        if verbose:
            print('#cls: {}, #inst: {}'.format(cls_num, inst_num))
        return lb2idxs, idx2lb

    def read_npz_files(self, file_path, verbose=True):
        if os.path.isfile(file_path):
            data = np.load(file_path,allow_pickle=True)
            pic_ids = data['pic_id']
            features = data['feature']
            labels = data['label']
            if verbose:
                print('[{}] features shape: {}'.format(file_path, features.shape))
        else:
            print(
                'meta file not found: {}.\n'
                'init `lb2idxs` and `idx2lb` as None.'
                .format(file_path))
            pic_ids = None
            features = None
            labels = None

        return pic_ids, features, labels 
    
    def sort_dataset(self):
        c = np.c_[self.pic_ids.reshape(len(self.pic_ids),-1),self.features.reshape(len(self.features), -1), self.labels.reshape(len(self.labels), -1)]
        if self.sort_type == 'pic_wise':
            c = c[c[:,0].argsort()]
        else:
            c = c[c[:,-1].argsort()]
        self.pic_ids = c[:,:self.pic_ids.size//len(self.pic_ids)].reshape(self.pic_ids.shape)
        self.features = c[:,self.pic_ids.size//len(self.pic_ids):-self.labels.size//len(self.labels)].reshape(self.features.shape).astype('float32')
        self.labels = c[:,-self.labels.size//len(self.labels):].reshape(self.labels.shape).astype('int32')

    def shuffle_dataset(self):
        # c = np.c_[self.features.reshape(len(self.features), -1), self.labels.reshape(len(self.labels), -1)]
        c = np.c_[self.pic_ids.reshape(len(self.pic_ids),-1),self.features.reshape(len(self.features), -1), self.labels.reshape(len(self.labels), -1)]
        np.random.shuffle(c)
        # self.features = c[:,:self.features.size//len(self.features)].reshape(self.features.shape).astype('float32')
        # self.labels = c[:,self.features.size//len(self.features):].reshape(self.labels.shape).astype('int32')
        self.pic_ids = c[:,:self.pic_ids.size//len(self.pic_ids)].reshape(self.pic_ids.shape)
        self.features = c[:,self.pic_ids.size//len(self.pic_ids):-self.labels.size//len(self.labels)].reshape(self.features.shape).astype('float32')
        self.labels = c[:,-self.labels.size//len(self.labels):].reshape(self.labels.shape).astype('int32')


    def merge_dataset(self,pic_ids,features,labels):
        len_pic_ids = len(labels)

        merged_pic_ids = pic_ids[0]
        merged_features = features[0]
        merged_labels = labels[0]

        for i in range(1,len_pic_ids):
            print("merging documents: ", i)
            merged_pic_ids = np.concatenate((merged_pic_ids, pic_ids[i]), axis=0)
            merged_features = np.concatenate((merged_features, features[i]), axis=0)
            merged_labels = np.concatenate((merged_labels, labels[i]), axis=0)

        return merged_pic_ids, merged_features, merged_labels

    def remove_duplicated_samples(self):
        feat_uniq, unique_index = np.unique(self.features,return_index=True,axis=0)
        self.features = self.features[unique_index]
        self.labels = self.labels[unique_index]
        self.pic_ids = self.pic_ids[unique_index]

    def split_dataset(self):
        split_index = len(self.features)//2
        print("Spliting index: ",split_index)
        train_feat = self.features[:split_index,:]
        print("training feature matrix shape: ", train_feat.shape)
        test_feat = self.features[split_index:,:]
        print("testing feature matrix shape: ", test_feat.shape)
        train_label = self.labels[:split_index,:]
        print("training label matrix shape: ", train_label.shape)
        test_label = self.labels[split_index:,:]
        print("testing label matrix shape: ", test_label.shape)
        label_train_path = self.label_prefix + '_train'
        label_test_path = self.label_prefix + '_test'
        feat_train_path = self.feat_prefix + '_train'
        feat_test_path = self.feat_prefix + '_test'
        if self.sort:
            label_train_path += '_sorted'
            label_test_path += '_sorted'
            feat_train_path += '_sorted'
            feat_test_path += '_sorted'
        else:
            label_train_path += '_shuffled'
            label_test_path += '_shuffled'
            feat_train_path += '_shuffled'
            feat_test_path += '_shuffled'
        print("Saving train labels ...")
        dump2npy(label_train_path, train_label, force=False)
        print("Saving test labels ...")
        dump2npy(label_test_path, test_label, force=False)
        print("Saving train features ...")
        dump2npy(feat_train_path, train_feat, force=False)
        print("Saving test features ...")
        dump2npy(feat_test_path, test_feat, force=False)

    def save2npy(self):
        print("pic_id matrix shape: ", self.pic_ids.shape)
        print("feature matrix shape: ", self.features.shape)
        print("label matrix shape: ", self.labels.shape)

        prefix = self.doc_prefix
        if self.clean_repeating_samples:
            prefix += '_cleaned'
        
        pic_id_path = prefix + '_pic_id'
        feature_path = prefix + '_feature'
        label_path = prefix + '_label'
        if self.sort:
            pic_id_path += '_sorted'
            feature_path += '_sorted'
            label_path += '_sorted'
        else:
            pic_id_path += '_shuffled'
            feature_path += '_shuffled'
            label_path += '_shuffled'
        
        print("Saving pic_ids ...")
        dump2npy(pic_id_path, self.pic_ids, force=False)
        print("Saving features ...")
        dump2npy(feature_path, self.features, force=False)
        print("Saving labels ...")
        dump2npy(label_path, self.labels, force=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--name",
                        type=str,
                        default='total',
                        help="image features")
    parser.add_argument("--prefix",
                        type=str,
                        default='./data',
                        help="prefix of dataset")
    parser.add_argument("--doc_amount",
                        type=int,
                        default=1,
                        help="number of documents")
    parser.add_argument("--start_index",
                        type=int,
                        default=0,
                        help="document start index")
    parser.add_argument("--dim",
                        type=int,
                        default=512,
                        help="dimension of feature")
    parser.add_argument("--no_normalize",
                        action='store_true',
                        help="whether to normalize feature")
    parser.add_argument("--sort",
                        action='store_true',
                        help="whether to sort the datasets via id labels")
    parser.add_argument("--sort_type",
                        type=str,
                        default='id_wise',
                        help="sort the dataset in what dimension: 'pic_wise' or 'id_wise' (default) ")
    args = parser.parse_args()

    ds = SHRLK_Dataset(name=args.name,
                      prefix=args.prefix,
                      doc_amount=args.doc_amount,
                      dim=args.dim,
                      normalize=args.no_normalize,
                      sort=args.sort,
                      sort_type=args.sort_type)
    ds.info()

    ds.save2npy()

    print('Done !!!')

