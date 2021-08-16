import os
import numpy as np
from utils.misc import TextColors, l2norm, read_probs, read_label_npy, dump2npy
# from utils import (load_data, dump_data, mkdir_if_no_exists, Timer)

class SplitDataset():

    def __init__(self,
                 name,
                 prefix='data',
                 dim=512,
                 normalize=False,
                 sort=False,
                 verbose=True):
        self.name = name
        self.dtype = np.float32
        self.dim = dim
        self.normalize = normalize
        self.sort = sort
        if not os.path.exists(prefix):
            raise FileNotFoundError(
                'folder({}) does not exist.'.format(prefix))
        self.prefix = prefix
        self.label_prefix = os.path.join(prefix, 'labels', name)
        self.label_path = self.label_prefix + '.npy'
        if os.path.isfile(self.label_path):
            self.lb2idxs, self.idx2lb, self.labels = read_label_npy(self.label_path,
                                                  verbose=verbose)
            self.inst_num = len(self.idx2lb)
            self.cls_num = len(self.lb2idxs)
        else:
            print(
                'meta file not found: {}.\n'
                'init `lb2idxs` and `idx2lb` as None.'
                .format(self.label_path))
            self.lb2idxs, self.idx2lb = None, None
            self.inst_num, self.cls_num = -1, -1
        self.feat_prefix = os.path.join(prefix, 'features', name)
        self.feat_path = self.feat_prefix + '.npy'
        self.features = read_probs(self.feat_path,
                                   self.inst_num,
                                   dim,
                                   self.dtype,
                                   verbose=verbose)
        if self.normalize:
            self.features = l2norm(self.features)

        if self.sort:
            print("Sort Enabled!")
            self.sort_dataset()
        else:
            print("Shuffle Enabled!")
            self.shuffle_dataset()

    def info(self):
        print('name:{}{}{}\ninst_num:{}\ncls_num:{}\ndim:{}\n'
              'feat_path:{}\nnormalization:{}{}{}\ndtype:{}'.format(
                  TextColors.OKGREEN, self.name, TextColors.ENDC,
                  self.inst_num, self.cls_num, self.dim, self.feat_path,
                  TextColors.FATAL, self.normalize, TextColors.ENDC,
                  self.dtype))

    def sort_dataset(self):
        c = np.c_[self.features.reshape(len(self.features), -1), self.labels.reshape(len(self.labels), -1)]
        c = c[c[:,-1].argsort()]
        self.features = c[:,:self.features.size//len(self.features)].reshape(self.features.shape).astype('float32')
        self.labels = c[:,self.features.size//len(self.features):].reshape(self.labels.shape).astype('int32')

    def shuffle_dataset(self):
        c = np.c_[self.features.reshape(len(self.features), -1), self.labels.reshape(len(self.labels), -1)]
        np.random.shuffle(c)
        self.features = c[:,:self.features.size//len(self.features)].reshape(self.features.shape).astype('float32')
        self.labels = c[:,self.features.size//len(self.features):].reshape(self.labels.shape).astype('int32')

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
    args = parser.parse_args()

    ds = SplitDataset(name=args.name,
                      prefix=args.prefix,
                      dim=args.dim,
                      normalize=args.no_normalize,
                      sort=args.sort)
    ds.info()

    ds.split_dataset()



