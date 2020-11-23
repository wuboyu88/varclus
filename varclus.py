"""This python script illustrates how to implement VarClus method which is
commonly used in SAS."""

__author__ = 'Boyu Wu'

import pandas as pd
import numpy as np
import random
import collections
import math
from factor_analyzer import Rotator
from copy import deepcopy


class VarClus(object):

    def __init__(self, df, feat_list=None, max_second_eig_val=1, max_cluster=None, n_rs=0):
        """
        变量聚类VarClus
        :param df: 数据集
        :param feat_list: 待聚类的变量
        :param max_second_eig_val: 第二大的主成分特征值最大值
        :param max_cluster: 类个数最大值
        :param n_rs: 随机排序的次数
        """
        self.df = df if feat_list is None else df[feat_list]
        self.feat_list = df.columns.tolist()
        self.max_second_eig_val = max_second_eig_val
        self.max_cluster = max_cluster
        self.n_rs = n_rs
        self.clusters = None
        self.corrs = None

    @staticmethod
    def correig(df, feat_list=None, n_pcs=2):
        """

        :param df: 数据集
        :param feat_list: 待聚类的变量
        :param n_pcs: 前n_pcs个主成分
        :return:
        """
        if feat_list is None:
            feat_list = df.columns.tolist()
        else:
            df = df[feat_list]

        if len(feat_list) <= 1:
            corr = [len(feat_list)]
            eigvals = [len(feat_list)] + [0] * (n_pcs - 1)
            eigvecs = np.array([[len(feat_list)]])
            varprops = [sum(eigvals)]
        else:
            # 协方差系数矩阵
            corr = np.corrcoef(df.values.T)

            # 矩阵分解得到特征值(按照从小到达顺序排列)和特征向量
            raw_eigvals, raw_eigvecs = np.linalg.eigh(corr)
            eigvals, eigvecs = raw_eigvals[::-1], raw_eigvecs[:, ::-1]

            # 只取前n_pcs个主成分
            eigvals, eigvecs = eigvals[:n_pcs], eigvecs[:, :n_pcs]
            varprops = eigvals / sum(raw_eigvals)

        corr_df = pd.DataFrame(corr, columns=feat_list, index=feat_list)
        return eigvals, eigvecs, corr_df, varprops

    @staticmethod
    def _calc_tot_var(df, *clusters):
        """
        每个cluster第一个主成分对应的方差求和
        :param df: 数据集
        :param clusters: 多个变量集合
        :return:
        """
        tot_var = sum([VarClus.correig(df[cluster])[0][0] for cluster in clusters if len(cluster) > 0])
        return tot_var

    @staticmethod
    def _reassign(df, cluster1, cluster2, feat_list=None):
        """
        重新分配
        :param df: 数据集
        :param cluster1: 变量集合1
        :param cluster2: 变量集合2
        :param feat_list: 待聚类的变量
        :return:
        """
        if feat_list is None:
            feat_list = cluster1 + cluster2

        init_var = VarClus._calc_tot_var(df, cluster1, cluster2)
        fin_cluster1, fin_cluster2 = deepcopy(cluster1), deepcopy(cluster2)
        check_var, max_var = (init_var,) * 2

        while True:

            for feat in feat_list:
                new_cluster1, new_cluster2 = deepcopy(fin_cluster1), deepcopy(fin_cluster2)
                if feat in new_cluster1:
                    new_cluster1.remove(feat)
                    new_cluster2.append(feat)
                elif feat in new_cluster2:
                    new_cluster1.append(feat)
                    new_cluster2.remove(feat)
                else:
                    continue

                new_var = VarClus._calc_tot_var(df, new_cluster1, new_cluster2)
                if new_var > check_var:
                    check_var = new_var
                    fin_cluster1, fin_cluster2 = deepcopy(new_cluster1), deepcopy(new_cluster2)

            if max_var == check_var:
                break
            else:
                max_var = check_var

        return fin_cluster1, fin_cluster2, max_var

    @staticmethod
    def _reassign_rs(df, cluster1, cluster2, n_rs=0):
        """
        如果分配给其他群集会增加整体的解释方差则重新分配
        :param df: 数据集
        :param cluster1: 变量集合1
        :param cluster2: 变量集合2
        :param n_rs: 随机排序的次数
        :return:
        """
        feat_list = cluster1 + cluster2
        fin_rs_cluster1, fin_rs_cluster2, max_rs_var = VarClus._reassign(df, cluster1, cluster2)

        for _ in range(n_rs):
            random.shuffle(feat_list)
            rs_cluster1, rs_cluster2, rs_var = VarClus._reassign(df, cluster1, cluster2, feat_list)
            if rs_var > max_rs_var:
                max_rs_var = rs_var
                fin_rs_cluster1, fin_rs_cluster2 = rs_cluster1, rs_cluster2

        return fin_rs_cluster1, fin_rs_cluster2, max_rs_var

    def varclus(self):
        """
        varclus的步骤
        :return:
        """
        ClusInfo = collections.namedtuple('ClusInfo', ['cluster', 'eigval1', 'eigval2', 'eigvecs', 'varprop'])
        c_eigvals, c_eigvecs, c_corrs, c_varprops = VarClus.correig(self.df[self.feat_list])

        self.corrs = c_corrs

        cluster0 = ClusInfo(cluster=self.feat_list,
                            eigval1=c_eigvals[0],
                            eigval2=c_eigvals[1],
                            eigvecs=c_eigvecs,
                            varprop=c_varprops[0]
                            )
        self.clusters = collections.OrderedDict([(0, cluster0)])

        while True:

            # 超过最大的cluster数目则停止
            if self.max_cluster is not None and len(self.clusters) >= self.max_cluster:
                break

            # 找出eigval2最大的cluster
            idx = max(self.clusters, key=lambda x: self.clusters.get(x).eigval2)

            # 如果最大的eigval2小于等于设定的阈值，则停止
            if self.clusters[idx].eigval2 <= self.max_second_eig_val:
                break

            # 分裂成两个簇的过程为：先计算该簇的前两个主成分，再进行斜交旋转，
            # 并把每个变量分配到旋转分量对应的簇里，分配的原则是变量与这个主成分相关系数的绝对值最大
            c_cluster = self.clusters[idx].cluster
            c_eigvals, c_eigvecs, c_corrs, _ = VarClus.correig(self.df[c_cluster])

            cluster1, cluster2 = [], []
            rotator = Rotator(method='quartimax')
            r_eigvecs = rotator.fit_transform(pd.DataFrame(c_eigvecs))

            comb_sigma1 = math.sqrt(np.dot(np.dot(r_eigvecs[:, 0], c_corrs.values), r_eigvecs[:, 0].T))
            comb_sigma2 = math.sqrt(np.dot(np.dot(r_eigvecs[:, 1], c_corrs.values), r_eigvecs[:, 1].T))

            for feat in c_cluster:

                comb_cov1 = np.dot(r_eigvecs[:, 0], c_corrs[feat].values.T)
                comb_cov2 = np.dot(r_eigvecs[:, 1], c_corrs[feat].values.T)

                corr_pc1 = comb_cov1 / comb_sigma1
                corr_pc2 = comb_cov2 / comb_sigma2

                if abs(corr_pc1) > abs(corr_pc2):
                    cluster1.append(feat)
                else:
                    cluster2.append(feat)

            fin_cluster1, fin_cluster2, _ = VarClus._reassign_rs(self.df, cluster1, cluster2, self.n_rs)
            c1_eigvals, c1_eigvecs, _, c1_varprops = VarClus.correig(self.df[fin_cluster1])
            c2_eigvals, c2_eigvecs, _, c2_varprops = VarClus.correig(self.df[fin_cluster2])

            self.clusters[idx] = ClusInfo(cluster=fin_cluster1,
                                          eigval1=c1_eigvals[0],
                                          eigval2=c1_eigvals[1],
                                          eigvecs=c1_eigvecs,
                                          varprop=c1_varprops[0]
                                          )
            self.clusters[len(self.clusters)] = ClusInfo(cluster=fin_cluster2,
                                                         eigval1=c2_eigvals[0],
                                                         eigval2=c2_eigvals[1],
                                                         eigvecs=c2_eigvecs,
                                                         varprop=c2_varprops[0]
                                                         )

    @property
    def info(self):
        """
        每个类对应的统计量
        :return:
        """
        cols = ['Cluster', 'N_Vars', 'Eig_Val1', 'Eig_Val2', 'VarProp']
        info_table = pd.DataFrame(columns=cols)

        n_row = 0
        for i, clusinfo in self.clusters.items():
            # repr函数让DataFrame显示的时候是整数1而不是浮点数1.0
            row = [repr(i), repr(len(clusinfo.cluster)), clusinfo.eigval1, clusinfo.eigval2, clusinfo.varprop]
            info_table.loc[n_row] = row
            n_row += 1

        return info_table

    @property
    def r_square(self):
        """
        每个变量对应的R2统计量，其中1-RS_Ratio越小越好，一般取每个类1-RS_Ratio最小的一个或几个变量作为该类的代表
        :return:
        """
        cols = ['Cluster', 'Variable', 'RS_Own', 'RS_Next_Closest', '1-RS_Ratio']
        rs_table = pd.DataFrame(columns=cols)

        sigmas = [math.sqrt(clusinfo.eigval1) for _, clusinfo in self.clusters.items()]

        n_row = 0
        for i, cluster_own in self.clusters.items():
            for feat in cluster_own.cluster:
                row = [i, feat]

                cov_own = np.dot(cluster_own.eigvecs[:, 0], self.corrs.loc[feat, cluster_own.cluster].values.T)
                rs_own = (cov_own / sigmas[i]) ** 2

                rs_others = []
                for j, cluster_other in self.clusters.items():
                    if j == i:
                        continue

                    cov_other = np.dot(cluster_other.eigvecs[:, 0],
                                       self.corrs.loc[feat, cluster_other.cluster].values.T)
                    rs = (cov_other / sigmas[j]) ** 2

                    rs_others.append(rs)

                rs_nc = max(rs_others) if len(rs_others) > 0 else 0
                row += [rs_own, rs_nc, (1 - rs_own) / (1 - rs_nc)]
                rs_table.loc[n_row] = row
                n_row += 1

        return rs_table


if __name__ == '__main__':
    demo_df = pd.read_csv('data/winequality-red.csv')
    demo_df.drop('quality', axis=1, inplace=True)
    demo_vc = VarClus(demo_df)
    demo_vc.varclus()
    print(demo_vc.info)
    print(demo_vc.r_square)
