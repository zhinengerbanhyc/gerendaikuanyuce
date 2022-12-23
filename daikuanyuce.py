import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import gc
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)


def workYearDIc(x):
    if str(x) == 'nan':
        return 0
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


def findDig2(x):
    if x >= pd.to_datetime('2021-01-01'):
        t = '20' + str(x)[8:10] + '-' + str(x)[5:7] + '-01'
        # print('t=', t)
        return pd.to_datetime(t)
    return x


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


print('######################第0步数据预处理#############################')
train_data = pd.read_csv('datasets/train_public.csv')
test_public = pd.read_csv('datasets/test_public.csv')
train_inte = pd.read_csv('datasets/train_internet.csv')

# 缺失值补充
loss_numerical_feas = ['recircle_u', 'pub_dero_bankrup', 'debt_loan_ratio']
f_feas = ['f0', 'f1', 'f2', 'f3', 'f4']

train_data[loss_numerical_feas] = train_data[loss_numerical_feas].fillna(train_data[loss_numerical_feas].median())
train_data[f_feas] = train_data[f_feas].fillna(train_data[f_feas].median())
train_data['post_code'] = train_data['post_code'].fillna(train_data['post_code'].mode()[0])
train_data.loc[train_data['debt_loan_ratio'] <= 0, 'debt_loan_ratio'] = 0
for f in f_feas:
    train_data[f'industry_to_mean_{f}'] = train_data.groupby('industry')[f].transform('mean')

test_public[loss_numerical_feas] = test_public[loss_numerical_feas].fillna(test_public[loss_numerical_feas].median())
test_public['post_code'] = test_public['post_code'].fillna(test_public['post_code'].mode()[0])
test_public.loc[test_public['debt_loan_ratio'] <= 0, 'debt_loan_ratio'] = 0
test_public[f_feas] = test_public[f_feas].fillna(test_public[f_feas].median())
for f in f_feas:
    test_public[f'industry_to_mean_{f}'] = test_public.groupby('industry')[f].transform('mean')

train_inte[loss_numerical_feas] = train_inte[loss_numerical_feas].fillna(train_inte[loss_numerical_feas].median())
train_inte['post_code'] = train_inte['post_code'].fillna(train_inte['post_code'].mode()[0])
train_inte['title'] = train_inte['title'].fillna(train_inte['title'].mode()[0])
train_inte.loc[train_inte['debt_loan_ratio'] <= 0, 'debt_loan_ratio'] = 0
train_inte[f_feas] = train_inte[f_feas].fillna(train_inte[f_feas].median())
for f in f_feas:
    train_inte[f'industry_to_mean_{f}'] = train_inte.groupby('industry')[f].transform('mean')
class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}

timeMax = pd.to_datetime('1-Dec-21')
train_data['work_year'] = train_data['work_year'].map(workYearDIc)

test_public['work_year'] = test_public['work_year'].map(workYearDIc)

train_data['class'] = train_data['class'].map(class_dict)
test_public['class'] = test_public['class'].map(class_dict)

train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDig))
test_public['earlies_credit_mon'] = pd.to_datetime(test_public['earlies_credit_mon'].map(findDig))

train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_data.loc[train_data[
                                                                                                      'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = test_public.loc[test_public[
                                                                                                         'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])

# Internet数据处理
train_inte['work_year'] = train_inte['work_year'].map(workYearDIc)
train_inte['class'] = train_inte['class'].map(class_dict)

train_inte['earlies_credit_mon'] = pd.to_datetime(train_inte['earlies_credit_mon'])
train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])

train_data['issue_date_month'] = train_data['issue_date'].dt.month
train_data['issue_date_year'] = train_data['issue_date'].dt.year
test_public['issue_date_month'] = test_public['issue_date'].dt.month
test_public['issue_date_year'] = test_public['issue_date'].dt.year
train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
test_public['issue_date_dayofweek'] = test_public['issue_date'].dt.dayofweek

train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
test_public['earliesCreditMon'] = test_public['earlies_credit_mon'].dt.month
train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
test_public['earliesCreditYear'] = test_public['earlies_credit_mon'].dt.year

train_inte['issue_date_month'] = train_inte['issue_date'].dt.month
train_inte['issue_date_dayofweek'] = train_inte['issue_date'].dt.dayofweek
train_inte['issue_date_year'] = train_inte['issue_date'].dt.year

train_inte['earliesCreditMon'] = train_inte['earlies_credit_mon'].dt.month
train_inte['earliesCreditYear'] = train_inte['earlies_credit_mon'].dt.year

###########################编码处理

cat_cols = ['employer_type', 'industry']
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])
    train_inte[col] = lbl.transform(train_inte[col])

col_to_drop = ['issue_date', 'earlies_credit_mon']
train_data = train_data.drop(col_to_drop, axis=1)
test_public = test_public.drop(col_to_drop, axis=1)
train_inte = train_inte.drop(col_to_drop, axis=1)

###参考天池的sub_clss维度信息，这个维度应该比较关键不能直接干掉而需要补充
print(train_inte['sub_class'].unique())
print(train_data['class'].unique())
print(test_public['class'].unique())
'''
['A1' 'A2' 'A3' 'A4' 'A5' 'B1' 'B2' 'B3' 'B4' 'B5' 'C1' 'C2' 'C3' 'C4' 'C5' 'D1' 'D2' 'D3' 'D4' 'D5' 'E1' 'E2' 
'E3' 'E4' 'E5' 'F1' 'F2' 'F3' 'F4' 'F5' 'G1' 'G2' 'G3' 'G4' 'G5']
共七大类其中每大类中有五小类，采用聚类方案
'''


def feature_Kmeans(data, label):
    mms = MinMaxScaler()
    feats = [f for f in data.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    data = data[feats]
    mmsModel = mms.fit_transform(data.loc[data['class'] == label])
    clf = KMeans(5, random_state=2021)
    pre = clf.fit(mmsModel)
    test = pre.labels_
    final_data = pd.Series(test, index=data.loc[data['class'] == label].index)
    if label == 1:
        final_data = final_data.map({0: 'A1', 1: 'A2', 2: 'A3', 3: 'A4', 4: 'A5'})
    elif label == 2:
        final_data = final_data.map({0: 'B1', 1: 'B2', 2: 'B3', 3: 'B4', 4: 'B5'})
    elif label == 3:
        final_data = final_data.map({0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5'})
    elif label == 4:
        final_data = final_data.map({0: 'D1', 1: 'D2', 2: 'D3', 3: 'D4', 4: 'D5'})
    elif label == 5:
        final_data = final_data.map({0: 'E1', 1: 'E2', 2: 'E3', 3: 'E4', 4: 'E5'})
    elif label == 6:
        final_data = final_data.map({0: 'F1', 1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5'})
    elif label == 7:
        final_data = final_data.map({0: 'G1', 1: 'G2', 2: 'G3', 3: 'G4', 4: 'G5'})
    return final_data


# 训练集合并
train_data1 = feature_Kmeans(train_data, 1)
train_data2 = feature_Kmeans(train_data, 2)
train_data3 = feature_Kmeans(train_data, 3)
train_data4 = feature_Kmeans(train_data, 4)
train_data5 = feature_Kmeans(train_data, 5)
train_data6 = feature_Kmeans(train_data, 6)
train_data7 = feature_Kmeans(train_data, 7)
train_dataall = pd.concat(
    [train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7]).reset_index(drop=True)
train_data['sub_class'] = train_dataall
# 测试集合并
test_data1 = feature_Kmeans(test_public, 1)
test_data2 = feature_Kmeans(test_public, 2)
test_data3 = feature_Kmeans(test_public, 3)
test_data4 = feature_Kmeans(test_public, 4)
test_data5 = feature_Kmeans(test_public, 5)
test_data6 = feature_Kmeans(test_public, 6)
test_data7 = feature_Kmeans(test_public, 7)
test_dataall = pd.concat(
    [test_data1, test_data2, test_data3, test_data4, test_data5, test_data6, test_data7]).reset_index(drop=True)
test_public['sub_class'] = test_dataall

cat_cols = ['sub_class']
for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])
    train_inte[col] = lbl.transform(train_inte[col])

#######尝试新的特征######################
train_data['post_code_interst_mean'] = train_data.groupby(['post_code'])['interest'].transform('mean')
train_inte['post_code_interst_mean'] = train_inte.groupby(['post_code'])['interest'].transform('mean')
test_public['post_code_interst_mean'] = test_public.groupby(['post_code'])['interest'].transform('mean')

train_data['industry_mean_interest'] = train_data.groupby(['industry'])['interest'].transform('mean')
train_inte['industry_mean_interest'] = train_inte.groupby(['industry'])['interest'].transform('mean')
test_public['industry_mean_interest'] = test_public.groupby(['industry'])['interest'].transform('mean')

train_data['recircle_u_b_std'] = train_data.groupby(['recircle_u'])['recircle_b'].transform('std')
test_public['recircle_u_b_std'] = test_public.groupby(['recircle_u'])['recircle_b'].transform('std')
train_inte['recircle_u_b_std'] = train_inte.groupby(['recircle_u'])['recircle_b'].transform('std')
train_data['early_return_amount_early_return'] = train_data['early_return_amount'] / train_data['early_return']
test_public['early_return_amount_early_return'] = test_public['early_return_amount'] / test_public['early_return']
train_inte['early_return_amount_early_return'] = train_inte['early_return_amount'] / train_inte['early_return']
# 可能出现极大值和空值
train_data['early_return_amount_early_return'][np.isinf(train_data['early_return_amount_early_return'])] = 0
test_public['early_return_amount_early_return'][np.isinf(test_public['early_return_amount_early_return'])] = 0
train_inte['early_return_amount_early_return'][np.isinf(train_inte['early_return_amount_early_return'])] = 0
# 还款利息
train_data['total_loan_monthly_payment'] = train_data['monthly_payment'] * train_data['year_of_loan'] * 12 - train_data[
    'total_loan']
test_public['total_loan_monthly_payment'] = test_public['monthly_payment'] * test_public['year_of_loan'] * 12 - \
                                            test_public['total_loan']
train_inte['total_loan_monthly_payment'] = train_inte['monthly_payment'] * train_inte['year_of_loan'] * 12 - train_inte[
    'total_loan']


#########################################################
# 目标编码
def gen_target_encoding_feats(train, train_inte, test, encode_cols, target_col, n_fold=10):
    '''生成target encoding特征'''
    # for training set - cv
    tg_feats = np.zeros((train.shape[0], len(encode_cols)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train[encode_cols], train[target_col])):
        df_train, df_val = train.iloc[train_index], train.iloc[val_index]
        for idx, col in enumerate(encode_cols):
            target_mean_dict = df_train.groupby(col)[target_col].mean()
            df_val[f'{col}_mean_target'] = df_val[col].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{col}_mean_target'].values

    for idx, encode_col in enumerate(encode_cols):
        train[f'{encode_col}_mean_target'] = tg_feats[:, idx]

    # for train_inte set - cv
    tg_feats = np.zeros((train_inte.shape[0], len(encode_cols)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train_inte[encode_cols], train_inte[target_col])):
        df_train, df_val = train_inte.iloc[train_index], train_inte.iloc[val_index]
        for idx, col in enumerate(encode_cols):
            target_mean_dict = df_train.groupby(col)[target_col].mean()
            df_val[f'{col}_mean_target'] = df_val[col].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{col}_mean_target'].values
    for idx, encode_col in enumerate(encode_cols):
        train_inte[f'{encode_col}_mean_target'] = tg_feats[:, idx]

    # for testing set
    for col in encode_cols:
        target_mean_dict = train.groupby(col)[target_col].mean()
        test[f'{col}_mean_target'] = test[col].map(target_mean_dict)

    return train, train_inte, test


features = ['house_exist', 'debt_loan_ratio', 'industry', 'title']
train_data, train_inte, test_public = gen_target_encoding_feats(train_data, train_inte, test_public, features,
                                                                'isDefault', n_fold=10)


########################样本扩充
def fiterDataModel(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = LGBMClassifier(
            n_estimators=4000,
            # learning_rate=0.08,
            learning_rate=0.06,
            num_leaves=2 ** 5,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y.astype(int)), (val_x, val_y.astype(int))],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 40
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['isDefault'] = sub_preds

    return test_[['loan_id', 'isDefault']]


tr_cols = set(train_data.columns)
same_col = list(tr_cols.intersection(set(train_inte.columns)))
train_inteSame = train_inte[same_col].copy()
Inte_add_cos = list(tr_cols.difference(set(same_col)))
for col in Inte_add_cos:
    train_inteSame[col] = np.nan
y = train_data['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
IntePre = fiterDataModel(train_data, train_inteSame, y, folds)
IntePre['isDef'] = train_inte['isDefault']
print(roc_auc_score(IntePre['isDef'], IntePre.isDefault))
## 选择阈值0.05，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
InteId = IntePre.loc[IntePre.isDefault < 0.5, 'loan_id'].tolist()
train_inteSame['isDefault'] = train_inte['isDefault']
use_te = train_inteSame[train_inteSame.loan_id.isin(InteId)].copy()
data = pd.concat([train_data, test_public, use_te]).reset_index(drop=True)
print('dataShape:', len(data))


###############开始最后一步的训练
def XGBModel(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = XGBClassifier(eval_metric='auc', max_depth=5, alpha=0.3, reg_lambda=0.3, subsample=0.8,
                            colsample_bylevel=0.867, objective='binary:logistic', use_label_encoder=False,
                            learning_rate=0.08, n_estimators=4000, min_child_weight=2, tree_method='hist',
                            n_jobs=-1)
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 40
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], ntree_limit=clf.best_ntree_limit)[:, 1] / folds_.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['isDefault'] = sub_preds
    return test_[['loan_id', 'isDefault']]


def LGBModel(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = LGBMClassifier(
            n_estimators=4000,
            # learning_rate=0.08,
            learning_rate=0.06,
            num_leaves=2 ** 5,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y.astype(int)), (val_x, val_y.astype(int))],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 40
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['isDefault'] = sub_preds
    return test_[['loan_id', 'isDefault']]


train = data[data['isDefault'].notna()]
test = data[data['isDefault'].isna()]
y = train['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
test_preds = LGBModel(train, test, y, folds)
test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('baseline891_参数自动优化.csv', index=None)
