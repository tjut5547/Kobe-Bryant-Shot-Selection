#!/usr/local/bin/python

import os, sys
import pandas as pd
import xgboost as xgb

from collections import Counter
from sklearn.model_selection import cross_val_score

class Kobe_Bryant (object):
    def __init__(self, input_file, id_file):
        self._all_dataFrame = pd.read_csv(input_file)
        self._id = (pd.read_csv(id_file)['shot_id']).values
        self._row_replace = ['action_type', 'combined_shot_type', 'shot_type', 'shot_zone_area', 'season',\
                             'shot_zone_basic', 'shot_zone_range','game_date', 'matchup', 'opponent']

    def _replace_str_int(self):
        self.replace = {}
        for row in self._row_replace:
            tag_number = 0
            Dict = Counter(self._all_dataFrame[row])
            for key, value in Dict.items():
                tag_number += 1
                self._all_dataFrame[row] = self._all_dataFrame[row].replace(key, tag_number)

    def _get_train_test(self):
        del self._all_dataFrame['team_name']
        del self._all_dataFrame['team_id']

        self._replace_str_int()
        self._all_dataFrame.to_csv('../data/result.csv', index = False)
        self._all_dataFrame['shot_made_flag'] = self._all_dataFrame['shot_made_flag'].fillna(-1) # 对flag缺失的数据填充-1
        self._train_dataFrame = self._all_dataFrame[self._all_dataFrame.shot_made_flag != -1] # 取出训练集
        self._test_dataFrame = self._all_dataFrame[self._all_dataFrame.shot_made_flag == -1] # 取出测试集
        self._train_y = self._train_dataFrame['shot_made_flag'] # 训练集的结果
        
        del self._train_dataFrame['shot_made_flag']
        del self._test_dataFrame['shot_made_flag']
        del self._train_dataFrame['shot_id']
        del self._test_dataFrame['shot_id']

        print ("columns of train : ", self._train_dataFrame.columns)
        print ("columns of test :  ", self._test_dataFrame.columns)

        index_train = self._train_dataFrame.loc[self._train_dataFrame.index]
        index_test = self._test_dataFrame.loc[self._test_dataFrame.index]
        self._train_x = index_train.values # 获得训练集数据, array
        self._test_x = index_test.values # 获得测试集数据, array

    def _xgboost_predict(self):
        param = ((20, 30, 40, 50))
        regr = xgb.XGBRegressor(colsample_bytree=0.4,
                gamma=0.045,
                learning_rate=0.07,
                max_depth=20,
                min_child_weight=1.5,
                n_estimators=60,
                reg_alpha=0.65,
                reg_lambda=0.45,
                objective='binary:logistic',
                subsample=0.95)

        regr.fit(self._train_x, self._train_y, eval_metric = 'logloss')
        y_pred_xgb = regr.predict(self._test_x)
        return y_pred_xgb

    def _submission(self, y_final):
        # Preparing for submissions
        submission_df = pd.DataFrame(data = {'shot_id' : self._id, 'shot_made_flag': y_final})
        submission_df.to_csv('submission.csv', index=False)

    def main(self):
        self._get_train_test()
        y_final = self._xgboost_predict()
        self._submission(y_final)



if __name__ == '__main__':
    Kobe = Kobe_Bryant('../data/data.csv', '../data/ans.csv')
    Kobe.main()
