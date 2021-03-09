# -*-coding:utf-8-*-
# date: 2021/01/14 16:12
# author Jiahui.Xu

import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import sklearn
import joblib
from tqdm import tqdm
import re
import time
from collections import defaultdict
import sys
import pprint
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sko.PSO import PSO
import copy

class database():
    def __init__(self, addr, port, passwd, db_name):
        self.addr = addr
        self.port = port
        self.passwd = passwd
        self.database_name = db_name
        
    def connect_msyql_engine(self):
        engine = create_engine('mysql+pymysql://root:{}@{}:{}/{}'.format(self.passwd, self.addr, self.port, self.database_name))
        return engine


class DataLoader():
    def __init__(self, db_connector):
        self.db_connector = db_connector
        self.input_min_max_scaler = preprocessing.MinMaxScaler()
        self.output_min_max_scaler = preprocessing.MinMaxScaler()

    def load(self, date_start, date_end, pressure_name, step=7, test_size=0.4):
        self.date_start = date_start
        self.date_end = date_end
        self.pressure_name = pressure_name
        self.step = step
        self.raw_timestamp = self._get_time_raw_stamp()
        pressure_data = self._get_pressure_data()
        # flow_data已经包含了时间数据
        flow_data = self._get_flow_data()
        # 将flow_data构造过去时间数据
        flow_data = self._shift(flow_data, pressure_data, step=7)
        print(flow_data.columns.values.tolist())
        print(flow_data.columns.values.tolist()[-11])
        print(flow_data.columns.values.tolist()[-10])
        print(flow_data.columns.values.tolist()[-9])
        # 删除nan
        flow_data.fillna(method='ffill', inplace=True)
        pressure_data.fillna(method='ffill', inplace=True)
        # 归一化数据
        norm_flow_data = self.input_min_max_scaler.fit_transform(flow_data)
        norm_pressure_data = self.output_min_max_scaler.fit_transform(pressure_data)
        print(flow_data)
        print(pressure_data)
        # 当infer的时候若数据长度不一致，需要截取最小的数据长度作为整体的最小的数据长度
        min_length = min(len(norm_flow_data), len(norm_pressure_data))
        norm_flow_data = norm_flow_data[0:min_length]
        norm_pressure_data = norm_pressure_data[0:min_length]
        self.raw_timestamp = self.raw_timestamp.iloc[0:min_length]
        print("长度：", len(self.raw_timestamp))
        if test_size == 1:  #当infer的时候
            print("此时正在infer")
            return {}, {}, norm_flow_data, norm_pressure_data
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(norm_flow_data, norm_pressure_data, test_size=test_size)
        return x_train, y_train, x_test, y_test

    def _corr(self, x, y):
        norm_x = self.min_max_scaler.fit_transform(x)
        norm_y = self.min_max_scaler.fit_transform(y)
        corelation = pd.concat([pd.DataFrame(norm_x), pd.DataFrame(norm_y)], axis=1)
        print("输入输出的相关性矩阵为：", corelation.corr())

    def _shift(self, x, y, step=7):
        df_top = pd.DataFrame({self.pressure_name:[y.iloc[0, 0] for i in range(step-1)]})
        df_button = pd.DataFrame({self.pressure_name:[y.iloc[-1, -1] for i in range(step-1)]})
        outcome = df_top.append(y, ignore_index=True).append(df_button, ignore_index=True)
        for i in range(step-1,0,-1):
            x['t-{}'.format(i)] = outcome.iloc[step-1-i:len(y)+step-1-i].reset_index(drop=True)
        return x

    def _get_pressure_data(self):
        pressure_data = defaultdict()
        pressure_table_list = self._get_pressure_tablename()
        for each in tqdm(pressure_table_list):
            sql_query = "select time, value from {} where time >= '{}' and time <= '{}'".format(each, self.date_start, self.date_end)
            data = pd.read_sql_query(sql_query, self.db_connector)
            data.interpolate(inplace=True)
            pressure_data[each] = data.value
            pressure_data = pd.DataFrame(pressure_data)
        return pressure_data

    def _get_flow_data(self):
        flow_data = defaultdict()
        flow_table_list = self._get_flow_tablename()
        time_stamp = []
        for each in tqdm(flow_table_list):
            sql_query = "select time, value from {} where time >= '{}' and time <= '{}'".format(each, self.date_start, self.date_end)
            data = pd.read_sql_query(sql_query, self.db_connector)
            data.interpolate(inplace=True)
            flow_data[each] = data.value
            if not time_stamp:
                times = data.time 
                time_stamp = []
                for i in times:
                    time_stamp.append(i.hour*60+i.minute)
        flow_data = pd.DataFrame(flow_data)
        flow_data['time'] = time_stamp
        return flow_data.iloc[0:-1]

    def _get_time_raw_stamp(self):
        sql_query = "select time from {} where time >= '{}' and time <= '{}'".format(self.pressure_name, self.date_start, self.date_end)
        return pd.read_sql_query(sql_query, self.db_connector)

    def _get_pressure_tablename(self):
        pressure_table_list = list()
        pressure_table_list.append(self.pressure_name)
        #pressure_table_list = ['南草压力', '宜昌宜昌压力', '展览展览压力', '本部本部压力', '瑞南压力', '瑞金压力', '红星红星压力']
        return pressure_table_list

    def _get_flow_tablename(self):
        flow_table_list = []
        with open('./tablelist.txt', 'r') as f:
            for line in f.readlines():
                if line.split()[0].endswith("瞬时流量"):
                    flow_table_list.append(line.split()[0])
        return flow_table_list


class PressureModel():
    def __init__(self, name, dataloader):
        self.name = name
        self.dataloader = dataloader

    def train(self, x_train, y_train):
        self.model = MLPRegressor()
        self.model.fit(x_train, y_train)

    def dump(self, path):
        joblib.dump(self.model, './{}/{}.pkl'.format(path, self.name))
    
    def load(self, path):
        self.model = joblib.load(path)

    def test(self, x_test, y_test):
        # score
        self.score = self.model.score(x_test, y_test)
        print("\033[1;31m 系统模型的性能为：\033[0m", self.score)
        # RMSE
        self.rmse = np.sqrt(mean_squared_error(self.dataloader.output_min_max_scaler.inverse_transform(self.model.predict(x_test).reshape(-1, 1)), self.dataloader.output_min_max_scaler.inverse_transform(y_test)))
        print("\033[1;32m 系统模型的RMSE为：\033[0m", self.rmse)
        # MAE
        self.mae = mean_absolute_error(self.dataloader.output_min_max_scaler.inverse_transform(self.model.predict(x_test).reshape(-1, 1)), self.dataloader.output_min_max_scaler.inverse_transform(y_test))
        print("\033[1;33m 系统模型的MAE为：\033[0m", self.mae)
        # 平均偏差率
        real = self.dataloader.output_min_max_scaler.inverse_transform(y_test)
        predicts = self.dataloader.output_min_max_scaler.inverse_transform(self.model.predict(x_test).reshape(-1, 1))
        self.deviation_rate = np.mean(np.abs(real-predicts)/predicts)
        print("\033[1;34m 系统模型的平均偏差率为：\033[0m", self.deviation_rate)
        # 根据时间stamp构造DF，用于绘图
        real_predict_dataframe = pd.DataFrame({'time':self.dataloader.raw_timestamp['time'].to_numpy(), 'real':[i[0] for i in real], 'predict':[i[0] for i in predicts]})

        print(real_predict_dataframe)
        return self.score, self.rmse, self.mae, self.deviation_rate, real_predict_dataframe


def train(pressure_name: str):
    # 设置数据库
    addr = "192.168.1.55"
    port = 3306
    passwd = "sjtu213"
    db_name = "sjtu_water"
    db = database(addr, port, passwd, db_name)
    db_connector = db.connect_msyql_engine()

    # 获取数据
    date_start = '2020-12-01'
    date_end = '2020-12-31'
    dataloader = DataLoader(db_connector)
    x_train, y_train, x_test, y_test = dataloader.load(date_start, date_end, pressure_name)

    # 进行模型训练
    model = PressureModel(pressure_name, dataloader)
    model.train(x_train, y_train)
    model.test(x_test, y_test)
    model.dump('pressure_models')

def infer(date_infer_start: str, data_infer_end: str, pressure_name: str):
    # 设置数据库
    addr = "192.168.1.55"
    port = 3306
    passwd = "sjtu213"
    db_name = "sjtu_water"
    db = database(addr, port, passwd, db_name)
    db_connector = db.connect_msyql_engine()

    # 获取数据
    date_start = date_infer_start
    date_end = data_infer_end
    dataloader = DataLoader(db_connector)
    x_train, y_train, x_test, y_test = dataloader.load(date_start, date_end, pressure_name, test_size=1)

    # 利用训练好的模型进行infer，infer一段连续时间的数据
    model = PressureModel(pressure_name, dataloader)
    model.load('./pressure_models/{}.pkl'.format(pressure_name))
    score, rmse, mae, deviation_rate, real_predict_dataframe = model.test(x_test, y_test)
    return score, rmse, mae, deviation_rate, real_predict_dataframe

def optimize(date_infer_start: str, data_infer_end: str, pressure_name: str, lowerbound: list, upperbound: list):
    start_time = time.time()
    # 设置数据库
    addr = "192.168.1.55"
    port = 3306
    passwd = "sjtu213"
    db_name = "sjtu_water"
    db = database(addr, port, passwd, db_name)
    db_connector = db.connect_msyql_engine()

    # 优化结果存储
    optimized_pressure = []
    real_pressure = []
    ## 用于存储优化后的控制变量
    fx_flow = []
    rg_flow_1 = []
    rg_flow_2 = []
    ## 用于存储优化前的控制变量
    fx_flow_raw = []
    rg_flow_1_raw = []
    rg_flow_2_raw = []

    # 获取数据
    date_start = date_infer_start
    date_end = data_infer_end
    dataloader = DataLoader(db_connector)
    x_train, y_train, x_test, y_test = dataloader.load(date_start, date_end, pressure_name, test_size=1)

    # 利用训练好的模型进行infer，infer一段连续时间的数据
    model = PressureModel(pressure_name, dataloader)
    model.load('./pressure_models/{}.pkl'.format(pressure_name))
    #score, rmse, mae, deviation_rate, real_predict_dataframe = model.test(x_test, y_test)

    for i in range(0, 1439):
        # 取其中一行的数据
        x_input = copy.deepcopy(x_test[i])
        y_output = copy.deepcopy(y_test[i])
        #print(50*'#', x_input)
        #print(50*'#', y_output)
        #print(x_test, y_test)
        # 复兴水库泵站瞬时流量 [-11]
        # 广场水库泵站1瞬时流量 [-10]
        # 广场水库泵站2瞬时流量 [-9]

        def lose_func(input):
            nonlocal x_input, y_output, model, i
            fx, gc1, gc2 = input
            x_input[-11] = fx
            x_input[-10] = gc1
            x_input[-9] = gc2
            #print(50*'#', x_input)
            #print("y_test", y_test)
            #print("")
            #print("原始的真实值：", model.dataloader.min_max_scaler.inverse_transform([y_test]))
            #print("将模型的输出转化为真实值：", model.dataloader.min_max_scaler.inverse_transform(model.model.predict([x_test]).reshape(-1, 1)))
            #print(50*'@', model.model.predict([x_input]))
            model_output = model.dataloader.output_min_max_scaler.inverse_transform(model.model.predict([x_input]).reshape(-1, 1))
            deviation = model_output - 0.5*(upperbound['hx'][i]+lowerbound['hx'][i])
            #print("偏差的平方是：", deviation**2)
            #print(upperbound['hx'][0])
            #print(lowerbound['hx'][0])
            return deviation**2

        pso = PSO(func=lose_func, n_dim=3, pop=40, max_iter=150, lb=[0.0, 0.0, 0.0], ub=[1.0, 1.0, 1.0], w=0.8, c1=0.5, c2=0.5)
        pso.run()
        print("\033[1;31m 正在优化第{}个数据 \033[0m".format(i))
        print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        x_input[-11] = pso.gbest_x[0]
        x_input[-10] = pso.gbest_x[1]
        x_input[-9] = pso.gbest_x[2]
        #model_input = model.dataloader.input_min_max_scaler.inverse_transform([x_input])   #这个反归一化有bug
        #print(x_input)
        model_output = model.dataloader.output_min_max_scaler.inverse_transform(model.model.predict([x_input]).reshape(-1, 1))
        real_output = model.dataloader.output_min_max_scaler.inverse_transform([y_output])
        print("\033[1;34m 优化后的压力是{} - {} - {} 原始的真实数据是：{}\033[0m".format(lowerbound['hx'][i], model_output, upperbound['hx'][i], real_output))
        #print(model_input)
        #return score, rmse, mae, deviation_rate, real_predict_dataframe
        #lose_func([0.0, 0.03038594, 0.19207131])
        #break
        real_pressure.append(real_output.tolist()[0][0])
        optimized_pressure.append(model_output.tolist()[0][0])
        # 将优化后的流量数据保存下来
        # 复兴水库泵站瞬时流量 [-11]
        # 广场水库泵站1瞬时流量 [-10]
        # 广场水库泵站2瞬时流量 [-9]
        new_x = model.dataloader.input_min_max_scaler.inverse_transform([x_input]).reshape(-1, 1)
        #print("\033[1;32m 原始的输入数据是：{} \033[0m".format(new_x))
        #print(new_x[-11])
        #print(new_x[-10])
        #print(new_x[-9])
        fx_flow.append(new_x[-11][0])
        rg_flow_1.append(new_x[-10][0])
        rg_flow_2.append(new_x[-9][0])
        fx_flow_raw.append(model.dataloader.input_min_max_scaler.inverse_transform([x_test[i]]).reshape(-1, 1)[-11][0])
        rg_flow_1_raw.append(model.dataloader.input_min_max_scaler.inverse_transform([x_test[i]]).reshape(-1, 1)[-10][0])
        rg_flow_2_raw.append(model.dataloader.input_min_max_scaler.inverse_transform([x_test[i]]).reshape(-1, 1)[-9][0])
        #if i == 3:
            #break
    #print(optimized_pressure)
    #print(fx_flow)
    #print(rg_flow_1)
    #print(rg_flow_2)
    #print(fx_flow_raw)
    #print(rg_flow_1_raw)
    #print(rg_flow_2_raw)
    with open("./datas/fx.txt", 'w+') as f:
        for i in range(len(fx_flow)):
            f.write(str(fx_flow_raw[i])+','+str(fx_flow[i])+'\n')
    with open("./datas/rg1.txt", 'w+') as f:
        for i in range(len(fx_flow)):
            f.write(str(rg_flow_1_raw[i])+','+str(rg_flow_1[i])+'\n')
    with open("./datas/rg2.txt", 'w+') as f:
        for i in range(len(fx_flow)):
            f.write(str(rg_flow_2_raw[i])+','+str(rg_flow_2[i])+'\n')
    with open("./datas/pressure.txt", 'w+') as f:
        for i in range(len(fx_flow)):
            f.write(str(real_pressure[i])+','+str(optimized_pressure[i])+'\n')
    print("总共花费的时间：{} 秒".format(time.time()-start_time))
    # 运行一次花费676秒
    return optimized_pressure


if __name__ == "__main__":
    for i in ['南草压力', '宜昌宜昌压力', '展览展览压力', '本部本部压力', '瑞南压力', '瑞金压力', '红星红星压力']:
        train(i)