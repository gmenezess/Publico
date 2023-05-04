import numpy as np
import pandas as pd
import sys
#print("Current version of Python is ", sys.version)
import requests
from requests.auth import HTTPBasicAuth
import json

import quantstats as qs
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import update
import timeit
import time
import string
import pygsheets
import seaborn as sns
import seaborn as sns; sns.set_theme()


"""     user = 'J07wqQSo7M'
    password = 'VWNeDUyKUseOp0S4VAPHl02Yxyw1ZpGD'
    res = requests.get('https://api.smartbraincloud.net:8099/api/v1/auth/login', verify=False, auth=HTTPBasicAuth(user, password))

    res_string = res.text
    token = (res_string.split(",")[0])[16:-1]
    suffix = 'Bearer '
    access_token = suffix + token

    #get_clientes(access_token)

    cod_user = 11307

    Estratégias = ['Caixa', 'VALORES A  LIQUIDAR', 'Inflação', 'Juros Nominal', 'Ações Brasil', 
        'Alternativos', 'Renda Variável Internacional', 'Moedas', 'Multimercado', 'Commodity', 'Real Estate']    
    tickers_asset_class = ['Pre_Caixa', 'Pre_Caixa', 
        'IMAB', 'IRFM', 'BOVA11.SA', 'BOVA11.SA', 'IVVB11.SA', 'Moedas', 'IHFA', 'GOLD11.SA', 'XFIX11.SA']

    df2_['Name'] = df2_['descricao'].replace(['Caixa', 'VALORES A  LIQUIDAR', 'Inflação', 'Juros Nominal', 'Ações Brasil', 
        'Alternativos', 'Renda Variável Internacional', 'Moedas', 'Multimercado', 'Commodity', 'Real Estate'], tickers_asset_class)

    df_allocate, Total_Assets = run_single_cliente(access_token, cod_user, end_dt, tickers_asset_class)
    print('Total_Assets:', Total_Assets)
    df_allocate """


def get_clientes(access_token):
    
    url = "https://api.smartbraincloud.net:8099/api/v1/Consulta/Cliente"
    head = {"Authorization": access_token}
    data= {'nomeCliente': "" }
    resp = requests.post(url, headers=head, json=data)
    json_data = json.loads(resp.text) 
    clientes = pd.read_json(resp.text)    
    clientes = pd.concat([clientes.drop(['cliente'], axis=1), clientes['cliente'].apply(pd.Series)], axis=1)
    clientes = clientes[['nomeCliente', 'codigoCliente']]    
    cod_users = clientes['codigoCliente'].tolist()
    
    return clientes

def run_single_cliente(access_token, cod_user, data_ref, tickers_asset_class):

    #tickers_asset_class = ['Pre_Caixa', 'Pre_Caixa', 
    #    'IMAB', 'IRFM', 'BOVA11.SA', 'BOVA11.SA', 'IVVB11.SA', 'Moedas', 'IHFA', 'GOLD11.SA', 'XFIX11.SA']

    # Parâmetros de acesso
    url = "https://api.smartbraincloud.net:8099/api/v1/AlocacaoFinanceira/Estrategia"
    head = {"Authorization": access_token}
    
    data_ref_st = data_ref.strftime('%Y%m%d')
    data= {'codigoUsuario': cod_user, "dataReferencia": data_ref_st }
    resp = requests.post(url, headers=head, json=data)    
    json_data = json.loads(resp.text)
    df2_ = pd.read_json(resp.text)

    df2_ = pd.concat([df2_.drop(['alocacaoFinanceiraEstrategia'], axis=1), df2_['alocacaoFinanceiraEstrategia'].apply(pd.Series)], axis=1)
    df2_ = df2_.loc[:, ('descricao', 'saldo')]

    df2_['saldo'] = df2_['saldo'].apply(lambda x : x if x > 0 else 0)
    df2_= df2_[df2_['saldo'] != 0]

    df2_['Allocation'] =  df2_.loc[:, 'saldo'] / df2_.loc[:, 'saldo'].sum()

    df2_['Allocation'] = df2_['Allocation'].apply(lambda x : x if x > 0.005 else 0)
    df2_= df2_[df2_['Allocation'] != 0]

    df2_['Name'] = df2_['descricao'].replace(['Caixa', 'VALORES A  LIQUIDAR', 'Inflação', 'Juros Nominal', 'Ações Brasil', 
    'Alternativos', 'Renda Variável Internacional', 'Moedas', 'Multimercado', 'Commodity', 'Real Estate'], tickers_asset_class)
    
    Total_Assets = round(df2_['saldo'].sum(),2)
    #['Pre_Caixa', 'Pre_Caixa', 
    #'IMAB11.SA', 'FIXA11.SA', 'BOVA11.SA', 'BOVA11.SA', 'IVVB11.SA', 'Moedas', 'BOVA11.SA', 'GOLD11.SA', 'XFIX11.SA']

    df_allocate = df2_.groupby(['Name']).sum()
    df_allocate = df_allocate.drop(['saldo'], axis=1)
    df_allocate.reset_index(inplace = True)

    return df_allocate, Total_Assets

