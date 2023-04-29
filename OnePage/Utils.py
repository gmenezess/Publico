# Importando bibliotecas
import sys
#print("Current version of Python is ", sys.version)
import requests
from requests.auth import HTTPBasicAuth
import json
import pandas as pd
import numpy as np
from datetime import datetime as date
import datetime
import quantstats as qs
#import sqlalchemy
#from sqlalchemy import create_engine
#from sqlalchemy import update
import timeit
import time
import string
#import pygsheets
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns; sns.set_theme()


from warnings import warn
import pandas as pd
import numpy as np
from math import ceil as _ceil, sqrt as _sqrt
from scipy.stats import (
    norm as _norm, linregress as _linregress
)

from quantstats import utils as _utils
from quantstats import plots as _plots

from quantstats._plotting import core as _core
from quantstats._plotting import wrappers as _wrappers
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import matplotlib.dates as _mdates


from datetime import date
import datetime
from pandas.tseries.offsets import BDay
pd.set_option("display.max_colwidth", 150)

#Gráficos
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import yfinance as yf
#from IPython.display import display

import calendar


#from datetime import datetime
from datetime import timezone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


from Utils import *
import pandas as pd



def consulta_bc(cod): 
    url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados?formato=json'
    df = pd.read_json(url)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df.set_index('data', inplace=True)
    return df

def busca_cadastro_cvm(data=(date.today()-BDay(5))):
  if data is not busca_cadastro_cvm.__defaults__[0]:
    data = pd.to_datetime(data)
  
  try:
    #url = 'http://dados.cvm.gov.br/dados/FI/CAD/DADOS/inf_cadastral_fi_{}{:02d}{:02d}.csv'.format(data.year, data.month, data.day)
    url = 'http://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv'
    return pd.read_csv(url, sep=';', encoding='ISO-8859-1')

  except: 
    print("Arquivo {} não encontrado!".format(url))
    print("Forneça outra data!")


def busca_informes_diarios_cvm_por_periodo(data_inicio, data_fim):
  datas = pd.date_range(data_inicio, data_fim, freq='MS') 
  informe_completo = pd.DataFrame()

  for data in datas:
    try:
      url ='http://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{}{:02d}.zip'.format(data.year, data.month)
      informe_mensal = pd.read_csv(url, sep=';')    
    
    except: 
      print("Arquivo {} não encontrado!".format(url))    

    informe_completo = pd.concat([informe_completo, informe_mensal], ignore_index=True)

  return informe_completo


def cagr(returns, rf=0, nperiods=252):            
    
    years = (returns.index[-1] - returns.index[0]).days / 365.
    
    if rf>0:
        excess_ret = qs.utils.to_excess_returns(returns, rf, nperiods)
        ret = qs.stats.comp(excess_ret)    
    else:
        ret = qs.stats.comp(returns)    

    res = abs(ret + 1.0) ** (1.0 / years) - 1

    if isinstance(ret, pd.DataFrame):
        res = pd.Series(res)
        res.index = ret.columns
    return res

def cvar(returns, sigma=1, confidence=0.95, prepare_returns=True):
    if prepare_returns:
        returns = qs.utils._prepare_returns(returns)    
    var = qs.stats.var(returns, sigma, confidence=0.95, prepare_returns=False)
    c_var = returns[returns < var].values.mean()
    res = c_var if ~np.isnan(c_var) else var
    
    if isinstance(returns, pd.DataFrame):
        res = pd.Series(res)
        res.index = returns.columns
    return res

def max_drawdown(prices): 
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def ranking_fundos(informes, cadastro, minimo_de_cotistas=100, classe=''):  

  cadastro = cadastro[cadastro['SIT'] == 'EM FUNCIONAMENTO NORMAL']
  fundos = informes[informes['NR_COTST'] >= minimo_de_cotistas]
  cnpj_informes = fundos['CNPJ_FUNDO'].drop_duplicates()

  fundos = fundos.pivot(index='DT_COMPTC', columns='CNPJ_FUNDO')
  cotas_normalizadas = fundos['VL_QUOTA'] / fundos['VL_QUOTA'].iloc[0]


  cnpj_cadastro = cadastro[cadastro['CLASSE'] == 'Fundo Multimercado']['CNPJ_FUNDO']
  cotas_normalizadas = cotas_normalizadas[cnpj_cadastro[cnpj_cadastro.isin(cnpj_informes)]]
  cotas_norma_clean = cotas_normalizadas.dropna(axis=1)

  ret = cotas_norma_clean.pct_change()
  
  ret.index = pd.to_datetime(ret.index)

  cagr_ = cagr(ret, rf=0, nperiods=0)

  vol = qs.stats.volatility(ret, prepare_returns=False)

  excess_ret = cagr(ret, rf=0.1150, nperiods=252)

  es = cvar(ret, confidence=0.95, prepare_returns=False)


  fundos_rk = pd.DataFrame()
  fundos_rk['Ret_Acum_Per'] = ((cotas_norma_clean.iloc[-1]) - 1) * 100

  fundos_rk['CAGR'] = cagr_.values*100
  fundos_rk['Vol'] = vol.values*100
  fundos_rk['Excess_Return'] = excess_ret.values*100
  fundos_rk['Sharpe'] = (excess_ret.values/vol.values)
  fundos_rk['ES_95'] = es.values*100
    
  fundos_rk.sort_values(by='Ret_Acum_Per', ascending=False, inplace=True)

  for cnpj in fundos_rk.index:
      fundo = cadastro[cadastro['CNPJ_FUNDO'] == cnpj]
      fundos_rk.at[cnpj, 'Fundo de Investimento'] = fundo['DENOM_SOCIAL'].values[0]
      fundos_rk.at[cnpj, 'Classe']                = fundo['CLASSE'].values[0]
      fundos_rk.at[cnpj, 'PL']                    = fundo['VL_PATRIM_LIQ'].values[0] 

  return fundos_rk


def consulta_fundo(informes, cnpj):  
  fundo = informes[informes['CNPJ_FUNDO'] == cnpj].copy()
  fundo.set_index('DT_COMPTC', inplace=True)
  fundo['cotas_normalizadas'] = (fundo['VL_QUOTA'] / fundo['VL_QUOTA'].iloc[0])*100
  return fundo


def cdi_acumulado(data_inicio, data_fim):
  codigo_bcb = 12
  
  url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
  cdi = pd.read_json(url)
  cdi['data'] = pd.to_datetime(cdi['data'], dayfirst=True)
  cdi.set_index('data', inplace=True) 
  
  cdi_acumulado = (1 + cdi[data_inicio : data_fim] / 100).cumprod()
  cdi_acumulado.iloc[0] = 1
  return cdi_acumulado


def set_date(year, month_start, month_end):

    def last_business_day_in_month(year: int, month: int) -> int:
        return max(calendar.monthcalendar(year, month)[-1:][0][:5])


    #year=2023
    #month_start=3
    #month_end=3

    if month_start == 1:
        data_inicio = date(year-1, 12, last_business_day_in_month(year-1, 12))
    else:    
        data_inicio = date(year, month_start-1, last_business_day_in_month(year, month_start-1))

    if month_end is not None:
        data_fim = date(year, month_end, last_business_day_in_month(year, month_end))
    else:
        data_fim = date.today() - datetime.timedelta(days= 2)

    #print(data_inicio)
    #print(data_fim)

    return data_inicio, data_fim

def df_fundos(informes, cadastro, minimo_de_cotistas=10, classe='Fundo Multimercado'):
 
    cadastro = cadastro[cadastro['SIT'] == 'EM FUNCIONAMENTO NORMAL']
    fundos = informes[informes['NR_COTST'] >= minimo_de_cotistas]    
      

    return fundos


def df_metrics(informes, cadastro, data_inicio, data_fim, classe='Fundo Multimercado'):    

    fundos = df_fundos(informes, cadastro, minimo_de_cotistas=10, classe='Fundo Multimercado')
    
    cnpj_informes = fundos['CNPJ_FUNDO'].drop_duplicates()
    fundos = fundos.pivot(index='DT_COMPTC', columns='CNPJ_FUNDO')
    fundos.index = pd.to_datetime(fundos.index)
        
    cnpj_cadastro = cadastro[cadastro['CLASSE']== classe]['CNPJ_FUNDO']    
    fundos_per = fundos[(fundos.index >= pd.to_datetime(data_inicio)) & (fundos.index <= pd.to_datetime(data_fim))]
    cotas_normalizadas = fundos_per['VL_QUOTA'] / fundos_per['VL_QUOTA'].iloc[0]
    cotas_normalizadas = cotas_normalizadas[cnpj_cadastro[cnpj_cadastro.isin(cnpj_informes)]]
    cotas_norma_clean = cotas_normalizadas.dropna(axis=1)
    

    ret = cotas_norma_clean.pct_change()
    ret.index = pd.to_datetime(ret.index)
    

    cagr_ = cagr(ret, rf=0, nperiods=0)
    vol = qs.stats.volatility(ret, prepare_returns=False)
    excess_ret = cagr(ret, rf=0.1150, nperiods=252)
    es = cvar(ret, confidence=0.95, prepare_returns=False)
    max_dd = max_drawdown(cotas_norma_clean)

    fundos_rk = pd.DataFrame()
    fundos_rk['Ret_Acum_Per'] = ((cotas_norma_clean.iloc[-1]) - 1) * 100

    fundos_rk['CAGR'] = cagr_.values*100
    fundos_rk['Vol'] = vol.values*100
    fundos_rk['Excess_Return'] = excess_ret.values*100
    fundos_rk['Sharpe'] = (excess_ret.values/vol.values)
    fundos_rk['ES_95'] = es.values*100
    fundos_rk['Max_DD'] = max_dd.values*100

    fundos_rk.sort_values(by='Ret_Acum_Per', ascending=False, inplace=True)

    for cnpj in fundos_rk.index:
        fundo = cadastro[cadastro['CNPJ_FUNDO'] == cnpj]
        fundos_rk.at[cnpj,
                    'Fundo de Investimento'] = fundo['DENOM_SOCIAL'].values[0]
        fundos_rk.at[cnpj, 'Classe'] = fundo['CLASSE'].values[0]
        fundos_rk.at[cnpj, 'PL'] = fundo['VL_PATRIM_LIQ'].values[0]
    
    return fundos_rk


def get_ret_cdi_ibov(data_inicio, data_fim):
    codigo_bcb = 12
    url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)

    cdi = pd.read_json(url)
    cdi['data'] = pd.to_datetime(cdi['data'], dayfirst=True)
    cdi.set_index('data', inplace=True)
    cdi_ts = cdi[data_inicio : data_fim]/100
    cdi_acum = (1 + cdi_ts).cumprod()
    cdi_ret_total = (cdi_acum.iloc[-1][0]-1)*100


    ibov = yf.download('^BVSP', start=data_inicio, end=data_fim)['Adj Close']
    ibov = (ibov / ibov.iloc[0])*100
    ibov_ret = ibov.pct_change()
    ret_ano_ibov = cagr(ibov_ret, rf=0, nperiods=0) * 100
    vol_ano_ibov = qs.stats.volatility(ibov_ret, prepare_returns=False) * 100
    ibov_ret_total = (ibov[-1]/100 -1)*100

    return cdi_ret_total, ibov_ret_total, vol_ano_ibov, ret_ano_ibov