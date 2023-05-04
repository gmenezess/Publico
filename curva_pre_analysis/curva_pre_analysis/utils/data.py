import numpy as np
import pandas as pd

import pytz
import yfinance as yf
#import matplotlib.pyplot as plt

from bcb import currency
from bcb import sgs

#from pyettj import ettj
#import pyettj.ettj as ettj
#import datetime
from datetime import datetime as date
from utils.ettj_curves import *
import calendar

def get_stocks(names, start_date, end_date): 
    
    def last_business_day_in_month(year: int, month: int) -> int:
        return max(calendar.monthcalendar(year, month)[-1:][0][:5])
    
    if start_date.month == 1:
        start_month = 12
        start_year = start_date.year - 1
    else:
        start_month = start_date.month - 1
        start_year = start_date.year

    data_inicio = date(start_year, start_month, last_business_day_in_month(start_year, start_month-1))    
    data_fim = date(end_date.year, end_date.month, end_date.day)

    #path = "/Users/Dell/OneDrive//Área de Trabalho/Comite Invest Axiom/anbima_index.xlsx"
    path = "/Users/Dell/OneDrive//Área de Trabalho/User02/AXIOM/Axiom_Feed/anbima_index.xlsx"
    anbima_index = pd.read_excel(path)

    anb_index = anbima_index.iloc[:,2:12].drop(['Date.2', 'Date.3', 'Date.4', 'Date.5'], axis=1)
    anb_index.rename(columns = {'Date.1':'Date'}, inplace = True)

    ihfa_index = anbima_index.iloc[:,0:2]
    df_index = pd.merge(ihfa_index, anb_index, on='Date')
    df_index.set_index('Date', inplace=True)
    df_index.index = pd.to_datetime(df_index.index)
    df_index = df_index.iloc[::-1]
    df_index = df_index[(df_index.index >= pd.to_datetime(data_inicio)) & (df_index.index <= pd.to_datetime(data_fim))]

    list_index = list(df_index.columns.values)
    
    if not isinstance(start_date, str):
            start_dt = start_date.strftime('%Y-%m-%d')
    if not isinstance(end_date, str):
            end_dt = end_date.strftime('%Y-%m-%d')

    list_full = list(set(names))
    #names_stocks = [x for x in list_full if x != 'Pre_Caixa' and x != 'Moedas']
    names_stocks = [x for x in list_full if x !=
        'Pre_Caixa' and x != 'Moedas' and x not in list_index]

    df_stocks = yf.download(names_stocks, start=start_dt, end=end_dt)
    if not isinstance(df_stocks.columns, pd.MultiIndex) and len(names_stocks) > 0:

        stock_tuples = [(col, names_stocks[0])
                            for col in list(df_stocks.columns)]
        df_stocks.columns = pd.MultiIndex.from_tuples(stock_tuples)
    
    df_stock_price = df_stocks[["Adj Close"]].droplevel(0, axis=1)
    #df_stock_price.index = pd.to_datetime(df_stock_price.index)

    
    return df_stock_price



def get_fx(start_date, end_date): 

    df_crncy_ret = pd.DataFrame()
    df_crncy = currency.get(['USD', 'EUR', 'GBP', 'CAD', 'SEK', 'CHF'], start=start_date, end=end_date)
    weights_DXY = np.array([0.50, 0.30, 0.05, 0.05, 0.05, 0.05])
    df_crncy_ret['Moedas'] = np.log(df_crncy).diff(periods = 1)@weights_DXY.T
    df_crncy_price = 100 * (1 + df_crncy_ret).cumprod()

    return df_crncy_price

def get_indexes(start_date, end_date): 
    
    def last_business_day_in_month(year: int, month: int) -> int:
        return max(calendar.monthcalendar(year, month)[-1:][0][:5])
    
    if start_date.month == 1:
        start_month = 12
        start_year = start_date.year - 1
    else:
        start_month = start_date.month - 1
        start_year = start_date.year

    data_inicio = date(start_year, start_month, last_business_day_in_month(start_year, start_month-1))    
    data_fim = date(end_date.year, end_date.month, end_date.day)

    #path = "/Users/Dell/OneDrive//Área de Trabalho/Comite Invest Axiom/anbima_index.xlsx"
    path = "/Users/Dell/OneDrive//Área de Trabalho/User02/AXIOM/Axiom_Feed/anbima_index.xlsx"
    anbima_index = pd.read_excel(path)

    anb_index = anbima_index.iloc[:,2:12].drop(['Date.2', 'Date.3', 'Date.4', 'Date.5'], axis=1)
    anb_index.rename(columns = {'Date.1':'Date'}, inplace = True)

    ihfa_index = anbima_index.iloc[:,0:2]
    df_index = pd.merge(ihfa_index, anb_index, on='Date')
    df_index.set_index('Date', inplace=True)
    df_index.index = pd.to_datetime(df_index.index)
    df_index = df_index.iloc[::-1]
    df_index = df_index[(df_index.index >= pd.to_datetime(data_inicio)) & (df_index.index <= pd.to_datetime(data_fim))]

    return df_index


    #last_date = pd.Timestamp(df_data.index.max(), tz=pytz.UTC) + datetime.timedelta(days= 1)
    #taxas = update_hist_ettj(last_date)
    #risk_free_rate = taxas['Pre_Caixa'][-1:][0]

    #get_ipca_monthly
    #start_dt_str = start_dt.strftime('%Y-%m-%d')
    #df_ipca_m = sgs.get({'IPCA': 433}, start=start_dt_str)
    #df_ipca_m.index = df_ipca_m.index.strftime('%Y-%m')


def get_prices(names, start_date, end_date):    

    #############################
    
    def last_business_day_in_month(year: int, month: int) -> int:
        return max(calendar.monthcalendar(year, month)[-1:][0][:5])
    
    if start_date.month == 1:
        start_month = 12
        start_year = start_date.year - 1
    else:
        start_month = start_date.month - 1
        start_year = start_date.year

    data_inicio = date(start_year, start_month, last_business_day_in_month(start_year, start_month))    
    data_fim = date(end_date.year, end_date.month, end_date.day)

    #path = "/Users/Dell/OneDrive//Área de Trabalho/Comite Invest Axiom/anbima_index.xlsx"
    path = "/Users/Dell/OneDrive//Área de Trabalho/User02/AXIOM/Axiom_Feed/anbima_index.xlsx"
    anbima_index = pd.read_excel(path)

    anb_index = anbima_index.iloc[:,2:12].drop(['Date.2', 'Date.3', 'Date.4', 'Date.5'], axis=1)
    anb_index.rename(columns = {'Date.1':'Date'}, inplace = True)

    ihfa_index = anbima_index.iloc[:,0:2]
    df_index = pd.merge(ihfa_index, anb_index, on='Date')
    df_index.set_index('Date', inplace=True)
    df_index.index = pd.to_datetime(df_index.index)
    df_index = df_index.iloc[::-1]
    df_index = df_index[(df_index.index >= pd.to_datetime(data_inicio)) & (df_index.index <= pd.to_datetime(data_fim))]
    df_index.index = df_index.index.strftime('%Y-%m-%d')

    list_index = list(df_index.columns.values)

    #############################

    if not isinstance(start_date, str):
            start_dt = start_date.strftime('%Y-%m-%d')
    if not isinstance(end_date, str):
            end_dt = end_date.strftime('%Y-%m-%d')
                    
    list_full = list(set(names))
    #names_stocks = [x for x in list_full if x != 'Pre_Caixa' and x != 'Moedas']
    names_stocks = [x for x in list_full if x !=
        'Pre_Caixa' and x != 'Moedas' and x not in list_index]

    names_index = [x for x in list_full if x in list_index]

    if names_stocks:

        df_stocks = yf.download(names_stocks, start=start_dt, end=end_dt)
        if not isinstance(df_stocks.columns, pd.MultiIndex) and len(names_stocks) > 0:            
            stock_tuples = [(col, names_stocks[0])
                            for col in list(df_stocks.columns)]
            df_stocks.columns = pd.MultiIndex.from_tuples(stock_tuples)
        df_stock_price = df_stocks[["Adj Close"]].droplevel(0, axis=1)
        #df_stock_price.index = pd.to_datetime(df_stock_price.index)
        df_stock_price.index = df_stock_price.index.strftime('%Y-%m-%d')


        if names_index:
            df_index = df_index[names_index]            
            df_price = pd.merge(df_stock_price, df_index, on='Date')
        else:
            df_price = df_stock_price

    elif not names_stocks:

        if names_index:
            df_index = df_index[names_index]
            df_price = df_index
    else:        
        df_price = None

    #############################

    taxas = update_hist_ettj(end_dt)
    df_pre_caixa = ((1 + taxas[['Pre_Caixa']]/100) ** (1/252) - 1)
    df_pre_caixa.index = df_pre_caixa.index.strftime('%Y-%m-%d')

    
    if 'Pre_Caixa' in list_full:
        if df_price is not None:
            
            df_price = pd.merge(df_price, df_pre_caixa, on='Date')
        else:        
            df_pre_caixa.index = pd.to_datetime(df_pre_caixa.index)
            df_price = df_pre_caixa[(df_pre_caixa.index >= np.datetime64(start_dt)) & (df_pre_caixa.index <= np.datetime64(end_dt))]


    if 'Moedas' in list_full:
    
        df_crncy_ret = pd.DataFrame()
        df_crncy = currency.get(['USD', 'EUR', 'GBP', 'CAD', 'SEK', 'CHF'], start=start_dt, end=end_dt)
        weights_DXY = np.array([0.50, 0.30, 0.05, 0.05, 0.05, 0.05])
        df_crncy_ret['Moedas'] = np.log(df_crncy).diff(periods = 1)@weights_DXY.T
        df_crncy_price = 100 * (1 + df_crncy_ret).cumprod()
        df_crncy_price.index = df_crncy_price.index.strftime('%Y-%m-%d')


        #df_price = pd.merge(pd.merge(df_crncy_price,df_price_0,on='Date'), df_pre_caixa, on='Date')
        if df_price is not None:
            #df_price = pd.merge(pd.merge(df_price_0, df_crncy_price, on='Date'), df_pre_caixa, on='Date')
            df_price = pd.merge(df_price, df_crncy_price, on='Date')
        else:
            df_price = df_crncy_price

    #else:

    #    df_price = pd.merge(df_price_0, df_pre_caixa, on='Date')

    df_price = df_price.reindex(columns=sorted(df_price.columns))
    
    df_price = df_price.groupby(['Date']).agg('first').reset_index()
    df_price.set_index('Date', inplace=True)
    df_price.index = pd.to_datetime(df_price.index)

    start_dt_str = start_date.strftime('%Y-%m-%d')
    df_ipca_m = sgs.get({'IPCA': 433}, start=start_dt_str)
    df_ipca_m.index = df_ipca_m.index.strftime('%Y-%m')

    return df_price, df_ipca_m
