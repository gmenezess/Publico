import numpy as np
import pandas as pd

from IPython.display import display

import pytz
import yfinance as yf
yf.pdr_override()
import pandas_datareader.data as web
from pandas_datareader import data as web

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import calendar

import itertools
import seaborn as sns
#import seaborn as sns; sns.set_theme()

import sys
sys.path.append('..')

from bcb import currency
from bcb import sgs
from pyettj import ettj
import pyettj.ettj as ettj

from operator import concat
#plt.rcParams['figure.figsize'] = (14,12)

import datetime
from datetime import datetime as date
from functools import reduce

##############################
#print("Current version of Python is ", sys.version)
import requests
from requests.auth import HTTPBasicAuth
import json

from functools import partial
import quantstats as qs
import numpy_financial as npf
import scipy.stats as ss

from fin_quant.port_building import build_portfolio, Portfolio, Stock, build_portfolio
from utils.data import get_prices

from quantstats import utils as _utils
from quantstats import plots as _plots

from quantstats._plotting import core as _core
from quantstats._plotting import wrappers as _wrappers
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import matplotlib.dates as _mdates



user = 'J07wqQSo7M'
password = 'VWNeDUyKUseOp0S4VAPHl02Yxyw1ZpGD'
res = requests.get('https://api.smartbraincloud.net:8099/api/v1/auth/login', verify=False, auth=HTTPBasicAuth(user, password))

res_string = res.text
token = (res_string.split(",")[0])[16:-1]
suffix = 'Bearer '
access_token = suffix + token

url = "https://api.smartbraincloud.net:8099/api/v1/Consulta/Cliente"
head = {"Authorization": access_token}
data= {'codigoExterno': "" }
resp = requests.post(url, headers=head, json=data)

    # Obtém a Lista de Clientes disponíveis para obter dados do Extrato

url = "https://api.smartbraincloud.net:8099/api/v1/Consulta/Cliente"
head = {"Authorization": access_token}
data= {'nomeCliente': "" }
resp = requests.post(url, headers=head, json=data)

    # Carregando a os dados no json
    # Criando DataFrame com os dados do cliente
    #json_data = json.loads(resp.text) 
clientes = pd.read_json(resp.text)

    # Tratando os dados no DataFrame
clientes = pd.concat([clientes.drop(['cliente'], axis=1), clientes['cliente'].apply(pd.Series)], axis=1)
clientes = clientes[['nomeCliente', 'codigoCliente']]


    # Criando lista de códigos dos clientes para o FOR LOOP
cod_users = clientes['codigoCliente'].tolist()

nomes = clientes.rename(columns={'nomeCliente':'Nome', 'codigoCliente': 'KPI_ID'}).set_index('KPI_ID')

#aloc_w_no_fund = ['Caixa', 'Inflacao', 'Nominal',
#    'RV_Br','RV_Int', 'Cmdty', 'MM', 'RE', 'Alt']

aloc_w_no_fund = ['Caixa', 'RF_Pós', 'Inflacao', 'Nominal',
 'RV_Br','RV_Int', 'Cmdty', 'Moedas', 'MM', 'RE', 'Alt']


def df_clone_func():
    token_bubble = 'bearer 5a4093eaaacf391c6fcf3e6ab1e6b95b'

    url_contas ='https://axiom-flow.bubbleapps.io/api/1.1/obj/cliente'
    headers = {
        'authorization': token_bubble
    }

    resp = requests.get(url_contas, headers = headers)
    json_data_contas = json.loads(resp.text)

    #criando a lista para paginação do dataframe
    remaining = json_data_contas['response']['remaining']

    fator = remaining / 100
    inteiro = remaining // 100

    if fator > inteiro:
        n_pag = inteiro + 1
        
    else:
        n_pag = inteiro
        
    n_pag

    list_cursor = [0]
    cursor = 0

    for i in range(n_pag):
        cursor += 100
        list_cursor.append(cursor)
        
    list_cursor

    #loop para apendar dataframes

    df_contas_final = pd.DataFrame()

    for cursor in list_cursor:
        url_contas = f'https://axiom-flow.bubbleapps.io/api/1.1/obj/cliente?cursor={cursor}&limit=100'
        headers = {
            'authorization': token_bubble
        }
        resp = requests.get(url_contas, headers = headers)
        json_data_contas = json.loads(resp.text)
        #json_data_contas
        contas = json_data_contas['response']['results']
        df_contas_bubble = pd.DataFrame(contas)
        df_contas_bubble.rename(columns = {'sinacor':'SinacorId'}, inplace = True)
        df_contas_final = pd.concat([df_contas_final, df_contas_bubble])  
    df_clone = df_contas_final.loc[:,['Nome','_id', 'taxa_aum', 'kpi_id', 'Empresa']]

    lista1=[]
    lista2=[]
    lista3=[]
    lista4=[]
    lista5=[]
    for i in range(len(df_clone)):
        if df_clone.iloc[i,4] == '1669646148558x918434927109734400':
            lista1.append(df_clone.iloc[i,0])
            lista2.append(df_clone.iloc[i,1])
            lista3.append(df_clone.iloc[i,2])
            lista4.append(df_clone.iloc[i,3])
            lista5.append(df_clone.iloc[i,4])
    df_clone = pd.DataFrame()
    df_clone['Nome'] = lista1
    df_clone['_id'] = lista2
    df_clone['taxa_aum'] = lista3
    df_clone['kpi_id'] = lista4
    df_clone['Empresa']=lista5
    df_clone.to_excel("clientes.xlsx", index = False) 
    df_clone=df_clone.dropna()
    df_clone['kpi_id'] = df_clone['kpi_id'].astype(int)
    #print(df_clone)

    return df_clone

def df_conta():
    token_bubble = 'bearer 5a4093eaaacf391c6fcf3e6ab1e6b95b'

    url_contas ='https://axiom-flow.bubbleapps.io/api/1.1/obj/conta'
    headers = {
        'authorization': token_bubble
    }

    resp = requests.get(url_contas, headers = headers)
    json_data_contas = json.loads(resp.text)

    #criando a lista para paginação do dataframe
    remaining = json_data_contas['response']['remaining']

    fator = remaining / 100
    inteiro = remaining // 100

    if fator > inteiro:
        n_pag = inteiro + 1
        
    else:
        n_pag = inteiro
        
    n_pag

    list_cursor = [0]
    cursor = 0

    for i in range(n_pag):
        cursor += 100
        list_cursor.append(cursor)
        
    list_cursor

    #loop para apendar dataframes

    df_contas_final = pd.DataFrame()

    for cursor in list_cursor:
        url_contas = f'https://axiom-flow.bubbleapps.io/api/1.1/obj/conta?cursor={cursor}&limit=100'
        headers = {
            'authorization': token_bubble
        }
        resp = requests.get(url_contas, headers = headers)
        json_data_contas = json.loads(resp.text)
        #json_data_contas
        contas = json_data_contas['response']['results']
        df_contas_bubble = pd.DataFrame(contas)
        df_contas_bubble.rename(columns = {'sinacor':'SinacorId'}, inplace = True)
        df_contas_final = pd.concat([df_contas_final, df_contas_bubble])
        df_contas = df_contas_final.loc[:,['Created By','codigo do cliente', 'conta', 'instituicao', 'cliente', 'empresa']]

    lista1=[]
    lista2=[]
    lista3=[]
    lista4=[]
    lista5=[]
    for i in range(len(df_contas)):
        if df_contas.iloc[i,5] == '1669646148558x918434927109734400':
            lista1.append(df_contas.iloc[i,0])
            lista2.append(df_contas.iloc[i,1])
            lista3.append(df_contas.iloc[i,2])
            lista4.append(df_contas.iloc[i,3])
            lista5.append(df_contas.iloc[i,4])
    contas = pd.DataFrame()
    contas['Created By'] = lista1
    contas['codigo do cliente'] = lista2
    contas['conta'] = lista3
    contas['instituicao'] = lista4
    contas['cliente']=lista5
    #df_contas=df_clone.dropna()
    #print(contas)

    return contas

def df_merge():
    merge = pd.DataFrame()
    
    df_clone = df_clone_func()
    contas = df_conta()

    nome = []
    conta = []
    codigo_flow=[]
    codigo_sb=[]
    taxa =[]
    instituicao=[]
    for i in range(len(df_clone)):
        cliente_flow = df_clone.iloc[i,1]
        for j in range(len(contas)):
            if cliente_flow == contas.iloc[j,4]:
                if contas.iloc[j,3] == '1670273355423x230203459983917120':
                    nome.append(df_clone.iloc[i,0])
                    conta.append(contas.iloc[j,2])
                    codigo_flow.append(df_clone.iloc[i,1])
                    codigo_sb.append(df_clone.iloc[i,3])
                    taxa.append(df_clone.iloc[i,2])
                    instituicao.append('BTG')
                elif contas.iloc[j,3] == '1670273302245x937406286564112600':
                    nome.append(df_clone.iloc[i,0])
                    conta.append(contas.iloc[j,2])
                    codigo_flow.append(df_clone.iloc[i,1])
                    codigo_sb.append(df_clone.iloc[i,3])
                    taxa.append(df_clone.iloc[i,2])
                    instituicao.append('Warren')

    merge['Nome'] = nome
    merge['kpi_id'] = codigo_sb
    merge['Codigo Flow'] = codigo_flow
    merge['taxa'] = taxa
    merge['Conta'] = conta
    merge['instituicao'] = instituicao
    #print(merge)

    #merge = merge.drop(['Codigo Flow', 'Conta'], axis=1)
    merge = merge.drop('Nome', axis=1)
    merge = merge.rename(columns = {'kpi_id': 'KPI_ID'})

    return merge

##############################################################################################################

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

def get_cotas_hist(cod_user, data_ref):

    indexador = 1
    nome = clientes.loc[clientes['codigoCliente'] == cod_user, 'nomeCliente']
    url = "https://api.smartbraincloud.net:8099/api/v1/EvolucaoRentabilidade"   
    head = {"Authorization": access_token}
    data_ref_st = data_ref.strftime('%Y%m%d')

    data= {'codigoUsuario': cod_user, "codigoIndexador": indexador, "dataReferencia": data_ref_st}
    resp = requests.post(url, headers=head, json=data)
    json_data = json.loads(resp.text)
    df = pd.read_json(resp.text)    
    if len(df) > 5:
        df = pd.concat([df.drop(['evolucaoRentabilidade'], axis=1), df['evolucaoRentabilidade'].apply(pd.Series)], axis=1)        
        cota = (df['cotaCarteira'] / 100) + 1
        df['cota'] = cota                               
        cdi = (df['cotaIndexadora'] / 100) + 1
        df['cdi'] = cdi
        df['ret'] = df['cota'].pct_change()
        df['cdi'] = df['cdi'].pct_change()
        df['Date'] = pd.to_datetime(df['dataRegistro'], format= '%Y/%m/%d')
        df = df.set_index('Date')
    
    return df

def get_cotas_evol_hist(cod_user, data_ref, dt_range = True):
        
        idx_metrics=['KPI_ID', 'data_inicio', 'Nome', 'data_final', 'Cota', 
            'HPR', 'CDI_Acum', 'CDI_Pct_TT', 'Ret_Mes_Atual', 'CDI_Pct_Mes_Atual', 'Ret_Acum_YTD', 'CDI_Pct_YTD', 'CAGR', 'Vol', 'Sharpe',
            'Win_r', 'VaR_95', 'ES', 'Payoff_Ratio', 'Profit_Factor', 'Max_DD']

        idx_aloc = ['PL', 'Caixa', 'RF_Pós', 'Infinity', 'Inflacao', 'Nominal', 'RV_Br',
                        'RV_Int', 'Cmdty', 'Moedas', 'MM', 'RE', 'Alt']                
        
        df_idx_metrics = pd.DataFrame(columns=idx_metrics)    
        df_idx_aloc = pd.DataFrame(columns=idx_aloc)
        
        cols = idx_metrics + idx_aloc
        df_metrics = pd.DataFrame(columns=cols)
        
        ##################################################################################

        indexador = 1
        nome = clientes.loc[clientes['codigoCliente'] == cod_user, 'nomeCliente']
        url = "https://api.smartbraincloud.net:8099/api/v1/EvolucaoRentabilidade"
        head = {"Authorization": access_token}

        data_ref_st = data_ref.strftime('%Y%m%d')

        data = {'codigoUsuario': cod_user,
                "codigoIndexador": indexador, "dataReferencia": data_ref_st}
        resp = requests.post(url, headers=head, json=data)        

        nome = nomes.loc[cod_user, 'Nome']        
        df = pd.read_json(resp.text)
        if len(df) > 5:
                df = pd.concat([df.drop(['evolucaoRentabilidade'], axis=1),
                        df['evolucaoRentabilidade'].apply(pd.Series)], axis=1)
                cota = (df['cotaCarteira'] / 100) + 1
                df['cota'] = cota
                cdi = (df['cotaIndexadora'] / 100) + 1
                df['cdi'] = cdi
                df['ret'] = df['cota'].pct_change()
                df['cdi'] = df['cdi'].pct_change()
                df['Date'] = pd.to_datetime(df['dataRegistro'], format='%Y/%m/%d')
                df = df.set_index('Date')

                start_date = pd.to_datetime(df.index[0]).strftime("%Y-%m-%d")
                end_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d")
                dif_days = np.busday_count(start_date, end_date)
                data_inicio = df.first_valid_index()
        
                if dt_range:                                                       
                    dt_interval = pd.date_range(start=start_date, end=end_date, freq='30D')
                        
                    for dt in np.arange(1,len(dt_interval)):
                            df_expanding = df.loc[start_date:dt_interval[dt]]  
                            end_date = pd.to_datetime(df_expanding.index[-1]).strftime("%Y-%m-%d") #last date of df_expanding

                            perf_metrics = performance_metrics(df_expanding) #row output                                                        

                            df_idx_aloc.loc[0] = get_updated_aloc(cod_user, end_date)                            
                                                                
                            df_idx_metrics.loc[0] = [cod_user, data_inicio, nome] + perf_metrics #dataframe
                                
                            teste_concat = pd.concat([df_idx_metrics, df_idx_aloc], axis=1)        
                            df_metrics = df_metrics.append(teste_concat)
                                                      
                else:
                        
                    end_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d") #last date of df

                    perf_metrics = performance_metrics(df) #row output                                                        

                    df_idx_aloc.loc[0] = get_updated_aloc(cod_user, end_date)                    
                                                                
                    df_idx_metrics.loc[0] = [cod_user, data_inicio, nome] + perf_metrics #dataframe
                                
                    teste_concat = pd.concat([df_idx_metrics, df_idx_aloc], axis=1)        
                    df_metrics = df_metrics.append(teste_concat)

        return df_metrics





def metrics_v2(returns, benchmark=None, rf=0., display=True,
            mode='basic', sep=False, compounded=True,
            periods_per_year=252, prepare_returns=True,
            match_dates=False, **kwargs):

    win_year, _ = qs.reports._get_trading_periods(periods_per_year)

    benchmark_col = 'Benchmark'
    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_col = f'Benchmark ({benchmark.upper()})'
        elif isinstance(benchmark, pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError("`benchmark` must be a pandas Series, "
                             "but a multi-column DataFrame was passed")

    blank = ['']

    if isinstance(returns, pd.DataFrame):
        if len(returns.columns) > 1:
            raise ValueError(
                "`returns` needs to be a Pandas Series or one column DataFrame. multi colums DataFrame was passed")
        returns = returns[returns.columns[0]]

    if prepare_returns:
        returns = qs.utils._prepare_returns(returns)

    df = pd.DataFrame({"returns": returns})

    if benchmark is not None:
        blank = ['', '']
        benchmark = qs.utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = qs.reports._match_dates(returns, benchmark)
        df["returns"] = returns
        df["benchmark"] = benchmark

    df = df.fillna(0)

    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # return df
    dd = qs.reports._calc_dd(df, display=(display or "internal" in kwargs),
                  as_pct=kwargs.get("as_pct", False))

    metrics = pd.DataFrame()

    s_start = {'returns': df['returns'].index.strftime('%Y-%m-%d')[0]}
    s_end = {'returns': df['returns'].index.strftime('%Y-%m-%d')[-1]}
    s_rf = {'returns': rf}

    if "benchmark" in df:
        s_start['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[0]
        s_end['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[-1]
        s_rf['benchmark'] = rf

    metrics['Start Period'] = pd.Series(s_start)
    metrics['End Period'] = pd.Series(s_end)
    metrics['Risk-Free Rate %'] = pd.Series(s_rf)*100
    metrics['Time in Market %'] = qs.stats.exposure(
        df, prepare_returns=False) * pct

    metrics['~'] = blank

    if compounded:
        metrics['Cumulative Return %'] = (
            qs.stats.comp(df) * pct).map('{:,.2f}'.format)
    else:
        metrics['Total Return %'] = (df.sum() * pct).map('{:,.2f}'.format)
        
    metrics['CAGR﹪%'] = qs.stats.cagr(df, rf, compounded) * pct
    metrics['~~~~~~~~~~~~~~'] = blank

##################################################################

    metrics['Sharpe'] = qs.stats.sharpe(df, rf, win_year, True)
    metrics['Prob. Sharpe Ratio %'] = qs.stats.probabilistic_sharpe_ratio(
        df, rf, win_year, False) * pct
    
    if mode.lower() == 'full':
        metrics['Smart Sharpe'] = qs.stats.smart_sharpe(df, rf, win_year, True)
        # metrics['Prob. Smart Sharpe Ratio %'] = _stats.probabilistic_sharpe_ratio(df, rf, win_year, False, True) * pct

    metrics['Sortino'] = qs.stats.sortino(df, rf, win_year, True)
    metrics['Omega'] = qs.stats.omega(df, rf, 0., win_year)

    metrics['~~~~~~~~'] = blank
##################################################################

    
    metrics['Volatility (ann.) %'] = qs.stats.volatility(
            df['returns'], win_year, True, prepare_returns=False) * pct
    metrics['Daily Value-at-Risk %'] = -abs(qs.stats.var(
            df, prepare_returns=False) * pct)
    metrics['Expected Shortfall (cVaR) %'] = -abs(qs.stats.cvar(
            df, prepare_returns=False) * pct)
    
    #qs.stats.cvar(df.loc[:,'ret'], confidence=0.95, prepare_returns=False)*100

    metrics['Max Drawdown %'] = qs.stats.max_drawdown(df) * pct
    #metrics['Longest DD Days'] = blank

    if "benchmark" in df:
            bench_vol = qs.stats.volatility(
                df['benchmark'], win_year, True, prepare_returns=False) * pct
            
            ret_vol = qs.stats.volatility(
            df['returns'], win_year, True, prepare_returns=False) * pct
            metrics['Volatility (ann.) %'] = [ret_vol, bench_vol]
            metrics['R^2'] = qs.stats.r_squared(
                df['returns'], df['benchmark'], prepare_returns=False)
            metrics['Information Ratio'] = qs.stats.information_ratio(
                df['returns'], df['benchmark'], prepare_returns=False)

    if mode.lower() == 'full':
            
        #else:
        metrics['~~~~~~~~~~'] = blank
        metrics['Calmar'] = qs.stats.calmar(df, prepare_returns=False)
        metrics['Skew'] = qs.stats.skew(df, prepare_returns=False)
        metrics['Kurtosis'] = qs.stats.kurtosis(df, prepare_returns=False)
        
        metrics['~~~~~~~~~~'] = blank
        
        metrics['Expected Return Monthly %%'] = qs.stats.expected_return(
            df, aggregate='M', prepare_returns=False) * pct
        metrics['Expected Return Yearly %%'] = qs.stats.expected_return(
            df, aggregate='A', prepare_returns=False) * pct

        metrics['Max Consecutive Wins *int'] = qs.stats.consecutive_wins(df)
        metrics['Max Consecutive Losses *int'] = qs.stats.consecutive_losses(df)

        metrics['~~~~~~~~~~'] = blank

        metrics['Payoff Ratio'] = qs.stats.payoff_ratio(df, prepare_returns=False)
        metrics['Profit Factor'] = qs.stats.profit_factor(df, prepare_returns=False)
        metrics['Tail Ratio'] = qs.stats.tail_ratio(df, prepare_returns=False)

        metrics['~~~~~~~~~~'] = blank

        metrics['Best Day %'] = qs.stats.best(df, prepare_returns=False) * pct
        metrics['Worst Day %'] = qs.stats.worst(df, prepare_returns=False) * pct
        metrics['Best Month %'] = qs.stats.best(
            df, aggregate='M', prepare_returns=False) * pct
        metrics['Worst Month %'] = qs.stats.worst(
            df, aggregate='M', prepare_returns=False) * pct        

        metrics['Win Days %%'] = qs.stats.win_rate(
            df, prepare_returns=False) * pct
        metrics['Win Month %%'] = qs.stats.win_rate(
            df, aggregate='M', prepare_returns=False) * pct
    
    comp_func = qs.stats.comp if compounded else np.sum

    
    if "benchmark" in df:
        metrics['~~~~~~~~~~~~'] = blank
        greeks = qs.stats.greeks(
                df['returns'], df['benchmark'], win_year, prepare_returns=False)
        metrics['Beta'] = [str(round(greeks['beta'], 2)), '-']
        metrics['Alpha'] = [str(round(greeks['alpha'], 2)), '-']
        metrics['Correlation'] = [
                str(round(df['benchmark'].corr(df['returns']) * pct, 2))+'%', '-']
        metrics['Treynor Ratio'] = [str(round(qs.stats.treynor_ratio(
                df['returns'], df['benchmark'], win_year, rf)*pct, 2))+'%', '-']

    # prepare for display
    for col in metrics.columns:
        try:
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except Exception:
            pass
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace('.0', '', regex=False)
            metrics.rename({col: col.replace("*int", "")},
                           axis=1, inplace=True)
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + '%'
   
    metrics.columns = [
        col if '~' not in col else '' for col in metrics.columns]
    metrics.columns = [
        col[:-1] if '%' in col else col for col in metrics.columns]
    metrics = metrics.T

    if "benchmark" in df:
        metrics.columns = ['Strategy', benchmark_col]
    else:
        metrics.columns = ['Strategy']

    # cleanups
    metrics.replace([-0, '-0'], 0, inplace=True)
    metrics.replace([np.nan, -np.nan, np.inf, -np.inf,
                     '-nan%', 'nan%', '-nan', 'nan',
                    '-inf%', 'inf%', '-inf', 'inf'], '-', inplace=True)

    
    if display:
        print(qs.reports._tabulate(metrics, headers="keys", tablefmt='simple'))
        return None

    if not sep:
        metrics = metrics[metrics.index != '']

    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [c.replace(' %', '').replace(
        ' *int', '').strip() for c in metrics.columns]
    metrics = metrics.T

    return metrics




def plot_histogram(returns, resample="M", bins=20,
                   fontname='Arial', grayscale=False,
                   title="Retorno", kde=True, figsize=(10, 6),
                   ylabel=True, subtitle=True, compounded=True,
                   savefig=None, show=True):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['silver', 'gray', 'black']

    apply_fnc = qs.stats.comp if compounded else np.sum
    returns = returns.fillna(0).resample(resample).apply(
        apply_fnc).resample(resample).last()

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%Y'),
            returns.index.date[-1:][0].strftime('%Y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.axvline(returns.mean(), ls="--", lw=1.5,
               color=colors[2], zorder=2, label="Retorno Médio")

    sns.histplot(returns, bins=bins,
                  color=colors[0],
                  alpha=1,
                  kde=kde,
                  stat="percent", #density | percent |frequency
                  ax=ax)
    sns.kdeplot(returns, color='black', linewidth=1.5)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    ax.axhline(0.01, lw=1, color="#000000", zorder=2)
    ax.axvline(0, lw=1, color="#000000", zorder=2)

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel("Frequência_%", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.legend(fontsize=12)

    # fig.autofmt_xdate()

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

def returns_line_acum(returns, benchmark=None,
            grayscale=False, figsize=(10, 6),
            fontname='Arial', lw=1.5,
            match_volatility=False, compound=True, cumulative=True,
            resample=None, ylabel="Retorno",
            subtitle=True, savefig=None, show=True,
            prepare_returns=True):

    title = 'Retorno Acumulado' if compound else 'Retorno'
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += ' vs %s' % benchmark.upper()
        else:
            title += ' vs Benchmark'
        if match_volatility:
            title += ' (Volatility Matched)'

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    fig = plot_timeseries(returns, benchmark, title,
                                ylabel=ylabel,
                                match_volatility=match_volatility,
                                log_scale=False,
                                resample=resample,
                                compound=compound,
                                cumulative=cumulative,
                                lw=lw,
                                figsize=figsize,
                                fontname=fontname,
                                grayscale=grayscale,
                                subtitle=subtitle,
                                savefig=savefig, show=show)
    if not show:
        return fig
    

def plot_timeseries(returns, benchmark=None,
                    title="Returns", compound=False, cumulative=True,
                    fill=False, returns_label="Strategy",
                    hline=None, hlw=None, hlcolor="red", hllabel="",
                    percent=True, match_volatility=False, log_scale=False,
                    resample=None, lw=1.5, figsize=(10, 6), ylabel="",
                    grayscale=False, fontname="Arial",
                    subtitle=True, savefig=None, show=True):

    colors, ls, alpha = _core._get_colors(grayscale)

    returns.fillna(0, inplace=True)
    if isinstance(benchmark, pd.Series):
        benchmark.fillna(0, inplace=True)

    if match_volatility and benchmark is None:
        raise ValueError('match_volatility requires passing of '
                         'benchmark.')
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    if compound is True:
        if cumulative:
            returns = qs.stats.compsum(returns)
            if isinstance(benchmark, pd.Series):
                benchmark = qs.stats.compsum(benchmark)
        else:
            returns = returns.cumsum()
            if isinstance(benchmark, pd.Series):
                benchmark = benchmark.cumsum()

    if resample:
        returns = returns.resample(resample)
        returns = returns.last() if compound is True else returns.sum()
        if isinstance(benchmark, pd.Series):
            benchmark = benchmark.resample(resample)
            benchmark = benchmark.last(
            ) if compound is True else benchmark.sum()
    # ---------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                  " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    if isinstance(benchmark, pd.Series):
        ax.plot(benchmark, lw=lw, ls=ls, label="Benchmark", color=colors[0])

    alpha = .25 if grayscale else 1
    ax.plot(returns, lw=lw, label=returns_label, color=colors[1], alpha=alpha)

    if fill:
        ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=.25)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    if hline:
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="-", lw=1,
               color='gray', zorder=1)
    ax.axhline(0, ls="--", lw=1,
               color='white' if grayscale else 'black', zorder=2)

    if isinstance(benchmark, pd.Series) or hline:
        ax.legend(fontsize=12)

    plt.yscale("symlog" if log_scale else "linear")

    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_core.format_pct_axis))
        # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        #     lambda x, loc: "{:,}%".format(int(x*100))))

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel(ylabel, fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

def returns_bar_period(returns, benchmark=None,
                   fontname='Arial', grayscale=False,
                   returns_label="Carteira_Clientes_X",
                   hlw=1.5, hlcolor="red", hllabel="",
                   resample = 'A',
                   match_volatility=False,
                   log_scale=False, figsize=(10, 5), ylabel=True,
                   subtitle=True, compounded=True,
                   savefig=None, show=True,
                   prepare_returns=True):

    if resample == 'A':
        title = 'Evolução Retorno Anual'
    elif resample == 'M':
        title = 'Evolução Retorno Mensal'
    else:
        title = 'Resample period should be mentioned'

    if benchmark is not None:
        title += '  vs Benchmark'
        benchmark = _utils._prepare_benchmark(
            benchmark, returns.index).resample(resample).apply(
                qs.stats.comp).resample(resample).last()

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    if compounded:
        returns = returns.resample(resample).apply(qs.stats.comp)
    else:
        returns = returns.resample(resample).apply(pd.DataFrame().sum) ###
    returns = returns.resample(resample).last()

    fig = plot_returns_bars(returns, benchmark,                                  
                                  returns_label=returns_label,
                                  hline=returns.mean(),
                                  hlw=hlw,
                                  hlcolor=hlcolor,
                                  hllabel=hllabel,
                                  resample=resample,
                                  title=title,
                                  match_volatility=match_volatility,
                                  log_scale=log_scale,                                                                   
                                  figsize=figsize,
                                  grayscale=grayscale,
                                  fontname=fontname,
                                  ylabel=ylabel,
                                  subtitle=subtitle,
                                  savefig=savefig, show=show)
    if not show:
        return fig
    

def rolling_volatility(returns, benchmark=None,
                       period=126, period_label="6-Meses",
                       periods_per_year=252,
                       lw=1.5, fontname='Arial', grayscale=False,
                       figsize=(10, 3), ylabel="Volatilidade",
                       subtitle=True, savefig=None, show=True):

    returns = qs.stats.rolling_volatility(returns, period, periods_per_year)

    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)
        benchmark = qs.stats.rolling_volatility(
            benchmark, period, periods_per_year, prepare_returns=False)

    fig = plot_rolling_stats(returns, benchmark,
                                   hline=returns.mean(),
                                   hlw=1.5,
                                   ylabel=ylabel,
                                   title='Volatilidade Janela Móvel- DU (%s)' % period,
                                   fontname=fontname,
                                   grayscale=grayscale,
                                   lw=lw,
                                   figsize=figsize,
                                   subtitle=subtitle,
                                   savefig=savefig, show=show)
    if not show:
        return fig

def rolling_sharpe(returns, benchmark=None, rf=0.,
                   period=126, period_label= '6 Meses',
                   periods_per_year=252,
                   lw=1.25, fontname='Arial', grayscale=False,
                   figsize=(10, 3), ylabel="Sharpe",
                   subtitle=True, savefig=None, show=True):

    returns = qs.stats.rolling_sharpe(
        returns, rf, period, True, periods_per_year, )

    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = qs.stats.rolling_sharpe(
            benchmark, rf, period, True, periods_per_year,
            prepare_returns=False)

    fig = plot_rolling_stats(returns, benchmark,
                                   hline=returns.mean(),
                                   hlw=1.5,
                                   ylabel=ylabel,
                                   title='Sharpe Janela Móvel - DU (%s)' % period,
                                   fontname=fontname,
                                   grayscale=grayscale,
                                   lw=lw,
                                   figsize=figsize,
                                   subtitle=subtitle,
                                   savefig=savefig, show=show)
    if not show:
        return fig
    

def plot_rolling_stats(returns, benchmark=None, title="",
                       returns_label="Carteira_Cliente_x",
                       hline=None, hlw=None, hlcolor="red", hllabel="",
                       lw=1.5, figsize=(10, 6), ylabel="",
                       grayscale=False, fontname="Arial", subtitle=True,
                       savefig=None, show=True):

    colors, _, _ = _core._get_colors(grayscale)

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    df = pd.DataFrame(index=returns.index, data={returns_label: returns})
    if isinstance(benchmark, pd.Series):
        df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
        df = df[['Benchmark', returns_label]].dropna()
        ax.plot(df['Benchmark'], lw=lw, label="Benchmark",
                color=colors[0], alpha=.8)

    ax.plot(df[returns_label].dropna(), lw=lw,
            label=returns_label, color=colors[1])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')\
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            df.index.date[:1][0].strftime('%e %b \'%y'),
            df.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    if hline:
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    if ylabel:
        ax.set_ylabel(ylabel, fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.legend(fontsize=12)

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)
    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

def monthly_heatmap(returns, annot_size=10, figsize=(10, 5),
                    cbar=True, square=False,
                    compounded=True, eoy=False,
                    grayscale=False, fontname='Arial',
                    ylabel=True, savefig=None, show=True):

    # colors, ls, alpha = _core._get_colors(grayscale)
    cmap = 'gray' if grayscale else 'RdYlGn'

    returns = qs.stats.monthly_returns(returns, eoy=eoy,
                                     compounded=compounded) * 100

    fig_height = len(returns) / 3

    if figsize is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0]*1.04, max([fig_height, figsize[1]]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title('      Retornos Mensais (%)\n', fontsize=14, y=.995,
                 fontname=fontname, fontweight='bold', color='black')

    # _sns.set(font_scale=.9)
    ax = sns.heatmap(returns, ax=ax, annot=True, center=0,
                      annot_kws={"size": annot_size},
                      fmt="0.2f", linewidths=0.5,
                      square=square, cbar=cbar, cmap=cmap,
                      cbar_kws={'format': '%.0f%%'})
    # _sns.set(font_scale=1)

    # align plot to match other
    if ylabel:
        ax.set_ylabel('Years', fontname=fontname,
                      fontweight='bold', fontsize=12)
        ax.yaxis.set_label_coords(-.1, .5)

    ax.tick_params(colors="#808080")
    plt.xticks(rotation=0, fontsize=annot_size*1.2)
    plt.yticks(rotation=0, fontsize=annot_size*1.2)

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

def drawdowns_periods(returns, periods=5, lw=1.5, log_scale=False,
                      fontname='Arial', grayscale=False, figsize=(10, 5),
                      ylabel=True, subtitle=True, compounded=True,
                      savefig=None, show=True,
                      prepare_returns=True):
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    fig = plot_longest_drawdowns(returns,
                                       periods=periods,
                                       lw=lw,
                                       log_scale=log_scale,
                                       fontname=fontname,
                                       grayscale=grayscale,
                                       figsize=figsize,
                                       ylabel=ylabel,
                                       subtitle=subtitle,
                                       compounded=compounded,
                                       savefig=savefig, show=show)
    if not show:
        return fig

def drawdown(returns, grayscale=False, figsize=(10, 5),
             fontname='Arial', lw=1, log_scale=False,
             match_volatility=False, compound=False, ylabel="Drawdown",
             resample=None, subtitle=True, savefig=None, show=True):

    dd = qs.stats.to_drawdown_series(returns)

    fig = _core.plot_timeseries(dd, title='Underwater Plot',
                                hline=dd.mean(), hlw=2, hllabel="Média",
                                returns_label="Drawdown",
                                compound=compound, match_volatility=match_volatility,
                                log_scale=log_scale, resample=resample,
                                fill=True, lw=lw, figsize=figsize,
                                ylabel=ylabel,
                                fontname=fontname, grayscale=grayscale,
                                subtitle=subtitle,
                                savefig=savefig, show=show)
    if not show:
        return fig
    

def plot_longest_drawdowns(returns, periods=5, lw=1.5,
                           fontname='Arial', grayscale=False,
                           log_scale=False, figsize=(10, 6), ylabel=True,
                           subtitle=True, compounded=True,
                           savefig=None, show=True):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['#000000'] * 3

    dd = qs.stats.to_drawdown_series(returns.fillna(0))
    dddf = qs.stats.drawdown_details(dd)
    longest_dd = dddf.sort_values(
        by='days', ascending=False, kind='mergesort')[:periods]

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle("%.0f Piores Períodos de Drawdown\n" %
                 periods, y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")
    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')
    series = qs.stats.compsum(returns) if compounded else returns.cumsum()
    ax.plot(series, lw=lw, label="Backtest", color=colors[0])

    highlight = 'black' if grayscale else 'red'
    for _, row in longest_dd.iterrows():
        ax.axvspan(*_mdates.datestr2num([str(row['start']), str(row['end'])]),
                   color=highlight, alpha=.1)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)
    plt.yscale("symlog" if log_scale else "linear")
    if ylabel:
        ax.set_ylabel("Cumulative Returns", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(FuncFormatter(_core.format_pct_axis))
    # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
    #     lambda x, loc: "{:,}%".format(int(x*100))))

    fig.autofmt_xdate()

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

def hist_aloc(returns, lw=1.5,
                           fontname='Arial', grayscale=False,
                           figsize=(10, 6), ylabel=True,
                           subtitle=True, savefig=None, show=True):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['#000000'] * 3

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle("Evolução Histórica Alocação\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")
    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index[:1][0],
            returns.index[-1:][0]), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')  									

    #Set data
    y1 = returns.Caixa
    y2 = returns.RF_Pós
    y3 = returns.Infinity
    y4 = returns.Inflacao
    y5 = returns.Nominal
    y6 = returns.RV_Br
    y7 = returns.RV_Int
    y8 = returns.Cmdty
    y9 = returns.MM
    y10 = returns.RE
    y11 = returns.Alt

    labels = returns.columns.values
    plt.stackplot(returns.index,y1, y2, y3, y4, y5, y6, y7,y8, y9, y10, y11, labels=labels);
    plt.legend(loc='lower left')
    
    fig.autofmt_xdate()
    ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)        
    ax.yaxis.set_major_formatter(FuncFormatter(_core.format_pct_axis))    
    fig.autofmt_xdate()

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

def plot_returns_bars(returns, benchmark=None,
                      returns_label="Strategy",
                      hline=None, hlw=None, hlcolor="red", hllabel="",
                      resample="A", title="Returns", match_volatility=False,
                      log_scale=False, figsize=(10, 6),
                      grayscale=False, fontname='Arial', ylabel=True,
                      subtitle=True, savefig=None, show=True):

    if match_volatility and benchmark is None:
        raise ValueError('match_volatility requires passing of '
                         'benchmark.')
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    colors, _, _ = _core._get_colors(grayscale)
    df = pd.DataFrame(index=returns.index, data={returns_label: returns})
    if isinstance(benchmark, pd.Series):
        df['Benchmark'] = benchmark[benchmark.index.isin(returns.index)]
        df = df[['Benchmark', returns_label]]

    df = df.dropna()
    if resample is not None:
        df = df.resample(resample).apply(
            qs.stats.comp).resample(resample).last()
    # ---------------

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # use a more precise date string for the x axis locations in the toolbar
    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            df.index.date[:1][0].strftime('%Y-%m-%d'),
            df.index.date[-1:][0].strftime('%Y-%m-%d')
        ), fontsize=12, color='gray')

    if benchmark is None:
        colors = colors[1:]
    df.plot(kind='bar', ax=ax, color=colors)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    try:
        if resample == 'A':            
            ax.set_xticklabels(df.index.year)
            years = sorted(list(set(df.index.year)))
        elif resample == 'M':
            df.index = pd.to_datetime(df.index).strftime('%Y-%m')
            ax.set_xticklabels(df.index)
            years = sorted(list(set(df.index)))
        else:
            ax.set_xticklabels(df.index)
            years = sorted(list(set(df.index)))
            
    except AttributeError:
        ax.set_xticklabels(df.index)
        years = sorted(list(set(df.index)))

    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')
    # years = sorted(list(set(df.index.year)))
    if len(years) > 10:
        mod = int(len(years)/10)
        plt.xticks(np.arange(len(years)), [
            str(year) if not i % mod else '' for i, year in enumerate(years)])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    if hline:
        if grayscale:
            hlcolor = 'gray'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    if isinstance(benchmark, pd.Series) or hline:
        ax.legend(fontsize=12)

    plt.yscale("symlog" if log_scale else "linear")

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel("Retorno", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(FuncFormatter(_core.format_pct_axis))

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

################################################################

def plots_v2(returns, benchmark=None, grayscale=False,
          figsize=(6, 4), mode='full', compounded=True,
          periods_per_year=252, prepare_returns=True, match_dates=False):

    win_year, win_half_year = qs.reports._get_trading_periods(periods_per_year)

    if prepare_returns:
        returns = qs.utils._prepare_returns(returns)

    if mode.lower() != 'full':
        qs.reports._plots.snapshot(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]),
                        show=True, mode=("comp" if compounded else "sum"))

        qs.reports._plots.monthly_heatmap(returns, grayscale=grayscale,
                               figsize=(figsize[0], figsize[0]*.5),
                               show=True, ylabel=False,
                               compounded=compounded)

        return

    if benchmark is not None:
        benchmark = qs.utils._prepare_benchmark(benchmark, returns.index)
        if match_dates is True:
            returns, benchmark = qs.reports._match_dates(returns, benchmark)

#########################################################################

    if benchmark is not None:
      
        returns_line_acum(returns, benchmark=benchmark,
            grayscale=False, figsize=(figsize[0], figsize[0]*.5),
            fontname='Arial', lw=1.5,
            match_volatility=False, compound=True, cumulative=True,
            resample=None, ylabel="Retorno",
            subtitle=True, savefig=None, show=True,
            prepare_returns=True)

    else:
        returns_line_acum(returns, benchmark=None,
            grayscale=False, figsize=(figsize[0], figsize[0]*.5),
            fontname='Arial', lw=1.5,
            match_volatility=False, compound=True, cumulative=True,
            resample=None, ylabel="Retorno",
            subtitle=True, savefig=None, show=True,
            prepare_returns=True)

#########################################################################
    
  
    returns_bar_period(returns, benchmark=None,
                   fontname='Arial', grayscale=False,
                   returns_label="Carteira_Clientes_X",
                   hlw=1.5, hlcolor="red", hllabel="",
                   resample = 'M',
                   match_volatility=False,
                   log_scale=False, figsize=(figsize[0], figsize[0]*.5), ylabel=True,
                   subtitle=True, compounded=True,
                   savefig=None, show=True,
                   prepare_returns=True)
  
    plot_histogram(returns, resample="M", bins=20,
                   fontname='Arial', grayscale=False,
                   title="Distribuição Retornos Mensais", kde=True, figsize=(figsize[0], figsize[0]*.5),
                   ylabel=True, subtitle=True, compounded=True,
                   savefig=None, show=True)

    if benchmark is not None:
        qs.reports._plots.rolling_beta(returns, benchmark, grayscale=grayscale,
                            window1=win_half_year, window2=win_year,
                            figsize=(figsize[0], figsize[0]*.3),
                            show=True, ylabel=False,
                            prepare_returns=False)

################################################################

    rolling_volatility(
        returns, benchmark, grayscale=grayscale,
        figsize=(figsize[0], figsize[0]*.4), show=True, ylabel=False,
        period=20)


    rolling_sharpe(returns, grayscale=grayscale,
                          figsize=(figsize[0], figsize[0]*.4),
                          show=True, ylabel=False, period=20)

    drawdowns_periods(returns, grayscale=grayscale,
                             figsize=(figsize[0], figsize[0]*.5),
                             show=True, ylabel=False,
                             prepare_returns=False)

    drawdown(returns, grayscale=grayscale,
                    figsize=(figsize[0], figsize[0]*.4),
                    show=True, ylabel=False)

    monthly_heatmap(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.5),
                           show=True, ylabel=False)

    

################################################################


def full_v2(returns, benchmark=None, rf=0., grayscale=False,
         figsize=(8, 5), display=True, compounded=True,
         periods_per_year=252, match_dates=False):

    # prepare timeseries
    returns = qs.utils._prepare_returns(returns)
    if benchmark is not None:
        benchmark = qs.utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = qs.reports._match_dates(returns, benchmark)

    dd = qs.reports._stats.to_drawdown_series(returns)
    col = qs.reports._stats.drawdown_details(dd).columns[4]
    dd_info = qs.reports._stats.drawdown_details(dd).sort_values(by=col,
                                                      ascending=True)[:5]

    if not dd_info.empty:
        dd_info.index = range(1, min(6, len(dd_info)+1))
        dd_info.columns = map(lambda x: str(x).title(), dd_info.columns)

    print('[Performance Metrics]\n')
    metrics_v2(returns=returns, benchmark=benchmark,
                rf=rf, display=display, mode='full',
                compounded=compounded,
                periods_per_year=periods_per_year,
                prepare_returns=False)
    print('\n\n')
    print('[5 Worst Drawdowns]\n')
    if dd_info.empty:
        print("(no drawdowns)")
    else:
        print(qs.reports._tabulate(dd_info, headers="keys",
                            tablefmt='simple', floatfmt=".2f"))
    print('\n\n')
    print('[Strategy Visualization]\nvia Matplotlib')

    plots_v2(returns=returns, benchmark=benchmark,
          grayscale=grayscale, figsize=figsize, mode='full',
          periods_per_year=periods_per_year, prepare_returns=False)
    


def plots(returns, df_assets, benchmark=None, grayscale=False,
          figsize=(6, 4), mode='full', compounded=True,
          periods_per_year=252, prepare_returns=True, match_dates=False):

    win_year, win_half_year = qs.reports._get_trading_periods(periods_per_year)

    if prepare_returns:
        returns = qs.utils._prepare_returns(returns)

    if mode.lower() != 'full':
        qs.reports._plots.snapshot(returns, grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]),
                        show=True, mode=("comp" if compounded else "sum"))

        qs.reports._plots.monthly_heatmap(returns, grayscale=grayscale,
                               figsize=(figsize[0], figsize[0]*.5),
                               show=True, ylabel=False,
                               compounded=compounded)

        return

    # prepare timeseries
    if benchmark is not None:
        benchmark = qs.utils._prepare_benchmark(benchmark, returns.index)
        if match_dates is True:
            returns, benchmark = qs.reports._match_dates(returns, benchmark)

#########################################################################

    #qs.reports._plots.returns(returns, benchmark, grayscale=grayscale,
    #               figsize=(figsize[0], figsize[0]*.6),
    #               show=True, ylabel=False,
    #               prepare_returns=False)

    #returns_line_acum(returns, benchmark=None,
    #        grayscale=False, figsize=(figsize[0], figsize[0]*.5),
    #        fontname='Arial', lw=1.5,
    #        match_volatility=False, compound=True, cumulative=True,
    #        resample=None, ylabel="Retorno",
    #        subtitle=True, savefig=None, show=True,
    #        prepare_returns=True)

    #qs.reports._plots.log_returns(returns, benchmark, grayscale=grayscale,
    #                   figsize=(figsize[0], figsize[0]*.5),
    #                   show=True, ylabel=False,
    #                   prepare_returns=False)

    if benchmark is not None:
        #qs.reports._plots.returns(returns, benchmark, match_volatility=True,
        #               grayscale=grayscale,
        #               figsize=(figsize[0], figsize[0]*.5),
        #               show=True, ylabel=False,
        #               prepare_returns=False)
        
        returns_line_acum(returns, benchmark=benchmark,
            grayscale=False, figsize=(figsize[0], figsize[0]*.5),
            fontname='Arial', lw=1.5,
            match_volatility=False, compound=True, cumulative=True,
            resample=None, ylabel="Retorno",
            subtitle=True, savefig=None, show=True,
            prepare_returns=True)

    else:
        
        returns_line_acum(returns, benchmark=None,
            grayscale=False, figsize=(figsize[0], figsize[0]*.5),
            fontname='Arial', lw=1.5,
            match_volatility=False, compound=True, cumulative=True,
            resample=None, ylabel="Retorno",
            subtitle=True, savefig=None, show=True,
            prepare_returns=True)

#########################################################################
    
    hist_aloc(df_assets, lw=1.5,
                           fontname='Arial', grayscale=False,
                           figsize=(figsize[0], figsize[0]*.6), ylabel=True,
                           subtitle=True, savefig=None, show=True)

    returns_bar_period(returns, benchmark=None,
                   fontname='Arial', grayscale=False,
                   returns_label="Carteira_Clientes_X",
                   hlw=1.5, hlcolor="red", hllabel="",
                   resample = 'M',
                   match_volatility=False,
                   log_scale=False, figsize=(figsize[0], figsize[0]*.5), ylabel=True,
                   subtitle=True, compounded=True,
                   savefig=None, show=True,
                   prepare_returns=True)

    #qs.reports._plots.yearly_returns(returns, benchmark,
    #                      grayscale=grayscale,
    #                      figsize=(figsize[0], figsize[0]*.5),
    #                      show=True, ylabel=False,
    #                      prepare_returns=False)

    #plot_histogram(returns, grayscale=grayscale,
    #                 figsize=(figsize[0], figsize[0]*.5),
    #                 show=True, ylabel=False,
    #                 prepare_returns=False)
    
    plot_histogram(returns, resample="M", bins=20,
                   fontname='Arial', grayscale=False,
                   title="Distribuição Retornos Mensais", kde=True, figsize=(figsize[0], figsize[0]*.5),
                   ylabel=True, subtitle=True, compounded=True,
                   savefig=None, show=True)

    #qs.reports._plots.daily_returns(returns, grayscale=grayscale,
    #                     figsize=(figsize[0], figsize[0]*.3),
    #                     show=True, ylabel=False,
    #                     prepare_returns=False)

    if benchmark is not None:
        qs.reports._plots.rolling_beta(returns, benchmark, grayscale=grayscale,
                            window1=win_half_year, window2=win_year,
                            figsize=(figsize[0], figsize[0]*.4),
                            show=True, ylabel=False,
                            prepare_returns=False)

################################################################

    rolling_volatility(
        returns, benchmark, grayscale=grayscale,
        figsize=(figsize[0], figsize[0]*.4), show=True, ylabel=False,
        period=win_half_year)


    rolling_sharpe(returns, grayscale=grayscale,
                          figsize=(figsize[0], figsize[0]*.4),
                          show=True, ylabel=False, period=win_half_year)

    #qs.reports._plots.rolling_sortino(returns, grayscale=grayscale,
    #                       figsize=(figsize[0], figsize[0]*.3),
    #                       show=True, ylabel=False, period=win_half_year)

    drawdowns_periods(returns, grayscale=grayscale,
                             figsize=(figsize[0], figsize[0]*.5),
                             show=True, ylabel=False,
                             prepare_returns=False)

    drawdown(returns, grayscale=grayscale,
                    figsize=(figsize[0], figsize[0]*.4),
                    show=True, ylabel=False)

    monthly_heatmap(returns, grayscale=grayscale,
                           figsize=(figsize[0], figsize[0]*.5),
                           show=True, ylabel=False)

    #qs.reports._plots.distribution(returns, grayscale=grayscale,
    #                    figsize=(figsize[0], figsize[0]*.5),
    #                    show=True, ylabel=False,
    #                    prepare_returns=False)
































##############################################################################################################

def get_assets_infos(code_users_list, data_ref):

    assets_info = pd.DataFrame()

    for cod_user in code_users_list:

        data_ref_st = data_ref.strftime('%Y%m%d')
        user = 'J07wqQSo7M'
        password = 'VWNeDUyKUseOp0S4VAPHl02Yxyw1ZpGD'
        res = requests.get('https://api.smartbraincloud.net:8099/api/v1/auth/login',
                        verify=False, auth=HTTPBasicAuth(user, password))

        res_string = res.text
        token = (res_string.split(",")[0])[16:-1]
        suffix = 'Bearer '
        access_token = suffix + token
        head = {"Authorization": access_token}

        #######################

        url1 = "https://api.smartbraincloud.net:8099/api/v1/Posicao/ExtratosAtivos"
        url2 = "https://api.smartbraincloud.net:8099/api/v1/Rentabilidade/Ativos"

        indexador = 1

        data1 = {'codigoUsuario': cod_user,
                "codigoIndexador": indexador, "dataReferencia": data_ref_st}
        resp1 = requests.post(url1, headers=head, json=data1)
        df1 = pd.read_json(resp1.text)
        df1 = (pd.concat([df1.drop(['posicaoExtratosAtivos'], axis=1),
                        df1['posicaoExtratosAtivos'].apply(pd.Series)], axis=1)).dropna()
        df1 = df1[['ativos', 'seguimento', 'saldoBruto', 'participacao']]
        df1 = df1[['seguimento', 'saldoBruto', 'participacao']]
        df1 = df1.rename(columns={'seguimento': 'Classe'})  # .set_index('Classe')
        df1.reset_index(inplace=True, drop=True)

        #######################

        data2 = {'codigoUsuario': cod_user, "dataReferencia": data_ref_st}
        resp2 = requests.post(url2, headers=head, json=data2)
        df2 = pd.read_json(resp2.text)
        df2 = pd.concat([df2.drop(['rentabilidadeAtivos'], axis=1),
                        df2['rentabilidadeAtivos'].apply(pd.Series)], axis=1)
        df2 = df2.filter(['codigoCliente', 'nomeEstrategiaAtivo', 'nomeAtivo', 'rent_Mes', 'rent_12M'])
        df2 = df2.rename(
            columns={'codigoCliente': 'KPI_ID', 'nomeEstrategiaAtivo': 'Classe', 'nomeAtivo': 'ativos'})

        #######################

        df3 = pd.concat([df2, df1[['saldoBruto', 'participacao']]], axis=1)
        df3['Contribuicao_Ret'] = (df3.participacao * df3.rent_Mes) / 100
        
        assets_info = pd.concat([assets_info, df3], axis=0)

    return assets_info

def get_assets_pnl(cod_user, data_ref):

    data_ref_st = data_ref.strftime('%Y%m%d')
    user = 'J07wqQSo7M'
    password = 'VWNeDUyKUseOp0S4VAPHl02Yxyw1ZpGD'
    res = requests.get('https://api.smartbraincloud.net:8099/api/v1/auth/login',
                    verify=False, auth=HTTPBasicAuth(user, password))

    res_string = res.text
    token = (res_string.split(",")[0])[16:-1]
    suffix = 'Bearer '
    access_token = suffix + token
    head = {"Authorization": access_token}

    #######################

    url1 = "https://api.smartbraincloud.net:8099/api/v1/Posicao/ExtratosAtivos"
    url2 = "https://api.smartbraincloud.net:8099/api/v1/Rentabilidade/Ativos"

    indexador = 1

    data1 = {'codigoUsuario': cod_user,
            "codigoIndexador": indexador, "dataReferencia": data_ref_st}
    resp1 = requests.post(url1, headers=head, json=data1)
    df1 = pd.read_json(resp1.text)
    df1 = (pd.concat([df1.drop(['posicaoExtratosAtivos'], axis=1),
                    df1['posicaoExtratosAtivos'].apply(pd.Series)], axis=1)).dropna()
    # df1 = df1[['ativos', 'seguimento', 'saldoBruto', 'participacao']]
    df1 = df1[['seguimento', 'saldoBruto', 'participacao']]
    df1 = df1.rename(columns={'seguimento': 'Classe'})  # .set_index('Classe')
    df1.reset_index(inplace=True, drop=True)

    #######################

    data2 = {'codigoUsuario': cod_user, "dataReferencia": data_ref_st}
    resp2 = requests.post(url2, headers=head, json=data2)
    df2 = pd.read_json(resp2.text)
    df2 = pd.concat([df2.drop(['rentabilidadeAtivos'], axis=1),
                    df2['rentabilidadeAtivos'].apply(pd.Series)], axis=1)
    df2 = df2.filter(
        ['nomeEstrategiaAtivo', 'nomeAtivo', 'rent_Mes', 'rent_12M'])
    # .set_index('Classe')
    df2 = df2.rename(
        columns={'nomeEstrategiaAtivo': 'Classe', 'nomeAtivo': 'ativos'})

    #######################

    df3 = pd.concat([df2, df1[['saldoBruto', 'participacao']]], axis=1)

    df3['Contribuicao_Ret'] = (df3.participacao * df3.rent_Mes) / 100

    # asset_cont = df5['Contribuicao_Ret'].sort_values(ascending=False)
    # asset_cont.plot(kind= 'bar', figsize= (20, 8), title= 'Atribuição de Resultado')

    df4 = pd.DataFrame(df3.groupby(by=['Classe'])['Contribuicao_Ret'].sum())
    df4.reset_index(inplace=True)

    aloc_aux = ['Caixa', 'Caixa', 'RF_Pós',
                'Inflacao', 'Nominal', 'RV_Br', 'RV_Int', 'Moedas', 'MM', 'Cmdty', 'RE', 'Alt']
    
    return df3

def get_asset_pnl_atrib_aux(cod_user, data_ref):        

    data_ref_st = data_ref.strftime('%Y%m%d')
    
    user = 'J07wqQSo7M'
    password = 'VWNeDUyKUseOp0S4VAPHl02Yxyw1ZpGD'
    res = requests.get('https://api.smartbraincloud.net:8099/api/v1/auth/login', verify=False, auth=HTTPBasicAuth(user, password))

    res_string = res.text
    token = (res_string.split(",")[0])[16:-1]
    suffix = 'Bearer '
    access_token = suffix + token
    head = {"Authorization": access_token}    

    #######################

    url1 = "https://api.smartbraincloud.net:8099/api/v1/Posicao/ExtratosAtivos"
    url2 = "https://api.smartbraincloud.net:8099/api/v1/Rentabilidade/Ativos"  

    indexador = 1

    data1= {'codigoUsuario': cod_user, "codigoIndexador": indexador, "dataReferencia": data_ref_st}
    resp1 = requests.post(url1, headers=head, json=data1)
    df1 = pd.read_json(resp1.text)
    df1 = (pd.concat([df1.drop(['posicaoExtratosAtivos'], axis=1),
                    df1['posicaoExtratosAtivos'].apply(pd.Series)], axis=1)).dropna()
    #df1 = df1[['ativos', 'seguimento', 'saldoBruto', 'participacao']]
    df1 = df1[['seguimento', 'saldoBruto', 'participacao']]
    df1 = df1.rename(columns={'seguimento':'Classe'})#.set_index('Classe')
    df1.reset_index(inplace=True, drop=True)

    #######################

    data2= {'codigoUsuario': cod_user, "dataReferencia": data_ref_st}
    resp2 = requests.post(url2, headers=head, json=data2)
    df2 = pd.read_json(resp2.text)
    df2 = pd.concat([df2.drop(['rentabilidadeAtivos'], axis=1), df2['rentabilidadeAtivos'].apply(pd.Series)], axis=1)
    df2 = df2.filter(['nomeEstrategiaAtivo','nomeAtivo', 'rent_Mes', 'rent_12M'])
    df2 = df2.rename(columns={'nomeEstrategiaAtivo':'Classe', 'nomeAtivo': 'ativos'})#.set_index('Classe')

    #######################

    df3 = pd.concat([df2, df1[['saldoBruto', 'participacao']]], axis=1)

    df3['Contribuicao_Ret'] = (df3.participacao * df3.rent_Mes) / 100

    #asset_cont = df5['Contribuicao_Ret'].sort_values(ascending=False)
    #asset_cont.plot(kind= 'bar', figsize= (20, 8), title= 'Atribuição de Resultado')

    df4 = pd.DataFrame(df3.groupby(by=['Classe'])['Contribuicao_Ret'].sum())
    df4.reset_index(inplace=True)

    aloc_aux = ['Caixa', 'Caixa', 'RF_Pós', 
        'Inflacao', 'Nominal', 'RV_Br', 'RV_Int', 'Moedas', 'MM', 'Cmdty', 'RE', 'Alt']


    df4['Name'] = df4['Classe'].replace(['Caixa', 'VALORES A  LIQUIDAR', 'Renda Fixa Pós Fixado', 'Inflação', 'Juros Nominal', 'Ações Brasil', 
        'Renda Variável Internacional', 'Moedas', 'Multimercado', 'Commodity', 'Real Estate', 'Alternativos'], aloc_aux)

    df4.drop('Classe', axis=1, inplace=True)
    df4 = df4.T

    df = pd.DataFrame(df4.values[0:1], columns=df4.iloc[1])
    

    df_pnl_atrib = pd.DataFrame(columns=aloc_w_no_fund)
    cols_df = list(df.columns)

    for i in range(len(cols_df)):
        lista = []
        for j in range(len(df)):
            lista.append(df.iloc[j, i])
        if cols_df[i] in list(df_pnl_atrib.columns):
            df_pnl_atrib[cols_df[i]] = lista

    df_pnl_atrib = df_pnl_atrib.replace(np.nan, 0)        
    df_pnl_atrib = df_pnl_atrib.add_prefix('CR_')

    return df_pnl_atrib

def get_asset_pnl_atrib(code_users, data_ref):        
    
    df_pnl_atrib = pd.DataFrame(columns=aloc_w_no_fund).add_prefix('CR_')    

    if  isinstance (code_users, int):            
        df_pnl_atrib = get_asset_pnl_atrib_aux(code_users, data_ref)    
    else:                        
        for cod_user in code_users:           
            try:                                                                         
                df_ = get_asset_pnl_atrib_aux(cod_user, data_ref)
                df_pnl_atrib = df_pnl_atrib.append(df_)
            except:
                pass   
    df_pnl_atrib = df_pnl_atrib.astype('float')

    return df_pnl_atrib

def get_df_rebal_filter(data_ref, df_, inst, w_target_aloc, caixa_trigger_pct, infinity_trigger_pct, vol_trigger_pct, pnl_atrib=True):

    df = df_[df_.instituicao == inst]
    df = df_[df_.instituicao == inst]

    df_caixa = df[df.Caixa > caixa_trigger_pct]
    df_caixa_inf = df_caixa[df_caixa.Infinity < infinity_trigger_pct]
    df_caixa_inf_vol = df_caixa_inf[df_caixa_inf.Vol < vol_trigger_pct]

    df_caixa_2 = df[df.Caixa < caixa_trigger_pct]
    df_caixa_inf_2 = df_caixa_2[df_caixa_2.Infinity < infinity_trigger_pct]

    df_filter = df_caixa_inf_vol.sort_values(by="PL")
    df_filter.sort_values(by="Infinity")

    df_filter_2 = df_caixa_inf_2.sort_values(by="PL")

    df_filter = df_filter[['KPI_ID', 'Nome', 'PL', 'CAGR', 'Ret_Mes_Atual', 'Vol', 'Infinity', 'Caixa', 'RF_Pós', 'Inflacao', 'Nominal',
    'RV_Br','RV_Int', 'Cmdty', 'MM', 'RE', 'Alt', 'Codigo Flow']] .sort_values(by="PL", ascending=False)

    df_filter_2 = df_filter_2[['KPI_ID', 'Nome', 'PL', 'CAGR', 'Vol', 'Infinity', 'Caixa', 'RF_Pós', 'Inflacao', 'Nominal',
    'RV_Br','RV_Int', 'Cmdty', 'MM', 'RE', 'Alt', 'Codigo Flow']] .sort_values(by="PL", ascending=False)
    df_filter_2.reset_index(inplace=True, drop=True)


    df_filter['Aloc_targ_Inf'] = np.ones(len(df_filter.Infinity))*infinity_trigger_pct
    df_filter['Dif_aloc_Inf'] = (infinity_trigger_pct - df_filter.Infinity)
    df_filter['Caixa_Off_Inf'] = ((df_filter.Caixa/100) - (w_target_aloc['Caixa']*df_filter.Infinity/100)) * 100
    df_filter['Caixa_Aloc_Inf'] = (w_target_aloc['Caixa']*df_filter.Infinity/100) * 100
    df_filter['Caixa_Off_Inf_Net'] = df_filter['Caixa_Off_Inf'] - df_filter['Dif_aloc_Inf']

    df_filter['Caixa_Total'] = df_filter['Caixa_Off_Inf_Net'] + (df_filter.Infinity + df_filter['Dif_aloc_Inf']) * w_target_aloc['Caixa']
    df_filter['Carteira_Adm_Off_Caixa'] = (100-infinity_trigger_pct) - df_filter['Caixa_Off_Inf_Net'] 
    df_filter = df_filter.sort_values(by="Caixa_Off_Inf_Net")
    df_filter.reset_index(inplace=True, drop=True)


    print('NumbPort:', len(df))
    print(f'NumbPort Caixa_Total (Carteira_Adm + Infinity) > {caixa_trigger_pct}:', len(df_caixa), np.round(len(df_caixa)/len(df),2))
    print(f'NumbPort Caixa_Total (Carteira_Adm + Infinity) > {caixa_trigger_pct} | Aloc_infinity < {infinity_trigger_pct}:', len(df_caixa_inf), np.round(len(df_caixa_inf)/len(df),2))
    print(f'NumbPort Caixa_Total (Carteira_Adm + Infinity) > {caixa_trigger_pct} | Aloc_infinity < {infinity_trigger_pct}: | Vol < {vol_trigger_pct}:', len(df_caixa_inf_vol), np.round(len(df_caixa_inf_vol)/len(df),2))
    print('Dif Aloc Infinity:', (df_filter['Dif_aloc_Inf']/100*df_filter.PL).sum())

    #df_warren_vol = df_warren_caixa[df_warren_caixa.Vol < vol_trigger_pct]
    #display(df_filter)

    print(f'NumbPort Caixa_Total (Carteira_Adm + Infinity) < {caixa_trigger_pct} | Aloc_infinity < {infinity_trigger_pct}:', len(df_caixa_inf_2), np.round(len(df_caixa_inf_2)/len(df),2))

    #display(df_filter_2)

    if pnl_atrib:
        df_assset_pnl_atrib = get_asset_pnl_atrib(df_filter.KPI_ID.values, data_ref)
        df_assset_pnl_atrib.reset_index(inplace=True, drop=True)
        df_filter = pd.concat([df_filter, df_assset_pnl_atrib], axis=1)
    else:
        pass

    return df_filter, df_filter_2

def risk_contribution(w,cov):

    ''''Compute the contributions to risk of the elements (asset class) of a potfolio'''

    #Portf_Vol = np.sqrt(np.transpose(w) @ cov @ w)
    cov_ = cov * 252

    Portf_Vol = np.sqrt(
    np.dot(w.T, np.dot(cov_, w))) 
    
    marginal_contribution = cov_ @ w
    risk_contrb = np.multiply(marginal_contribution, w.T) / Portf_Vol**2 #scaled (normalize) RC to portf variance -> sum = 1
    #risk_contrb = np.multiply(marginal_contribution, w.T) / Portf_Vol

    return risk_contrb

#First Order Taylor - linear approx  by Dif_Vol --. vector product between the gradient and the change in portf weights
def incremental_portf_risk(cov, w_1, w_2):
    
    cov_ = cov * 252
    marginal_contribution = cov_ @ np.array(w_1) * 100
    inc_vol = np.transpose(marginal_contribution) @ np.transpose(w_2 - w_1)  

    #vol_1 = np.sqrt(np.dot(w_1.T, np.dot(cov, w_1))) * np.sqrt(252)
    #vol_2 = np.sqrt(np.dot(w_2.T, np.dot(cov, w_2))) * np.sqrt(252)
    #inc_vol = (vol_2 - vol_1) * 100

    return inc_vol

def incremental_portf_risk_v2(df_alloc_pre_reb, df_alloc_pos_reb, tickers, df_data, df_ipca_m):
        
    df_alloc = pd.concat([df_alloc_pre_reb, df_alloc_pos_reb], axis=0)
    names_assets = {i: j for i, j in zip(aloc_w_no_fund, tickers)}
    df_expec_vol = []

    for z in range(len(df_alloc)):

        df_allocate = df_alloc.iloc[[z]]

        df_allocate.rename(columns=names_assets, inplace=True)
        df_allocate = df_allocate.T
        df_allocate = df_allocate.reset_index()
        df_allocate.columns = ['Name', 'Allocation']
        df_allocate = df_allocate.sort_values(by="Name", ignore_index=True)
        df_allocate['Allocation'] = df_allocate['Allocation'].apply(
            lambda x: x if x > 0.5 else 0)
        df_allocate = df_allocate[df_allocate['Allocation'] != 0]
        df_allocate = df_allocate.groupby(['Name'], as_index=False).sum()
        df_allocate = df_allocate.reset_index(drop=True)
        
        df_data_filter = df_data[df_allocate.Name]  # input df_data        
        pf = build_portfolio(df_data_filter, df_allocate, df_ipca_m)
        pf.risk_free_rate = 0.1150
        pf.freq = 252

        #expec_ret = pf.comp_expected_return() * 100
        expec_vol = pf.comp_volatility() * 100
        #cov_matrix = pf.comp_daily_log_returns().cov()
        #weights = pf.comp_weights()
        
        df_expec_vol.append(expec_vol)

    inc_vol = (df_expec_vol[1] - df_expec_vol[0]) * 100
    
    return inc_vol

def get_inc_risk_pos_reb(data_ref, df_pre_reb, df_pos_reb, tickers, df_data, df_ipca_m):
    
    rc_names = ['RC_' + str(z) for z in tickers]
    vol_names = ['VOL_' + str(z) for z in tickers]
    port_names = ['Exp_Ret', 'Exp_Vol', 'Inc_Risk_bps']

    df_rc_full = pd.DataFrame(columns=rc_names)
    df_vol_full = pd.DataFrame(columns=vol_names)
    port_exp = pd.DataFrame(columns=port_names)
    cols = port_names + vol_names + rc_names
    metrics_portf = pd.DataFrame(columns=cols)

    for n in range(len(df_pre_reb)):

        df_alloc_pre_reb = df_pre_reb.loc[[n]][aloc_w_no_fund]
        df_alloc_pos_reb = df_pos_reb.loc[[n]][aloc_w_no_fund]        
        df_allocate = df_alloc_pos_reb.copy()

        names_assets = {i: j for i, j in zip(aloc_w_no_fund, tickers)}
        df_allocate.rename(columns=names_assets, inplace=True)
        df_allocate = df_allocate.T
        df_allocate = df_allocate.reset_index()
        df_allocate.columns = ['Name', 'Allocation']
        df_allocate = df_allocate.sort_values(by="Name", ignore_index=True)
        df_allocate['Allocation'] = df_allocate['Allocation'].apply(
            lambda x: x if x > 0.5 else 0)
        df_allocate = df_allocate[df_allocate['Allocation'] != 0]
        df_allocate = df_allocate.groupby(['Name'], as_index=False).sum()
        df_allocate = df_allocate.reset_index(drop=True)                    
        

        df_data_filter = df_data[df_allocate.Name]  # input df_data
        pf = build_portfolio(df_data_filter, df_allocate, df_ipca_m)
        pf.risk_free_rate = 0.1150
        pf.freq = 252

        expec_ret = pf.comp_expected_return() * 100
        expec_vol = pf.comp_volatility() * 100
        cov_matrix = pf.comp_daily_log_returns().cov()
        weights = pf.comp_weights()

        ##################################################

        #aux_1 = df_alloc_pre_reb.copy()
        #aux_2 = df_alloc_pos_reb.copy()
        #aux_1.rename(columns=names_assets, inplace=True)
        #aux_2.rename(columns=names_assets, inplace=True)

        #w_1 = np.array(aux_1[df_allocate.Name]).flatten()
        #w_2 = np.array(aux_2[df_allocate.Name]).flatten()
        
        inc_risk = incremental_portf_risk_v2(df_alloc_pre_reb, df_alloc_pos_reb, tickers, df_data, df_ipca_m)
        #inc_risk = incremental_portf_risk(cov_matrix, w_1, w_2)

        ##################################################

        df_rc = pd.DataFrame(risk_contribution(
            np.array(weights), cov_matrix)).T
        df_rc = df_rc.add_prefix('RC_')

        df_vol = pd.DataFrame(pf.comp_stock_volatility() * 100).T
        df_vol = df_vol.add_prefix('VOL_')

        cols_rc = list(df_rc.columns)
        cols_vol = list(df_vol.columns)

        for i in range(len(cols_vol)):
            lista = []
            for j in range(len(df_vol)):
                lista.append(df_vol.iloc[j, i])
            if cols_vol[i] in list(df_vol_full.columns):
                df_vol_full[cols_vol[i]] = lista

        for i in range(len(cols_rc)):
            lista = []
            for j in range(len(df_rc)):
                lista.append(df_rc.iloc[j, i])
            if cols_rc[i] in list(df_rc_full.columns):
                df_rc_full[cols_rc[i]] = lista

        port_exp.loc[0] = [expec_ret, expec_vol, inc_risk]
        portf_infos = pd.concat([port_exp, df_vol_full, df_rc_full], axis=1)
            
        metrics_portf = pd.concat([metrics_portf, portf_infos], axis=0)
    metrics_portf.reset_index(inplace=True, drop=True)
        
    return metrics_portf

def target_aloc_infinity(end_dt, df_full, df_pre_rebal, w_dif, code_sb, lista_new_aloc, index_names, w_target_aloc, tickers):
    
    col_ID = 'KPI_ID'
    col_aloc_pct = 'Aloc_targ_Inf'    
    
    def procura(df_pre_rebal, col_ID, value):        
        rows = df_pre_rebal.index[df_pre_rebal[col_ID] == value].tolist()
        if len(rows) == 0:
            return -1 
        else:
            return rows[0] 
        
    indices = []
    lista2 = []
    for j in range(len(code_sb)):
        indices.append(procura(df_pre_rebal, col_ID, code_sb[j]))
    
    for i in range(len(df_pre_rebal)):
        if i in indices:
            lista2.append(lista_new_aloc[indices.index(i)])
        else: lista2.append(df_pre_rebal.loc[i, col_aloc_pct])
    
    df_pre_rebal['Aloc_targ_Inf_Adj'] = lista2
    df_pre_rebal['Aloc_targ_Inf'] = df_pre_rebal.Aloc_targ_Inf_Adj
    df_pre_rebal['Dif_aloc_Inf'] = (df_pre_rebal.Aloc_targ_Inf_Adj - df_pre_rebal.Infinity)
    df_pre_rebal['Caixa_Off_Inf'] = (((df_pre_rebal.Caixa/100) - (w_target_aloc['Caixa']*df_pre_rebal.Infinity/100)) * 100) 
    df_pre_rebal['Caixa_Aloc_Inf'] = (w_target_aloc['Caixa']*df_pre_rebal.Aloc_targ_Inf_Adj/100) * 100
    df_pre_rebal['Caixa_Off_Inf_Net'] = df_pre_rebal['Caixa_Off_Inf'] - df_pre_rebal['Dif_aloc_Inf'] #assumindo que sai caixa e entra no fundo | não sai de nenhum outro título da carteira
    df_pre_rebal['Caixa_Total'] = df_pre_rebal['Caixa_Off_Inf_Net'] + (df_pre_rebal.Infinity + df_pre_rebal['Dif_aloc_Inf']) * w_target_aloc['Caixa']

    df_pre_rebal['Carteira_Adm_Off_Caixa'] = (100-df_pre_rebal.Aloc_targ_Inf_Adj) - df_pre_rebal['Caixa_Off_Inf_Net'] 
    #df_filter_adj = df_filter_adj.sort_values(by="Caixa_Off_Inf_Net")
    df_pre_rebal.drop('Aloc_targ_Inf_Adj', axis=1, inplace=True)

    df_pre_rebal = df_pre_rebal.sort_values(by="KPI_ID")
    df_pre_rebal.reset_index(inplace=True, drop=True)

    #########################################################################
    df_w_dif = pd.DataFrame(columns=aloc_w_no_fund)
    df_w_dif.loc[0] = w_dif

    reb_infinity = pd.DataFrame(columns=aloc_w_no_fund)    

    df_pos_rebal = df_full[df_full["KPI_ID"].isin(df_pre_rebal.KPI_ID.values.tolist())]
    df_pos_rebal = df_pos_rebal.sort_values(by="KPI_ID")
    df_pos_rebal.reset_index(inplace=True, drop=True)

    #df_pos_rebal = df_pos_rebal[index_names]
    #df_pos_rebal['Aloc_targ_Inf'] = df_pre_rebal['Aloc_targ_Inf']
    #df_pos_rebal['Dif_aloc_Inf'] = df_pre_rebal['Dif_aloc_Inf']

    for n in range(len(df_pre_rebal)):
        reb_infinity.loc[n] = df_pre_rebal[aloc_w_no_fund].loc[n].values + df_pre_rebal.Dif_aloc_Inf[n]/100*np.array(list(w_target_aloc.values()))*100 

    reb_infinity['Caixa'] = reb_infinity.Caixa - df_pre_rebal.Dif_aloc_Inf
    df_pos_rebal[aloc_w_no_fund] = reb_infinity
    df_pos_rebal['Sum_w'] = df_pos_rebal[aloc_w_no_fund].sum(axis=1)
    df_pos_rebal['Infinity'] = df_pre_rebal['Aloc_targ_Inf']

    #########################################################################
    
    shift_dataset = 2 #years        
    end_dataset_aux = pd.to_datetime(end_dt)
    end_dataset = date(end_dataset_aux.year, end_dataset_aux.month, end_dataset_aux.day)
    start_dataset = date(end_dataset.year-shift_dataset, end_dataset.month, end_dataset.day)
    df_data, df_ipca_m = get_prices(tickers, start_dataset, end_dataset) 

    inc_risk_pos_reb = get_inc_risk_pos_reb(end_dt, df_pre_rebal, df_pos_rebal, tickers, df_data, df_ipca_m)    
    df_pos_rebal = pd.concat([df_pos_rebal, inc_risk_pos_reb], axis=1)

    #df_pos_rebal = df_pos_rebal[['KPI_ID', 'Nome', 'PL', 'Infinity', 'Vol', 'Inc_Risk_bps', 'Caixa', 'Inflacao', 'Nominal',
    #'RV_Br','RV_Int', 'Cmdty', 'MM', 'RE', 'Alt']]

    df_pos_rebal = df_pos_rebal[index_names]

    df_pos_rebal.rename(columns = {'Vol':'Vol_Pre_Rebal'}, inplace=True)
    df_pos_rebal['Vol_Pos_Rebal'] = df_pos_rebal.Vol_Pre_Rebal + df_pos_rebal.Inc_Risk_bps.iloc[:,-1] / 100
    df_pos_rebal.reset_index(inplace=True, drop=True)
    

    return df_pre_rebal, df_pos_rebal


def get_updated_aloc(cod_user, data_ref):
    
    if not isinstance(data_ref, str):            
        data_ref_st = data_ref.strftime('%Y-%m-%d')            
    else:
        data_ref_st = data_ref

    
    url = "https://api.smartbraincloud.net:8099/api/v1/Posicao/PosicaoAtivo"
    head = {"Authorization": access_token}

    data = {'codigoUsuario': cod_user, "dataReferencia": data_ref_st}
    resp = requests.post(url, headers=head, json=data)
    df_pos_ativo = pd.read_json(resp.text)
    try:
        df_pos_ativo = pd.concat([df_pos_ativo.drop(['posicaoAtivos'], axis=1),
                                    df_pos_ativo['posicaoAtivos'].apply(pd.Series)], axis=1)

        #df_pos_ativo = df_pos_ativo.set_index('Date')
        #end_date = pd.to_datetime(df_pos_ativo.index[-1]).strftime("%Y-%m-%d") #last date of df

        ###################################################################################

        df_posicao_ativo = df_pos_ativo.groupby('nomeSegAtivo').sum()

        pl_total = np.round(df_posicao_ativo['saldoBruto'].sum(), 2)
    except:
        pass

    try:
        df_ = df_pos_ativo[['descricao', 'saldoBruto',]]
        df_ = df_.set_index('descricao')
    
    except:
        pass

    try:
        aloc_caixa = df_posicao_ativo.loc['Caixa', 'porcentagem'].round(2)
    except:
        aloc_caixa = 0
    try:
        aloc_rf_pos = df_posicao_ativo.loc['Renda Fixa Pós Fixado', 'porcentagem'].round(
                2)
    except:
        aloc_rf_pos = 0
    try:
        aloc_axiom_fund = ((df_[df_.index == 'FUNDO AXIOM INFINITY'].saldoBruto.values[0] / pl_total) * 100).round(2)
    except:
        aloc_axiom_fund = 0
    try:
        aloc_rvbr = df_posicao_ativo.loc['Ações Brasil', 'porcentagem'].round(
                2)
    except:
        aloc_rvbr = 0
    try:
        aloc_rvint = df_posicao_ativo.loc['Renda Variável Internacional', 'porcentagem'].round(
                2)
    except:
        aloc_rvint = 0
    try:
        aloc_alt = df_posicao_ativo.loc['Alternativos', 'porcentagem'].round(
                2)
    except:
        aloc_alt = 0
    try:
        aloc_commo = df_posicao_ativo.loc['Commodity', 'porcentagem'].round(
                2)
    except:
        aloc_commo = 0
    try:
        aloc_moedas = df_posicao_ativo.loc['Moedas', 'porcentagem'].round(
                2)
    except:
        aloc_moedas = 0
    try:
        aloc_re = df_posicao_ativo.loc['Real Estate', 'porcentagem'].round(
                2)
    except:
        aloc_re = 0
    try:
        aloc_infl = df_posicao_ativo.loc['Inflação', 'porcentagem'].round(
                2)
    except:
        aloc_infl = 0
    try:
        aloc_nom = df_posicao_ativo.loc['Juros Nominal', 'porcentagem'].round(
                2)
    except:
        aloc_nom = 0
    try:
        aloc_mm = (df_posicao_ativo.loc['Multimercado', 'porcentagem'].round(
                2)) - aloc_axiom_fund
    except:
        aloc_mm = 0

    to_append_aloc = [pl_total, aloc_caixa, aloc_rf_pos, aloc_axiom_fund, aloc_infl, aloc_nom,
                        aloc_rvbr, aloc_rvint, aloc_commo, aloc_moedas, aloc_mm, aloc_re, aloc_alt]



    return to_append_aloc

def performance_metrics(df):

    start_date = pd.to_datetime(df.index[0]).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d")
    dif_days = np.busday_count(start_date, end_date)           

    cagr = qs.stats.cagr(df.loc[:, 'ret']) * 100
    sharpe = qs.stats.sharpe(df.loc[:, 'ret'])
    win_r = qs.stats.win_rate(df.loc[:, 'ret']) * 100
    var_95 = qs.stats.var(df.loc[:, 'ret']) * 100
    vol = qs.stats.volatility(df.loc[:, 'ret']) * 100
    es = qs.stats.expected_shortfall(df.loc[:, 'ret']) * 100
    # av_loss = qs.stats.avg_loss(df_expanding.loc[:, 'ret'])

    # rol_vol = qs.stats.rolling_volatility(df_expanding.loc[:, 'ret'], rolling_period=21)
    # rol_sharpe = qs.stats.rolling_sharpe(df_expanding.loc[:, 'ret'], rf=0., rolling_period=21)
    payoff_ratio = qs.stats.payoff_ratio(df.loc[:, 'ret'])
    profit_factor = qs.stats.profit_factor(df.loc[:, 'ret'])
    max_drawdown = qs.stats.max_drawdown(df.loc[:, 'ret']) * 100
    # to_drawdown_ = qs.stats.to_drawdown_series(df_expanding.loc[:, 'ret'])

    ult_cota = df['cota'][-1]
    hpr = ((ult_cota / 1) - 1) * 100
    # hpr_annual = (ult_cota / 1) ** (252/dif_days) - 1
    mes_atual = df.iloc[-1, 2]
    ano_atual = df.iloc[-1, 3]
    df_mes = df.loc[df['mes'] == mes_atual]
    df_ok = df_mes.loc[df_mes['ano'] == ano_atual]

    ret_mes_atual = (((df_ok.cota[-1] / df_ok.cota[0]) - 1)*100).round(4)
    start_dt_mes = pd.to_datetime(df_ok.index[0]).strftime("%Y-%m-%d")
    end_dt_mes = pd.to_datetime(df_ok.index[-1]).strftime("%Y-%m-%d")
    dif_days_mes = np.busday_count(start_dt_mes, end_dt_mes)

    df_ytd = df.loc[df['ano'] == ano_atual]
    ret_ytd = (((df_ytd.cota[-1] / df_ytd.cota[0]) - 1)*100).round(4)

    start_dt_ytd = pd.to_datetime(df_ytd.index[0]).strftime("%Y-%m-%d")
    end_dt_ytd = pd.to_datetime(df_ytd.index[-1]).strftime("%Y-%m-%d")
    dif_days_ytd = np.busday_count(start_dt_ytd, end_dt_ytd)

    df_ytd['cdi_acum'] = (1 + df_ytd.cdi).cumprod() - 1
    cdi_porc_ytd = ((ret_ytd/100+1)**(1/dif_days_ytd)-1) / \
        ((df_ytd['cdi_acum'][-1]+1)**(1/dif_days_ytd)-1)*100

    df_ok['cdi_acum'] = (1 + df_ok.cdi).cumprod() - 1
    cdi_porc_mes = ((ret_mes_atual/100+1)**(1/dif_days_mes)-1) / \
        ((df_ok['cdi_acum'][-1]+1)**(1/dif_days_mes)-1)*100

    df['cdi_acum'] = (1 + df.cdi).cumprod() - 1
    cdi_acum_tt = df['cdi_acum'][-1] * 100
    cdi_porc_tt = ((hpr/100+1)**(1/dif_days)-1) / \
        ((df['cdi_acum'][-1]+1)**(1/dif_days)-1)*100

    performance_metrics = [end_date, ult_cota, hpr, cdi_acum_tt, cdi_porc_tt, ret_mes_atual, cdi_porc_mes, ret_ytd, cdi_porc_ytd, cagr, vol,
        sharpe, win_r, var_95, es, payoff_ratio, profit_factor, max_drawdown]

    return performance_metrics

def create_portf(data_ref, cod_user, df_allocation, w_target, tickers, infinity_aloc, df_data, df_ipca_m):
            
    rc_names = ['RC_' + str(z) for z in tickers]
    vol_names = ['VOL_' + str(z) for z in tickers]
    
    df_rc_full = pd.DataFrame(columns=rc_names)
    df_vol_full = pd.DataFrame(columns=vol_names)
    port_exp = pd.DataFrame(columns=['Exp_Ret', 'Exp_Vol'])
                
    qty_infinity = (df_allocation['Infinity'][0]/100 *np.array(list(infinity_aloc.values())))*100    
    df_alloc = df_allocation[aloc_w_no_fund] + qty_infinity
        
    #O df_allocate considera o df_alloc_target para criar o portfólio e as métricas (mat. cov | ret e vol esp)
    #Caso haja a adição de um ativo no df_alloc_target que não tinha no df_alloc original, precisamos adicionar esse ativo na mat de cov para calcular o MRC
    if w_target is not None:        
        qty_infinity_target = (df_allocation['Infinity'][0]/100 *np.array(w_target))*100
        df_alloc_target = df_allocation[aloc_w_no_fund] + qty_infinity_target
        df_allocate = df_alloc_target.copy()
    else:
        df_allocate = df_alloc.copy()        
                
    names_assets = {i: j for i, j in zip(aloc_w_no_fund, tickers)}
    df_allocate.rename(columns=names_assets, inplace=True)
    df_allocate = df_allocate.T
    df_allocate = df_allocate.reset_index()
    df_allocate.columns = ['Name', 'Allocation']
    df_allocate = df_allocate.sort_values(by="Name", ignore_index=True)
    df_allocate['Allocation'] = df_allocate['Allocation'].apply(
            lambda x: x if x > 0.5 else 0)
    df_allocate = df_allocate[df_allocate['Allocation'] != 0]
    df_allocate = df_allocate.groupby(['Name'], as_index=False).sum()
    df_allocate = df_allocate.reset_index(drop=True)

    #shift_dataset = 2 #years        
    #end_dataset_aux = pd.to_datetime(data_ref)
    #end_dataset = date(end_dataset_aux.year, end_dataset_aux.month, end_dataset_aux.day)
    #start_dataset = date(end_dataset.year-shift_dataset, end_dataset.month, end_dataset.day)
    #df_data_filter, df_ipca_m = get_prices(df_allocate.Name, start_dataset, end_dataset)        
    
    df_data_filter = df_data[df_allocate.Name]  # input df_data
    #df_data_filter = df_data_filter[(df_data_filter.index >= pd.to_datetime(start_dataset)) & (
    #df_data_filter.index <= pd.to_datetime(end_dataset))]

    pf = build_portfolio(df_data_filter, df_allocate, df_ipca_m)
    pf.risk_free_rate = 0.1150
    pf.freq = 252
    print(cod_user, data_ref)
    # return_means = pf.comp_mean_returns()
    # vol_ativos = pf.comp_stock_volatility() * 100
    expec_ret = pf.comp_expected_return() * 100        
    expec_vol = pf.comp_volatility() * 100
    cov_matrix = pf.comp_daily_log_returns().cov()
    weights = pf.comp_weights()
    
    ##################################################    
    if w_target is not None:
        aux_1 = df_alloc.copy()
        aux_2 = df_alloc_target.copy()
        aux_1.rename(columns=names_assets, inplace=True)
        aux_2.rename(columns=names_assets, inplace=True)
        
        w_1 = aux_1[df_allocate.Name]
        w_2 = aux_2[df_allocate.Name]
        w_1 = w_1.groupby(by=w_1.columns, axis=1).sum()        
        w_2 = w_2.groupby(by=w_2.columns, axis=1).sum()
        #w_1 = np.array(aux_1[df_allocate.Name]).flatten()
        #w_2 = np.array(aux_2[df_allocate.Name]).flatten()
        #inc_risk = incremental_portf_risk(cov_matrix, w_1, w_2)   
        inc_risk = incremental_portf_risk_v2(w_1, w_2, tickers, df_data, df_ipca_m)             
    else:
        inc_risk = 0
    ##################################################
    
    df_rc = pd.DataFrame(risk_contribution(np.array(weights), cov_matrix)).T    
    df_rc = df_rc.add_prefix('RC_')

    df_vol = pd.DataFrame(pf.comp_stock_volatility() * 100).T
    df_vol = df_vol.add_prefix('VOL_')

    cols_rc = list(df_rc.columns)
    cols_vol = list(df_vol.columns)

    for i in range(len(cols_vol)):
        lista = []
        for j in range(len(df_vol)):
            lista.append(df_vol.iloc[j, i])
        if cols_vol[i] in list(df_vol_full.columns):
            df_vol_full[cols_vol[i]] = lista

    for i in range(len(cols_rc)):
        lista = []
        for j in range(len(df_rc)):
            lista.append(df_rc.iloc[j, i])
        if cols_rc[i] in list(df_rc_full.columns):
            df_rc_full[cols_rc[i]] = lista        

    port_exp.loc[0] = [expec_ret, expec_vol] 
    port_exp['Inc_Risk_bps'] = inc_risk
    
    if w_target is not None:
        metrics_portf = pd.concat([port_exp, pd.concat([df_allocation[['PL', 'Infinity']], df_alloc_target], axis=1),
                                df_vol_full, df_rc_full], axis=1)                                           
    else:
        metrics_portf = pd.concat([port_exp, pd.concat([df_allocation[['PL', 'Infinity']], df_alloc], axis=1),
                                df_vol_full, df_rc_full], axis=1)                                           
    
    return metrics_portf


def get_rf_pos_assets_weights(df_filter_rf, df_rebal_filter, assets, col_names):

    #col_names = ['KPI_ID', 'CDB','CRA','CRI','LCA','LCI']
    df_data = pd.DataFrame(columns=col_names)
    df_data_full = pd.DataFrame(columns=col_names)
    df_filter = df_filter_rf[df_filter_rf['ativos'].str.contains(assets, na=False)]

    df_filter_others = df_filter_rf[~df_filter_rf['ativos'].str.contains(assets, na=False)]
    df_filter_others_sum = pd.DataFrame(df_filter_others.groupby('KPI_ID')['saldoBruto'].sum() )
    df_filter_others_sum.reset_index(inplace=True)

    for z in df_filter.KPI_ID.unique():
        lista_1 = []
        pl_tt = df_rebal_filter[df_rebal_filter.KPI_ID == z].PL.sum()
        df_single = df_filter[df_filter.KPI_ID == z]

        for i in range(len(df_single)):
            lista_1.append(df_single.ativos.iloc[i][:3])
            
        df_single['x'] = lista_1
        df_single3 = (df_single.groupby(
                'x')['saldoBruto'].sum() / pl_tt) * 100
        df_single3 = pd.DataFrame(df_single3).T
        df_single3['KPI_ID'] = z

        cols_list = list(df_single3.columns)       
        df_data = pd.DataFrame(columns=col_names)

        for i in range(len(cols_list)):    
            lista_2 = []

            for j in range(len(df_single3)):
                lista_2.append(df_single3.iloc[j, i])            
                if cols_list[i] in list(df_data.columns):
                    df_data[cols_list[i]] = lista_2            
        
        try:
            pct_fund_overall = (df_filter_others_sum[df_filter_others_sum.KPI_ID == z].saldoBruto.values[0] / pl_tt) * 100
            df_data['Outros'] = pct_fund_overall
        except:
            df_data['Outros'] = 0

        df_data_full = pd.concat([df_data_full, df_data], axis=0)
        df_data_full = df_data_full.fillna(0)

    return df_data_full

    """ 
    lista = []
    for i in range(len(df_single)):
        separador = ''
        lst = re.findall('\S+%+', df_single.ativos.iloc[i])
        lista.append(separador.join(lst))
    #print(lista)
    df_single['%']=lista
    #display(df)

    df_single2 = df_single.groupby('%')['saldoBruto'].sum()
    display(df_single2) """


def get_metrics(data_ref, aloc_w_no_fund, infinity_aloc, cod_user, tickers):        
    
    idx_metrics=['KPI_ID', 'data_inicio', 'Nome', 'data_final', 'Cota', 
            'HPR', 'CDI_Acum', 'CDI_Pct_TT', 'Ret_Mes_Atual', 'CDI_Pct_Mes_Atual', 'Ret_Acum_YTD', 'CDI_Pct_YTD', 'CAGR', 'Vol', 'Sharpe',
            'Win_r', 'VaR_95', 'ES', 'Payoff_Ratio', 'Profit_Factor', 'Max_DD', 'Exp_Ret', 'Exp_Vol']

    idx_aloc = ['PL', 'Caixa', 'RF_Pós', 'Infinity', 'Inflacao', 'Nominal', 'RV_Br',
                'RV_Int', 'Cmdty', 'MM', 'RE', 'Alt']

    rc_names = ['RC_' + str(z) for z in tickers]
    vol_names = ['VOL_' + str(z) for z in tickers]
    
    df_idx_metrics = pd.DataFrame(columns=idx_metrics)    
    df_idx_aloc = pd.DataFrame(columns=idx_aloc)

    df_rc_full = pd.DataFrame(columns=rc_names)
    df_vol_full = pd.DataFrame(columns=vol_names)

    cols = idx_metrics + idx_aloc + vol_names + rc_names
    df_metrics = pd.DataFrame(columns=cols)
    
    ###################
    #try:
    to_append_idx_metrics = get_sb_infos_metrics(cod_user, data_ref)

    to_append_aloc = get_updated_aloc(cod_user, data_ref)
        
    df_idx_aloc.loc[0] = to_append_aloc

    metrics_portf = create_portf(data_ref, cod_user, df_idx_aloc, aloc_w_no_fund, tickers, infinity_aloc)        
    ###################
        
    df_idx_metrics.loc[0] = to_append_idx_metrics                                      
    
    df_metrics = pd.concat([df_idx_metrics, metrics_portf], axis=1)                                              
    df_metrics = df_metrics.fillna(0)
    
    return df_metrics

def get_sb_infos_metrics(cod_user, data_ref, w_target, tickers, infinity_aloc, df_data, df_ipca_m, dt_range = True):
        
        idx_metrics=['KPI_ID', 'data_inicio', 'Nome', 'data_final', 'Cota', 
            'HPR', 'CDI_Acum', 'CDI_Pct_TT', 'Ret_Mes_Atual', 'CDI_Pct_Mes_Atual', 'Ret_Acum_YTD', 'CDI_Pct_YTD', 'CAGR', 'Vol', 'Sharpe',
            'Win_r', 'VaR_95', 'ES', 'Payoff_Ratio', 'Profit_Factor', 'Max_DD']

        idx_aloc = ['PL', 'Caixa', 'RF_Pós', 'Infinity', 'Inflacao', 'Nominal', 'RV_Br',
                        'RV_Int', 'Cmdty', 'Moedas', 'MM', 'RE', 'Alt']
        
        idx_port_infos = ['Exp_Ret', 'Exp_Vol', 'Inc_Risk_bps', 'PL', 'Infinity', 'Caixa', 'RF_Pós', 'Inflacao', 'Nominal', 'RV_Br',
                        'RV_Int', 'Cmdty', 'Moedas', 'MM', 'RE', 'Alt']

        rc_names = ['RC_' + str(z) for z in tickers]
        vol_names = ['VOL_' + str(z) for z in tickers]
        
        df_idx_metrics = pd.DataFrame(columns=idx_metrics)    
        df_idx_aloc = pd.DataFrame(columns=idx_aloc)
        
        cols = idx_metrics + idx_port_infos + vol_names + rc_names
        df_metrics = pd.DataFrame(columns=cols)
        
        ##################################################################################

        indexador = 1
        nome = clientes.loc[clientes['codigoCliente'] == cod_user, 'nomeCliente']
        url = "https://api.smartbraincloud.net:8099/api/v1/EvolucaoRentabilidade"
        head = {"Authorization": access_token}

        data_ref_st = data_ref.strftime('%Y%m%d')

        data = {'codigoUsuario': cod_user,
                "codigoIndexador": indexador, "dataReferencia": data_ref_st}
        resp = requests.post(url, headers=head, json=data)

        #perf_metrics_res = []

        nome = nomes.loc[cod_user, 'Nome']        
        df = pd.read_json(resp.text)
        if len(df) > 5:
                df = pd.concat([df.drop(['evolucaoRentabilidade'], axis=1),
                        df['evolucaoRentabilidade'].apply(pd.Series)], axis=1)
                cota = (df['cotaCarteira'] / 100) + 1
                df['cota'] = cota
                cdi = (df['cotaIndexadora'] / 100) + 1
                df['cdi'] = cdi
                df['ret'] = df['cota'].pct_change()
                df['cdi'] = df['cdi'].pct_change()
                df['Date'] = pd.to_datetime(df['dataRegistro'], format='%Y/%m/%d')
                df = df.set_index('Date')

                start_date = pd.to_datetime(df.index[0]).strftime("%Y-%m-%d")
                end_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d")
                dif_days = np.busday_count(start_date, end_date)
                data_inicio = df.first_valid_index()
        
                if dt_range:                                                       
                    dt_interval = pd.date_range(start=start_date, end=end_date, freq='30D')
                        
                    for dt in np.arange(1,len(dt_interval)):
                            df_expanding = df.loc[start_date:dt_interval[dt]]  
                            end_date = pd.to_datetime(df_expanding.index[-1]).strftime("%Y-%m-%d") #last date of df_expanding

                            perf_metrics = performance_metrics(df_expanding) #row output                                                        

                            df_idx_aloc.loc[0] = get_updated_aloc(cod_user, end_date)

                            port_infos = create_portf(end_date, cod_user, df_idx_aloc, w_target, tickers, infinity_aloc, df_data, df_ipca_m)
                                                                
                            df_idx_metrics.loc[0] = [cod_user, data_inicio, nome] + perf_metrics #dataframe
                                
                            teste_concat = pd.concat([df_idx_metrics, port_infos], axis=1)        
                            df_metrics = df_metrics.append(teste_concat)
                                                      
                else:
                        
                    end_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d") #last date of df

                    perf_metrics = performance_metrics(df) #row output                                                        

                    df_idx_aloc.loc[0] = get_updated_aloc(cod_user, end_date)

                    port_infos = create_portf(end_date, cod_user, df_idx_aloc, w_target, tickers, infinity_aloc, df_data, df_ipca_m)
                                                                
                    df_idx_metrics.loc[0] = [cod_user, data_inicio, nome] + perf_metrics #dataframe
                                
                    teste_concat = pd.concat([df_idx_metrics, port_infos], axis=1)        
                    df_metrics = df_metrics.append(teste_concat)

        return df_metrics

def get_port_analytic_sb(code_users, data_ref, w_target, tickers, infinity_aloc, dt_range = False):

    idx_metrics=['KPI_ID', 'data_inicio', 'Nome', 'data_final', 'Cota', 
            'HPR', 'CDI_Acum', 'CDI_Pct_TT', 'Ret_Mes_Atual', 'CDI_Pct_Mes_Atual', 'Ret_Acum_YTD', 'CDI_Pct_YTD', 'CAGR', 'Vol', 'Sharpe',
            'Win_r', 'VaR_95', 'ES', 'Payoff_Ratio', 'Profit_Factor', 'Max_DD']
        
    idx_port_infos = ['Exp_Ret', 'Exp_Vol', 'Inc_Risk_bps', 'PL', 'Infinity', 'Caixa', 'RF_Pós', 'Inflacao', 'Nominal', 'RV_Br',
                        'RV_Int', 'Cmdty', 'Moedas', 'MM', 'RE', 'Alt']

    shift_dataset = 2 #years        
    end_dataset_aux = pd.to_datetime(data_ref)
    end_dataset = date(end_dataset_aux.year, end_dataset_aux.month, end_dataset_aux.day)
    start_dataset = date(end_dataset.year-shift_dataset, end_dataset.month, end_dataset.day)
    df_data, df_ipca_m = get_prices(tickers, start_dataset, end_dataset)    

    rc_names = ['RC_' + str(z) for z in tickers]
    vol_names = ['VOL_' + str(z) for z in tickers]            
    cols = idx_metrics + idx_port_infos + vol_names + rc_names
    df_metrics = pd.DataFrame(columns=cols)        

    if code_users == 'All':    
        for cod_user in cod_users: 
                  
            try:                                                                         
                df_metrics = pd.concat([df_metrics, get_sb_infos_metrics(cod_user, data_ref, w_target, tickers, infinity_aloc, df_data, df_ipca_m, dt_range = dt_range)])                    
            except:
                pass    
    else:                
        df_metrics = get_sb_infos_metrics(code_users, data_ref, w_target, tickers, infinity_aloc, df_data, df_ipca_m, dt_range = dt_range)

    df_merge_inst = pd.merge(df_metrics, 
                      df_merge(), 
                      on ='KPI_ID', 
                      how ='inner')
    
    df_metrics = df_metrics.drop_duplicates(subset=["KPI_ID"], keep="first")
    df_merge_inst = df_merge_inst.drop_duplicates(subset=["KPI_ID"], keep="first")

    print('lenght df_metrics:', len(df_metrics))
    print('lenght df_merge_inst:', len(df_merge_inst))    

    return df_metrics, df_merge_inst
    #return df_metrics










