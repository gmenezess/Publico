import numpy as np
import pyettj.ettj as ettj
import pandas as pd
import pytz
import datetime
from datetime import datetime as date
import itertools
import plotly.graph_objects as go
import os


PATH = os.path.abspath("feriados.xlsx")
#PATH = "/Users/Dell/OneDrive//Área de Trabalho/User02/AXIOM/Axiom_Feed"


def get_implicita_fwd(end_dt, pz_days):

    end_dt_str = end_dt.strftime('%d/%m/%Y')
    df_curves = ettj.get_ettj(end_dt_str)        
    df_curves['Implícita'] = ((1+df_curves['DI x pré 252']/100) / (1+df_curves['DI x IPCA 252']/100) - 1 ) * 100
    df_curves_ = df_curves.filter(items=['Dias Corridos', 'Anos', 'DI x pré 252', 'DI x IPCA 252', 'Implícita'])
    implicita_interpol = np.interp(pz_days, np.array(df_curves_['Dias Corridos']), np.array(df_curves_['Implícita']))
    
    return implicita_interpol

def get_juro_real_fwd(end_dt, pz_days):

    end_dt_str = end_dt.strftime('%d/%m/%Y')
    df_curves = ettj.get_ettj(end_dt_str)        
    juro_real_interpol = np.interp(pz_days, np.array(df_curves['Dias Corridos']), np.array(df_curves['DI x IPCA 252']))
    
    return juro_real_interpol


#curva_juros = interp1d(days, taxas, kind='linear')

def dv01(days_vcto, taxa, delta):    
    
    duration = days_vcto/252

    pv_positivo = (1 + (taxa + delta)) ** (-duration)
    pv_negativo = (1 + (taxa - delta)) ** (-duration)    
    
    p_plus = (1 - duration * delta) ** (-1)
    p_minus = (1 + duration * delta) ** (-1)

    var_pv = pv_negativo - pv_positivo 
    dv01_val = var_pv / (2 * delta)    
    #convexidade = (pv_positivo + pv_negativo - 2) / (2  * delta ** 2)
    convexidade = (p_plus + p_minus - 2) / (2*delta**2)

    return dv01_val, duration, convexidade


def get_last_date(path):

    taxas = pd.read_excel(path)
    taxas = taxas.loc[:, ~taxas.columns.str.contains('^Unnamed')]
    last_date = pd.Timestamp(taxas['Date'].max(), tz=pytz.UTC)    

    return last_date

def updated_curve_fixed_vert(end_date, vertices):

    if not isinstance(end_date, str):            
        end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    file_path = os.path.abspath("curvas_updated.xlsx")
    #path = "/Users/Dell/OneDrive//Área de Trabalho/User02/AXIOM/Axiom_Feed/curvas_updated.xlsx"

    last_date = pd.to_datetime(get_last_date(file_path)).strftime("%Y-%m-%d")
    dif_days = np.busday_count(last_date, end_date)
    taxas = pd.read_excel(file_path)    
    taxas['Date'] = pd.to_datetime(taxas['Date'])
    taxas.set_index(['Date'], inplace=True)

    if dif_days >= 1:

        last_date_plus_1 = get_last_date(file_path) + datetime.timedelta(days=1)        
        last_date_plus_1_str = last_date_plus_1.strftime('%Y-%m-%d')        

        col_name_pre = []
        for x in range(len(vertices)):
            col_name_pre.append(f'Pre_Jan_{vertices[x][:4][2:]}')
        col_name_pre.insert(0, 'Date')

        col_name_ipca = []
        for x in range(len(vertices)):
            col_name_ipca.append(f'IPCA_Jan_{vertices[x][:4][2:]}')
        col_name_ipca.insert(0, 'Date')

        datelist = pd.date_range(last_date_plus_1_str, end_date, freq="B").shift(0, freq = 'B').strftime('%Y-%m-%d').tolist()

        curva_pre = pd.DataFrame(columns=col_name_pre)
        curva_ipca = pd.DataFrame(columns=col_name_ipca)

        for dat in datelist:            
            dat_ = pd.to_datetime(dat, dayfirst=True).strftime("%Y-%m-%d")
            days_interpol = np.busday_count(dat_, vertices)
            
            try:
                pre=ettj.get_ettj(dat, curva="PRE")
                ipca=ettj.get_ettj(dat, curva="DIC")

                pre_interp = np.interp(days_interpol, np.array(
                    pre['Dias Corridos']), np.array(pre.iloc[:, 1])).tolist()
                
                pre_interp.insert(0, dat_)
                curva_pre.loc[len(curva_pre)] = pre_interp        

                ipca_interp = np.interp(days_interpol, np.array(
                    ipca['Dias Corridos']), np.array(ipca.iloc[:, 1])).tolist()
                
                ipca_interp.insert(0, dat_)
                curva_ipca.loc[len(curva_ipca)] = ipca_interp
            
            except:
                pass

        curvas = pd.merge(curva_pre,curva_ipca,on='Date')        
        curvas.set_index('Date', inplace=True)
        curvas.index = pd.to_datetime(curvas.index)#.strftime("%d-%m-%Y")
        taxas = pd.concat([taxas, curvas])                    
        taxas.to_excel(file_path)
    return taxas

def update_hist_ettj(end_date):
    
    days_interpol = [90, 360, 1800, 3600] #over, 1y, 5y, 10y
    cols_pre=['Date','Pre_Caixa','Pre 1y','Pre 5y', 'Pre 10y']
    cols_ipca=['Date','Ipca 3m','Ipca 1y','Ipca 5y', 'Ipca 10y']
    cols_imp=['Date','Imp 3m','Imp 1y','Imp 5y', 'Imp 10y']

    if not isinstance(end_date, str):            
            end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    path = os.path.abspath("hist_ettj.xlsx")        
    #path = f"{PATH}/hist_ettj.xlsx"
    last_date = pd.to_datetime(get_last_date(path)).strftime("%Y-%m-%d")    
    dif_days = np.busday_count(last_date, end_date)            
    taxas = pd.read_excel(path)    
    taxas['Date'] = pd.to_datetime(taxas['Date'])
    taxas.set_index(['Date'], inplace=True)        

    if dif_days >= 2:
       
        last_date_plus_1 = get_last_date(path) + datetime.timedelta(days= 1)        
        last_date_plus_1_str = last_date_plus_1.strftime('%d/%m/%Y')                
        datas = ettj.listar_dias_uteis(last_date_plus_1_str, end_date)

        curva_pre = pd.DataFrame(columns=cols_pre)
        curva_ipca = pd.DataFrame(columns=cols_ipca)
        curva_implicita = pd.DataFrame(columns=cols_imp)        

        for dat in datas:
            pre=ettj.get_ettj(dat, curva="PRE")
            ipca=ettj.get_ettj(dat, curva="DIC")    
            
            try:
                pre_interp = np.interp(days_interpol, np.array(pre['Dias Corridos']), np.array(pre.iloc[:,1])).tolist()
                pre_interp.insert(0, dat)
                curva_pre.loc[len(curva_pre)] = pre_interp
                ########
                ipca_interp = np.interp(days_interpol, np.array(ipca['Dias Corridos']), np.array(ipca.iloc[:,1])).tolist()
                ipca_interp.insert(0, dat)
                curva_ipca.loc[len(curva_ipca)] = ipca_interp
                ########
                implicita = list(((1+np.array(pre_interp[1:])/100) / (1+np.array(ipca_interp[1:])/100) - 1) * 100)
                implicita.insert(0, dat)
                curva_implicita.loc[len(curva_implicita)] = implicita
                
            except:      

                pre_interp = list(itertools.repeat(0, len(curva_pre.columns) - 1))
                pre_interp.insert(0, dat)
                curva_pre.loc[len(curva_pre)] = pre_interp
                ########
                ipca_interp = list(itertools.repeat(0, len(curva_ipca.columns) - 1))
                ipca_interp.insert(0, dat)
                curva_ipca.loc[len(curva_ipca)] = ipca_interp
                ########
                implicita = list(itertools.repeat(0, len(curva_implicita.columns) - 1))
                implicita.insert(0, dat)
                curva_implicita.loc[len(curva_implicita)] = implicita

        curvas = pd.merge(pd.merge(curva_pre,curva_ipca,on='Date'),curva_implicita,on='Date')        
        #curvas = pd.merge(curva_pre, curva_ipca)
        #curvas = curvas.rename(columns={'Data': 'Date'})        
        curvas['Date'] = pd.to_datetime(curvas['Date'])
        curvas.set_index(['Date'], inplace=True)                                
        taxas = pd.concat([taxas, curvas])                

        #path_hist_ettj_updated = f"{PATH}/hist_ettj_updated.xlsx"        
        taxas.to_excel(path)
    return taxas


def update_hist_pre(end_date):

    days_interpol = [90, 180, 270, 360, 720, 1080, 1800, 2880, 3600]  
    cols_pre = ['Date', '3M', '6M', '9M', '1Y', '2Y', '3Y', '5Y', '8Y', '10Y']
    
    if not isinstance(end_date, str):            
            end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    path = f"{PATH}/hist_pre.xlsx"
    last_date = pd.to_datetime(get_last_date(path)).strftime("%Y-%m-%d")    
    dif_days = np.busday_count(last_date, end_date)
        
    #get last curve hist_ettj (storaged)    
    taxas = pd.read_excel(path)    
    taxas['Date'] = pd.to_datetime(taxas['Date'])
    taxas.set_index(['Date'], inplace=True)        

    if dif_days >= 2:               
        
        last_date_plus_1 = get_last_date(path) + datetime.timedelta(days=1)
        last_date_plus_1_str = last_date_plus_1.strftime('%Y/%m/%d')

        datas = ettj.listar_dias_uteis(last_date_plus_1_str, end_date)
        curva_pre = pd.DataFrame(columns=cols_pre)

        for dat in datas:
            pre = ettj.get_ettj(dat, curva="PRE")    

            try:
                pre_interp = np.interp(days_interpol, np.array(
                pre['Dias Corridos']), np.array(pre.iloc[:, 1])).tolist()
                pre_interp.insert(0, dat)
                curva_pre.loc[len(curva_pre)] = pre_interp
            
            except:
                pre_interp = list(itertools.repeat(
                0, len(curva_pre.columns) - 1))
                pre_interp.insert(0, dat)
                curva_pre.loc[len(curva_pre)] = pre_interp
                    
        curva_pre['Date'] = pd.to_datetime(curva_pre['Date'])
        curva_pre.set_index(['Date'], inplace=True)                
        taxas = pd.concat([taxas, curva_pre])                        
        taxas.to_excel(path)            

    return taxas

def get_curvas_pre(start_date, end_date):

    days_interpol = [90, 180, 270, 360, 720, 1080, 1800, 2880, 3600]  # over, 1y, 5y, 10y
    cols_pre = ['Date', '3M', '6M', '9M', '1Y', '2Y', '3Y', '5Y', '8Y', '10Y']
    
    start_dt_str = start_date.strftime('%d/%m/%Y')
    end_dt_str = end_date.strftime('%d/%m/%Y')
    datas = ettj.listar_dias_uteis(start_dt_str, end_dt_str)
    curva_pre = pd.DataFrame(columns=cols_pre)

    for dat in datas:
        pre = ettj.get_ettj(dat, curva="PRE")    

        try:
            pre_interp = np.interp(days_interpol, np.array(
            pre['Dias Corridos']), np.array(pre.iloc[:, 1])).tolist()
            pre_interp.insert(0, dat)
            curva_pre.loc[len(curva_pre)] = pre_interp
        
        except:
            pre_interp = list(itertools.repeat(
            0, len(curva_pre.columns) - 1))
            pre_interp.insert(0, dat)
            curva_pre.loc[len(curva_pre)] = pre_interp
    
    curva_pre.index = pd.to_datetime(curva_pre.index)
    curva_pre.set_index(['Date'], inplace=True)

    return curva_pre


def plot_curves1(start_date, end_date, interval='W'):
    
    taxas = update_hist_pre(end_date)
    start_date = start_date.strftime('%d/%m/%Y')
    curva = taxas[taxas.index > start_date]  
    curva_adj = curva.resample(interval).last()
    
    fig = go.Figure()

    for i in curva_adj.index:
        fig.add_trace(go.Scatter(x=curva_adj.columns, y=curva_adj.loc[i], mode='lines', name=str(i)))

    fig.show()


def plot_curves2(start_date, end_date, interval='W'):
    
    taxas = update_hist_pre(end_date)
    start_date = start_date.strftime('%d/%m/%Y')
    curva = taxas[taxas.index > start_date]  
    curva_adj = curva.resample(interval).last()
    
    fig = go.Figure()

    for i in curva_adj.index:
        fig.add_trace(
            go.Scatter(
                x=curva_adj.columns,
                y=curva_adj.loc[i], 
                mode='lines', 
                name=str(i),
                visible=False
        )
    )

    fig.data[0].visible = True
    steps = []

    for i in range(len(fig.data)):
        step = dict(
            method='restyle',
            args=['visible', [False] * len(fig.data)],
            label=fig.data[i]['name'][:7])

        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': "Mês: "},
        pad={'t':10},
        steps=steps)]

    fig.update_layout(
        sliders=sliders,
        yaxis = dict(range=[3,14.5]))

    fig.show()





def ret_taxas(end_dt):

    dt = [90, 360, 1800, 3600, 90, 360, 1800, 3600, 90, 360, 1800, 3600,] #over, 1y, 5y, 10y    
    cols = ['Pre_Caixa','Pre 1y','Pre 5y', 'Pre 10y', 'Ipca 3m','Ipca 1y','Ipca 5y', 'Ipca 10y',
            'Imp 3m','Imp 1y','Imp 5y', 'Imp 10y']    

    taxas = update_hist_ettj(end_dt)    
    df_pu = taxas.copy()
    
    for i, ticker in enumerate(cols):
        df_pu[ticker] = 1e5 / np.power(1 + df_pu[ticker]/100, dt[i]/360)     
        
    df_ret_pu = np.log(df_pu).diff(periods = 1).dropna()
    df_ret_m = df_ret_pu.resample('M').agg(lambda x: (1 + x).prod() - 1) 

    start_date_format = pd.to_datetime( df_ret_m.index[0]).strftime("%Y-%m-%d")
    end_date_format = pd.to_datetime(df_ret_m.index[-1]).strftime("%Y-%m-%d")
    T_days = np.busday_count(start_date_format, end_date_format)

    res = pd.DataFrame()
    res['ret_cagr_per'] = df_ret_m.apply(lambda x: (1 + x).prod() - 1)*100
    res['ret_cagr_anual'] = df_ret_m.apply(lambda x: (1 + x).prod() ** (252/T_days) - 1)*100        
    res['res_vol_anual'] = df_ret_pu.std()*np.sqrt(252)*100
    
    return res
    
        
