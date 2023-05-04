import numpy as np
import pandas as pd


def get_titulos_tesouro():
    url = 'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv'
    df = pd.read_csv(url, sep=';', decimal=',')
    df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], dayfirst=True)
    df['Data Base'] = pd.to_datetime(df['Data Base'], dayfirst=True)
    multi_indice = pd.MultiIndex.from_frame(df.iloc[:,:3])
    df = df.set_index(multi_indice).iloc[:,3:]
    return df


def get_hist_titulo(start_dt, vcto, type_name, type_price):
    
    titulos = get_titulos_tesouro()
    titulos.sort_index(inplace=True)
    vctos = titulos.loc[(type_name)].reset_index()['Data Vencimento'].unique()
    
    df = titulos.loc[(type_name, vcto)][type_price]    
    df = df[df.index > np.datetime64(start_dt.to_pydatetime())]
    df_ret = np.log(df).diff(periods = 1)
    
    return df, df_ret, vctos


def get_taxas_ntnb(start_dt, vctos, type_name, type_price):
    
    titulos = get_titulos_tesouro()
    titulos.sort_index(inplace=True)
        
    ipca = pd.DataFrame()        
    for vcto in vctos:
            
        ipca[vcto] = titulos.loc[(type_name, vcto)][type_price]        
        ipca = ipca[ipca.index > np.datetime64(start_dt.to_pydatetime())]
    
    return ipca
    


def ret_acum_per(df_ret, data_base):

    df_ret_m = df_ret.resample('M').agg(lambda x: (1 + x).prod() - 1)
    df_ret_m = df_ret_m[df_ret_m.index < data_base]      
        
    retorno_cumulativo_carteira = (np.cumprod(np.array(df_ret_m)+1)-1 )*100
    return retorno_cumulativo_carteira[-1]


def ret_acum_ano(df_ret, data_base):

    df_ret_m = df_ret.resample('M').agg(lambda x: (1 + x).prod() - 1)
    df_ret_m = df_ret_m[df_ret_m.index < data_base]      
    
    start_date_format = pd.to_datetime( df_ret_m.index[0]).strftime("%Y-%m-%d")
    end_date_format = pd.to_datetime(df_ret_m.index[-1]).strftime("%Y-%m-%d")
    T_days = np.busday_count(start_date_format, end_date_format)

    retorno_cumulativo_carteira = (np.cumprod(np.array(df_ret_m)+1)-1 )  
    ret_acum_ano = ((1 + retorno_cumulativo_carteira[-1]) ** (252/T_days) - 1)*100
        
    return ret_acum_ano