# Libs

import pandas as pd

# libs gráficas
import matplotlib.pyplot as plt

# Streamlit
import streamlit as st

# Configurando a página
st.set_page_config(
    page_title="Tech-Challenge",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Projeto criado para o *tech-challenge* do curso de pós-graduação da FIAP/Alura."
    }
)

# Função para a leitura da base de dados
@st.cache_data
def read_csv_file(file):
    return pd.read_csv(file)

df_ibovespa = read_csv_file('Assets/DataFrames/ibov.csv')

# Titulo de Página
st.title('Análise de dados: explorando dados do histórico de fechamento do índice Ibovespa (BVSP)')

# Código para alinhar imagens expandidas no centro da tela e justificar textos
st.markdown(
    """
    <style>
        body {text-align: justify}
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

# Carregamento de imagens por cach
@st.cache_data
def load_img(img):
    return plt.imread(img)

# Layout do aplicativo
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔷Introdução",
                                                    "🌐Base de Dados",
                                                    "🔍Visualização",
                                                    "📝ADF", 
                                                    "📊ARIMA",
                                                    "📈Previsão",
                                                    "📑Referências"])

# Separando as Tabs
with tab0:
    '''
    ## Explorando dados do histórico de fechamento do índice Ibovespa

    Links importantes:

    [b3.com.br](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/ibovespa.htm) - Dados de fechamento do índice Bovespa

    [investing.com](https://br.investing.com/indices/bovespa-historical-data) - Base de dados Ibovespa

    Links dos integrantes do projeto:

    [github.com/GabrielPCO](https://github.com/GabrielPCO/tech-challenge-ml) - Github Gabriel Oliveira

    [github.com/jackson-simionato](https://github.com/jackson-simionato) - Github Jackson Simionato

    gabrielpcoliveira@gmail.com - Email Gabriel Oliveira

    simionato.jackson@gmail.com - Email Jackson Simionato

    haendelf@hotmail.com - Email Haendel Oliveira

    '''
    st.divider()
    '''
    
    ## Resumo

    O Ibovespa é o principal indicador de desempenho das ações negociadas na B3 e reúne as empresas mais importantes do mercado de capitais brasileiro. 
    
    Ele foi criado em 1968 e, ao longo desses 50 anos, consolidou-se como referência para investidores ao redor do mundo.

    Reavaliado a cada quatro meses, o índice é resultado de uma carteira teórica de ativos. 
    
    Composto pelas ações e units de companhias listadas na B3 que atendem aos critérios descritos na sua metodologia, correspondendo a cerca de 80% do número de negócios e do volume financeiro do nosso mercado de capitais.
    
    Neste documento iremos analizar dados históricos do fechamento do índice Ibovespa e criar um modelo preditivo com intuito de evidenciar padrões e tendências futuras dentro de um intervalo apropriado de confiança.

    Os tópicos foram divididos em cinco categorias: base de dados, visualização, ADF, ARIMA e previsão. Cada categoria será tratada e mais aprofundada em sua respectiva aba dentro desse documento.

    
    A seguir, disponibilizamos os dados utilizados para a análise no momento da publicação deste documento.

    '''
    st.divider()
    '''

    #### DataFrame dos dados do histórico de fechamento do Ibovespa entre os anos de 2003 a 2023
    '''

    # Função do botão de Download para converter o DataFrame em .csv
    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    # Convertendo o DataFrame em .csv
    csv = convert_df(df_ibovespa)

    # Adicionando o DataFrame
    @st.cache_data
    def create_df(df):
        return pd.DataFrame(df)
    
    dataframe = create_df(df_ibovespa)
    dataframe = dataframe.set_index(['Data'])
    st.dataframe(dataframe, use_container_width=True)

    # Botão de Download do DataFrame
    st.download_button(
        label="Download do CSV",
        data=csv,
        file_name='df_ibovespa.csv',
        mime='text/csv',
    )

    st.divider()
    '''

    ## Observação

    Os demais dados, DataFrames e outras análises mais aprofundadas podem ser encontradas na página de Github dos integrantes do grupo referenciadas no início desse documento.
    '''

with tab1:
    '''

    ## Coleta e Manipulação dos dados

    Inicialmente, realizamos o carregamento dos dados utilizados na análise.

    Esses dados contem o histórico de fechamento do índice Ibovespa durante o período de 15/10/2003 a 15/08/2023.

    Os dados foram obtidos do site da investing.com que é uma plataforma e site de notícias sobre o mercado financeiro.
    ```python
    # Carregando o DataFrame com os dados da base
    df_ibovespa = pd.read_csv('Assets/Base/ibovespa.csv', sep=',')
    ```

    '''
    st.divider()
    '''

    ## Dados Nulos

    Foi então verificado a presença de dados nulos que poderiam comprometer nossa análise
    ```python
    # Verificando valores nulos no DataFrame
    df_ibovespa.isnull().sum()
    ```
    Encontramos um dado nulo na coluna 'Volume' em nossos dados
    ```
    Data        0
    Último      0
    Abertura    0
    Máxima      0
    Mínima      0
    Vol.        1
    Var%        0
    dtype: int64
    ```
    ```python
    # Encontrando o dado nulo na linha correspondente
    df_ibovespa[df_ibovespa['Vol.'].isna()]
    ```
    ```
    |       Data | Último | Abertura | Máxima | Mínima | Vol. |   Var% |
    | 10.02.2016 | 40.377 |   40.592 | 40.592 |  39.96 |  NaN | -0,53% |
    ```
    Decidimos não alterar o valor NaN nessa ocasião, pois essa alteração não afetará nossa previsão, pois utilizaremos apenas os dados de fechamento de mercado no estudo.
    '''
    st.divider()
    '''

    ## Dados Duplicados

    Foi feita a verificação de valores duplicados nos dados
    ```python
    # Verificando dados duplicados
    df_ibovespa.duplicated().sum()
    ```
    ```
    0
    ```
    Nenhum dado duplicado foi encontrado, o que significa que nossos dados estão íntegros.
    '''
    st.divider()
    '''

    ## Informações dos Dados

    Verificamos o shape
    ```python
    # Verificando o shape do DataFrame
    df_ibovespa.shape
    ```
    ```
    (4912, 7)
    ```
    E as principais informações dos nossos dados
    ```python
    # Verificando informações do Dataframe
    df_ibovespa.info()
    ```
    ```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4912 entries, 0 to 4911
    Data columns (total 7 columns):
    #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
    0   Data      4912 non-null   object 
    1   Último    4912 non-null   float64
    2   Abertura  4912 non-null   float64
    3   Máxima    4912 non-null   float64
    4   Mínima    4912 non-null   float64
    5   Vol.      4911 non-null   object 
    6   Var%      4912 non-null   object 
    dtypes: float64(4), object(3)
    memory usage: 268.8+ KB
    ```
    '''
    st.divider()
    '''

    ## Conversão Datetime

    Observamos que o Dtype da coluna 'Data' está como 'object'.

    Como a coluna 'Data' possui os dados do período de funcionamento de mercado temos que transformar seu Dtype de 'object' para 'datetime'
    ```python
    # Convertendo coluna Data de object para datetime
    df_ibovespa['Data'] = pd.to_datetime(df_ibovespa['Data'],format='%d.%m.%Y')
    df_ibovespa.head()
    ```
    ```
    |       Data |  Último | Abertura |  Máxima |  Mínima |   Vol. |   Var% |
    | 2023-08-15 | 116.552 | 116.809  | 117.697 | 116.238 | 11,79M | -0,22% |
    | 2023-08-14 | 116.810 | 118.067  | 118.082 | 116.530 | 11,20M | -1,06% |
    | 2023-08-11 | 118.065 | 118.350  | 119.054 | 117.415 | 11,87M | -0,24% |
    | 2023-08-10 | 118.350 | 118.412  | 119.438 | 118.113 | 12,69M | -0,05% |
    | 2023-08-09 | 118.409 | 119.090  | 119.090 | 117.901 | 11,25M | -0,57% |
    ```
    '''
    st.divider()
    '''

    ## Conversão Inteiros

    Observamos que o Dtype das colunas numericas estão como 'float64'.

    Como os dados representam pontos de merdado e devem ser um número inteiro, temos que fazer a conversão.
    ```python
    # Transformando as colunas que estão como float para int
    df_ibovespa['Último'] = df_ibovespa['Último'] * 1000
    df_ibovespa['Último'] = df_ibovespa['Último'].astype(int)
    df_ibovespa['Abertura'] = df_ibovespa['Abertura'] * 1000
    df_ibovespa['Abertura'] = df_ibovespa['Abertura'].astype(int)
    df_ibovespa['Máxima'] = df_ibovespa['Máxima'] * 1000
    df_ibovespa['Máxima'] = df_ibovespa['Máxima'].astype(int)
    df_ibovespa['Mínima'] = df_ibovespa['Mínima'] * 1000
    df_ibovespa['Mínima'] = df_ibovespa['Mínima'].astype(int)
    df_ibovespa.head()
    ```
    ```
    |       Data | Último | Abertura | Máxima | Mínima |   Vol. |   Var% |
    | 2023-08-15 | 116552 | 116809   | 117697 | 116238 | 11,79M | -0,22% |
    | 2023-08-14 | 116810 | 118067   | 118082 | 116530 | 11,20M | -1,06% |
    | 2023-08-11 | 118065 | 118350   | 119054 | 117415 | 11,87M | -0,24% |
    | 2023-08-10 | 118350 | 118412   | 119438 | 118113 | 12,69M | -0,05% |
    | 2023-08-09 | 118409 | 119090   | 119090 | 117901 | 11,25M | -0,57% |
    ```
    '''
    st.divider()
    '''

    ## Finalização

    Por fim, indexamos nossos dados pela coluna 'Data' e salvamos as modificações para o uso em nosso projeto.
    ```python
    # indexando o DataFrame pela data
    df_ibovespa_indexData = df_ibovespa.set_index(['Data'])

    # Salvando o DataFrame
    df_ibovespa_indexData.to_csv('Assets/DataFrames/ibov.csv')
    ```
    Agora nossos dados estão prontos para a próxima etapa de visualização.

    Na visualização, poderemos analizar melhor as tendências e padrões de nossos dados
    '''

with tab2:
    '''

    ## Visualização dos Dados

    Inicialmente iremos visualizar o fechamento diário do Ibovespa no período entre 15/10/2003 a 15/08/2023
    '''
    graf_1 = load_img('Assets/Graficos/historico.jpg')
    st.image(graf_1)
    '''

    Em um primeiro momento, parece que o gráfico apresenta uma tendencia de subida ao longo dos anos.

    Também podemos notar que a maioria dos dados se concentra na região entre os 40.000 a 80.000 pontos, porém precisamos de mais análises gráficas para poder confirmar essa nossa hipótese.
    '''
    st.divider()
    '''

    ## Densidade

    Podemos então analisar a distribuição do nosso dataset através de um gráfico de densidade
    '''
    graf_2 = load_img('Assets/Graficos/densidade.jpg')
    st.image(graf_2)
    '''

    Como suspeitamos, o gráfico de densidade indica uma concentração maior em torno dos 50.000 pontos.

    Isso nos indica que dentro dos nossos dados, durante a maior parte do tempo, o índice flutuou próximo desse valor.
    '''
    st.divider()
    '''

    ## Transformação Logaritmica

    Para uma melhor visualização dos nossos dados, iremos realizar a transformação logaritmica da nossa série temporal.
    ```python
    # Transformação logarítmica da série temporal
    df_ibovespa_indexData_log = np.log(df_ibovespa_indexData['Último'])
    ```

    '''
    graf_3 = load_img('Assets/Graficos/log.jpg')
    st.image(graf_3)
    st.divider()
    '''
    
    ## Média Móvel & Desvio Padrão

    Em seguida, traçamos as retas da média móvel e do desvio padrão para entender melhor o comportamento da nossa série.
    A média móvel é um estimador calculado a partir de amostras sequenciais, podendo indicar tendências em um determinado período.
    Já o desvio padrão expressará o grau de dispersão do nosso conjunto de dados.
    '''
    graf_4 = load_img('Assets/Graficos/mm_std.jpg')
    st.image(graf_4)
    '''

    Pelo gráfico, observamos uma certa tendência de ascensão dos pontos de fechamento ao longo do histórico dos dados.
    Para determinarmos com maior certeza essa hipótese, usaremos a seguir o teste Dickey-Fuller aumentado (ADF) para verificar se a série é ou não estacionária.
    '''
with tab3:
    '''

    ## Teste Dickey-Fuller Aumentado (ADF)

    Utilizaremos o teste dickey-fuller aumentado em nossa serie temporal, pois ele é um dos testes mais comumente utilizados na estatistica para determinar a estacionaridade em séries complexas.
    
    Traremos as seguintes hipóteses para sobre a nossa série:

    ```
    Hipótese nula - A série tem uma raíz unitária

    Hipótese Alternativa - A série não tem uma raíz unitária
    ```

    Se a hipótese nula não for rejeitada, a série será considerada não-estacionária.
    A série se torna estacionária caso a média e o desvio padrão forem linhas retas (média constante e variância constante)
    '''
    st.divider()
    '''

    ## Aplicando o Teste

    Realizamos então o teste para análise dos resultados

    ```python
    # Performando teste aumentado de Dickey-Fuller para verificar se a nossa série temporal (df_ibovespa_indexData_log) é estacionária
    print('Resultados do teste Dickey Fuller:')
    dftest = adfuller(df_ibovespa_indexData_log, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Estatística de Teste', 'p-valor', 'Lags utilizados', 'Número de observações utilizadas'])
    for key, value in dftest[4].items():
        dfoutput['Valor crítico (%s)'%key] = value

    print(dfoutput)
    ```
    ```
    Resultados do teste Dickey Fuller:
    Estatística de Teste                  -0.178197
    p-valor                                0.941071
    Lags utilizados                        6.000000
    Número de observações utilizadas    4905.000000
    Valor crítico (1%)                    -3.431684
    Valor crítico (5%)                    -2.862129
    Valor crítico (10%)                   -2.567084
    dtype: float64
    ```

    Pelos resultados do teste de Dickey-Fuller não podemos rejeitar a hipótese nula, pois o p-valor é maior que 0.05.

    Além disso, a estatística de teste excede os valores críticos. Sendo assim, os dados são considerados não lineares.
    '''
    st.divider()
    '''
    
    ## Isolando Sazonalidade e Tendência

    A sazonalidade e a tendência precisam ser separadas de nossa série antes que possamos realizar uma análise mais aprofundada.
    '''
    graf_5 = load_img('Assets/Graficos/seasonal.jpg')
    st.image(graf_5)
    st.divider()
    '''
    
    ## Reduzindo a Magnitude

    Para reduzir a magnitude dos valores e a tendência crescente da série, primeiro fazemos um logaritmo da série. Em seguida, calculamos a média móvel da série após obter o logaritmo da série.
    '''
    graf_6 = load_img('Assets/Graficos/mm_std_log.jpg')
    st.image(graf_6)
    st.divider()
    '''
    
    ## Separando Treino e Teste

    Agora vamos desenvolver um modelo ARIMA e treiná-lo usando o preço de fechamento da ação a partir dos dados do treino. Então, vamos visualizar os dados dividindo-os em conjuntos de treinamento e teste.
    
    Decidimos dividir treinamento e teste em 80% e 20% dos dados, respectivamente.
    ```python
    train_len = int(df_ibovespa_indexData_log.shape[0]*0.8)
    train_data, test_data = train_test_split(df_ibovespa_indexData_log.sort_index(),train_size=train_len)
    ```
    '''
    graf_7 = load_img('Assets/Graficos/treino_teste.jpg')
    st.image(graf_7)
    '''
    
    A seguir, iniciaremos a construção do nosso modelo ARIMA com os dados de treino selecionados.
    '''
with tab4:
    '''

    ## Modelo ARIMA

    Decidimos utilizar a função Auto ARIMA para podermos testar e descobrir a ordem mais otimizada para o Modelo ARIMA.

    Ela retorna um modelo ARIMA ajustado após determinar os parâmetros mais ideais de p, q e d.

    ```python
        arima_fit = auto_arima(train_data.sort_index(), start_p=0, start_q=0,
                        test='adf',       # usa o teste adf para achar o 'd' otimizado
                        max_p=5, max_q=5, # máximo p e q
                        m=1,              # frequência da série
                        d=None,           # deixa o modelo decidir o 'd'
                        seasonal=False,   # Sem sazonalidade
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
    print(arima_fit.summary())
    arima_fit.plot_diagnostics(figsize=(15,8))
    plt.show()
    ```
    ```
    Performing stepwise search to minimize aic
    ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-20868.132, Time=0.14 sec
    ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-20866.387, Time=0.16 sec
    ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-20866.399, Time=0.32 sec
    ARIMA(0,1,0)(0,0,0)[0]             : AIC=-20867.508, Time=0.08 sec
    ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-20864.132, Time=0.39 sec

    Best model:  ARIMA(0,1,0)(0,0,0)[0] intercept
    Total fit time: 1.094 seconds
                                SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3929
    Model:               SARIMAX(0, 1, 0)   Log Likelihood               10436.066
    Date:                Tue, 15 Aug 2023   AIC                         -20868.132
    Time:                        16:51:36   BIC                         -20855.581
    Sample:                             0   HQIC                        -20863.679
                                - 3929                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept      0.0004      0.000      1.620      0.105   -9.22e-05       0.001
    sigma2         0.0003   3.46e-06     83.415      0.000       0.000       0.000
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.25   Jarque-Bera (JB):              4246.78
    ...
    ===================================================================================
    ```
    '''
    graf_8 = load_img('Assets/Graficos/auto_arima.jpg')
    st.image(graf_8)
    st.divider()
    '''

    ## Resultados

    Observamos que como resultado o modelo Auto ARIMA atribuiu os valores 0, 1 e 0 para p, d e q, respectivamente.

    A seguir iremos utilizar o modelo para a previsão dos valores de fechamento do Ibovespa.
    '''
with tab5:
    '''

    ## Previsão

    Após o ajuste dos dados, iremos realizar a previsão dos valores de fechamento do Ibovespa utilizando o modelo que criamos.
    Utilizaremos a função 'predict' do statsmodel para a previsão dos dados.
    ```python
    # Fazendo a previsão com os dados ajustados
    prediction, confint = arima_fit.predict(len(test_data), return_conf_int=True,alpha=0.05)
    ```

    Em seguida, vamos plotar o gráfico com a previsão e os intervalos de confiança.
    ```python
    # Previsão
    # Transformando em Series Pandas
    fc_series = pd.Series(prediction.values, index=test_data.sort_index().index)
    lower_series = pd.Series(confint[:, 0], index=test_data.sort_index().index)
    upper_series = pd.Series(confint[:, 1], index=test_data.sort_index().index)

    # Plotando
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(train_data, label='Treino')
    plt.plot(test_data, color = 'blue', label='Original')
    plt.plot(fc_series, color = 'orange',label='Previsão')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                    color='k', alpha=.15)
    plt.title('Previsão do Fechamento do Ibovespa')
    plt.xlabel('Data')
    plt.ylabel('Fechamento')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid()
    plt.show()
    ```
    '''
    graf_9 = load_img('Assets/Graficos/previsao.jpg')
    st.image(graf_9)
    st.divider()
    '''

    ## Relatório de Performance

    ```python
    # Relatório de performance
    mse = mean_squared_error(test_data.values, prediction.values)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test_data.values, prediction.values)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(test_data.values, prediction.values))
    print('RMSE: '+str(rmse))
    mape = np.mean(np.abs(prediction.values - test_data.values)/np.abs(test_data.values))
    print('MAPE: '+str(mape))
    ```
    ```
    MSE: 0.040290750459765705
    MAE: 0.16540199812023165
    RMSE: 0.20072556005592737
    MAPE: 0.014319964423499875
    ```
    Com um MAPE de aproximadamente 1,4% o modelo tem 98,6% de acurácia para prever as próximas observações dentro do intervalo de confiânça
    '''
    st.divider()
    '''

    ## Conclusão:

    Após organizar, modificar e ajustar os dados de fechamento através do modelo de previsão ARIMA, fomos capazes de prever os dados de teste com uma alta taxa de confiança.
    
    Provavelmente, o modelo em questão seria capaz de prever alguma das próximas observações com acurácia, dentro dos limites de confiânça.
    '''
with tab6:
    '''

    ## Referências

    1. DHADUK, Hardikkumar. Stock market forecasting using Time Series analysis With ARIMA model. Analytics Vidhya, 2021. Disponível em: https://www.analyticsvidhya.com/blog/2021/07/stock-market-forecasting-using-time-series-analysis-with-arima-model/. Acesso em: 15, agosto de 2023.

    2. ORDORICA, David. Forecasting Time Series with Auto-Arima. All Data Science, 2021. Disponível em: https://www.alldatascience.com/time-series/forecasting-time-series-with-auto-arima/. Acessado em: 15, agosto de 2023.

    3. SMITH, Taylor G. Forecasting the stock market with pmdarima. alkaline-ml, 2019. Disponível em: https://alkaline-ml.com/2019-12-18-pmdarima-1-5-2/. Acessado em: 15, agosto de 2023.

    4. Índice Bovespa (Ibovespa B3). B3, 2023. Disponível em: https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/ibovespa.htm. Acessado em: 15, agosto de 2023.

    5. Dados Históricos - Ibovespa. Investing.com, 2023. Disponível em: https://br.investing.com/indices/bovespa-historical-data. Acessado em: Acessado em: 15, agosto de 2023.
    '''

