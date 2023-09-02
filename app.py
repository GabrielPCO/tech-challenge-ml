# Libs

import pandas as pd
import datetime as dt

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
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔷Introdução",
                                              "🌐Base de Dados",
                                              "🔍Análise Exploratória dos Dados",
                                              "📋ARIMA",
                                              "📈XGB", 
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
    
    Neste documento iremos analizar dados históricos do fechamento do índice Ibovespa e criar um modelo preditivo com precisão adequada (acima de 70%) com intuito de evidenciar padrões e tendências futuras.

    Os tópicos foram divididos em quatro categorias principais: base de dados, análise exploratória dos dados, ARIMA, XGB. Cada categoria será tratada e mais aprofundada em sua respectiva aba dentro desse documento.

    
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
    Decidimos então remover a linha, pois o valor nulo contido na coluna "Vol." impedirá a construção adequada do nosso modelo de previsão.
    ```python
    # Removendo a linha com valor nulo
    df_ibovespa = df_ibovespa.drop(df_ibovespa[df_ibovespa['Vol.'].isna()].index)
    ```
    Removemos também a coluna "Var%", pois essa coluna não será interessante para nosso modelo de previsões.
    ```python
    # Removendo coluna Var%
    del df_ibovespa['Var%']
    ```
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
    (4911, 6)
    ```
    E as principais informações dos nossos dados
    ```python
    # Verificando informações do Dataframe
    df_ibovespa.info()
    ```
    ```
    <class 'pandas.core.frame.DataFrame'>
    Index: 4911 entries, 0 to 4911
    Data columns (total 6 columns):
    #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
    0   Data      4911 non-null   object 
    1   Último    4911 non-null   float64
    2   Abertura  4911 non-null   float64
    3   Máxima    4911 non-null   float64
    4   Mínima    4911 non-null   float64
    5   Vol.      4911 non-null   object 
    dtypes: float64(4), object(2)
    memory usage: 268.6+ KB
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
    |       Data |  Último | Abertura |  Máxima |  Mínima |   Vol. |
    | 2023-08-15 | 116.552 |  116.809 | 117.697 | 116.238 | 11,79M |
    | 2023-08-14 | 116.810 |  118.067 | 118.082 | 116.530 | 11,20M |
    | 2023-08-11 | 118.065 |  118.350 | 119.054 | 117.415 | 11,87M |
    | 2023-08-10 | 118.350 |  118.412 | 119.438 | 118.113 | 12,69M |
    | 2023-08-09 | 118.409 |  119.090 | 119.090 | 117.901 | 11,25M |
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
    |       Data | Último | Abertura | Máxima | Mínima |   Vol. |
    | 2023-08-15 | 116552 |   116809 | 117697 | 116238 | 11,79M |
    | 2023-08-14 | 116810 |   118067 | 118082 | 116530 | 11,20M |
    | 2023-08-11 | 118065 |   118350 | 119054 | 117415 | 11,87M |
    | 2023-08-10 | 118350 |   118412 | 119438 | 118113 | 12,69M |
    | 2023-08-09 | 118409 |   119090 | 119090 | 117901 | 11,25M |
    ```
    '''
    st.divider()
    '''

    ## Conversão dos Valores de Volume
    Observamos que o Dtype da coluna "Vol." estão como 'object'.

    Como os dados representam volumes em milhões (M) ou milhares (K) de reais, teremos que fazer a conversão dos dados.
    ```python
    # Transformando a coluna Vol. em numérica
    df_ibovespa["Vol."] = df_ibovespa["Vol."].replace({",":".","K":"*1e3", "M":"*1e6"}, regex=True).map(pd.eval).astype(int)
    df_ibovespa.head()
    ```
    ```
    |       Data | Último | Abertura | Máxima | Mínima |     Vol. |
    | 2023-08-15 | 116552 |   116809 | 117697 | 116238 | 11790000 |
    | 2023-08-14 | 116810 |   118067 | 118082 | 116530 | 11200000 |
    | 2023-08-11 | 118065 |   118350 | 119054 | 117415 | 11870000 |
    | 2023-08-10 | 118350 |   118412 | 119438 | 118113 | 12690000 |
    | 2023-08-09 | 118409 |   119090 | 119090 | 117901 | 11250000 |
    ```
    '''
    st.divider()
    '''

    ## Finalização

    Por fim, indexamos nossos dados pela coluna 'Data' em ordem ascendente e salvamos as modificações do DataFrame para o uso em nosso projeto.
    ```python
    # indexando o DataFrame pela data
    df_ibovespa_indexData = df_ibovespa.set_index(['Data'])

    # Ajustando o DataFrame para os dados ficarem em ordem ascendente quanto a data
    df_ibovespa_indexData = df_ibovespa_indexData.sort_index()

    # Salvando o DataFrame
    df_ibovespa_indexData.to_csv('Assets/DataFrames/ibov_modelo.csv')
    ```
    Agora nossos dados estão prontos para a próxima etapa de análise.

    Na análise, poderemos visualizar melhor as tendências e padrões de nossos dados.
    '''
with tab2:
    '''

    ## Análise exploratória dos dados

    Inicialmente iremos visualizar o fechamento diário do Ibovespa no período entre 15/10/2003 a 15/08/2023
    '''
    graf_1 = load_img('Assets/Graficos/historico.jpg')
    st.image(graf_1)
    '''

    Analisando a série temporal do valor de fechamento diário do IBOVESPA, de maneira geral ficam evidentes seis momentos distintos, marcados por alguns grandes eventos socioeconômicos:

    1. Tendência de aumento do índice entre 2004 e 2008
    2. Crise econômica de 2008, com recuperação em meados de 2010
    3. Tendência de queda de 2010 a 2016
    4. Alta tendência de subida entre 2016 e 2020
    5. Queda abrupta com a pandemia em 2020
    6. Retomada da normalidade a partir do final de 2021, com série variando em momentos de queda e alta

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
    
    ## Volume negociado

    Além do valor de fechamento, analisar variáveis como o Volume negociado pode ser interessante para entendermos o contexto do mercado financeiro brasileiro.

    '''
    graf_volume = load_img('Assets/Graficos/volume1.png')
    st.image(graf_volume)
    '''
    O gráfico ilustra bem a evolução do mercado variável no Brasil. É visível que, até meados de 2019, o volume de negociações sofreu pouca alteração com uma leve tendência de crescimento.

    Com a redução nas taxas de juros e Selic, e consequentemente a baixa nos investimentos de Renda Fixa, o mercado de Renda Variável teve um "boom" a partir do ano de 2020.

    https://www.cnnbrasil.com.br/economia/numero-de-investidores-na-bolsa-cresce-15-em-2022-apostando-na-diversificacao/
    '''

    '''
    ## Volume x Fechamento

    Possivelmente, os valores de fechamento do índice IBOVESPA e volume total negociado no mercado estão positivamente correlacionados,
    tendo em vista que com ações mais valorizadas há mais chance de ocorrorem negociações de compra e venda de ações.
    '''
    graf_vol_fechamento = load_img('Assets/Graficos/volume_fechamento.png')
    st.image(graf_vol_fechamento)
    '''
    Neste gráfico de dispersão é possível visualizar uma forte correlação entre o valor do índice IBOVESPA e o volume negociado em bolsa. 
    Para estas duas variáveis, foi calculada uma correlação de aproximadaente 0.70, um valor bastante alto e que confirma a hipótese incial.
    '''
    st.divider()
    '''
    ## Diferença entre mínimo e máximo

    Uma maneira interessante de verificar comportamentos incomuns nesta série temporal é identificar os dias com maior diferença entre os valores diários mínimos e máximos
    '''
    graf_dif_min_max = load_img('Assets/Graficos/dif_min_max1.png')
    st.image(graf_dif_min_max)
    '''
    **Número de dias com diferença entre mínimo e máximo maior que 5 pontos:**

    ```
    |      |  N dias | 
    |  Ano |         | 
    | 2020 |    19   |
    | 2021 |    3    |
    | 2022 |    2    |
    | 2018 |    2    |
    ```    
    Este resultado reforça o comportamento atípico do IBOVESPA a partir de 2020, por conta do contexto da pandemia e aquecimento do mercado de renda variável. 
    
    Possivelmente, 2021 e 2022 aparecem em seguida no ranking também por reflexo dos efeitos da crise causada pela pandemia.
    '''
    st.divider()
    '''
    ## Componentes da série temporal original

    Para entender mais a fundo comportamento da variável target (Fechamento) ao longo do tempo, é uma opção visualizar os diferentes componentes da série temporal

    '''
    graf_serie_componentes = load_img('Assets/Graficos/serie_temporal_componentes.png')
    st.image(graf_serie_componentes)
    '''
    Não foi possível extrair insights muito valiosos com a decomposição da série temporal em seus componentes. A tendência representa a mesma curva da própria série porém um pouco mais suavizada.

    A sazonalidade têm padrão caótico, indicando que não é sazonalidade aparente nos dados, o que faz bastante sentido se tratando do mercado de ações.

    Já o resíduo reforça a ideia de 2020 ser um ano fora do padrão de comportamento da curva.
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
    
    ## Média móvel & desvio padrão

    Em seguida, traçamos as retas da média móvel e do desvio padrão para entender melhor o comportamento da nossa série.
    A média móvel é um estimador calculado a partir de amostras sequenciais, podendo indicar tendências em um determinado período.
    Já o desvio padrão expressará o grau de dispersão do nosso conjunto de dados.
    '''
    graf_4 = load_img('Assets/Graficos/mm_std.jpg')
    st.image(graf_4)
    '''

    Pelo gráfico, observamos uma certa tendência de ascensão dos pontos de fechamento ao longo do histórico dos dados.

    A seguir, iniciaremos a construção dos nossos modelos.
    Utilizaremos dois métodos diferentes para uma melhor análise do fechamento do Ibovespa, visto que é um tipo de dado sensível, volátil e sem a presença de sazonalidade.
    Inicialmente, vamos fazer a análise utilizando o algoritmo ARIMA retroalimentado. 
    Em seguida, realizaremos uma outra análise aplicando o Extreme Gradient Boosting Regressor em nossos dados.
    '''
with tab3:
    '''

    ## ARIMA

    Em Construção...
    '''
with tab4:
    '''

    ## Modelo XGBRegressor

    Decidimos utilizar também o método **Extreme Gradient Boosting Regressor** para o nosso modelo de previsões.

    Esse método em questão compõe um conjunto de classes de algoritmos de aprendizado de máquina que podem ser usados para problemas de classificação ou modelagem preditiva de regressão.

    Os conjuntos são construídos a partir de modelos de árvore de decisão. As árvores são adicionadas uma de cada vez ao conjunto e ajustadas para corrigir os erros de previsão cometidos pelos modelos anteriores. Este é um tipo de modelo de aprendizado de máquina conjunto conhecido como boosting.

    Os modelos são ajustados usando qualquer função de perda diferenciável arbitrária e algoritmo de otimização de gradiente descendente. Isso dá à técnica o nome de “aumento de gradiente” (Gradient boosting), pois o gradiente de perda é minimizado à medida que o modelo se ajusta, como uma rede neural.

    Por esse motivo, o método se torna resistente ao "overfitting" (sobreajuste), ou seja, quando um modelo estatístico se ajusta muito bem ao conjunto de dados anteriormente observado.

    Desse modo, julgamos o método como sendo de extrema utilidade para a previsão de dados sensíveis como os financeiros.
    '''
    st.divider()
    '''
    
    ## Separando treino e teste

    Agora vamos desenvolver um modelo de previsão e treiná-lo. Para isso, vamos visualizar os dados dividindo-os em conjuntos de treinamento e teste.
    
    Por se tratar de dados sensíveis e de maior volatilidade, decidimos dividir treinamento e teste em aproximadamente 85% e 15% dos dados, respectivamente.
    ```python
    treino = df_ibovespa_indexData.iloc[:int(.85*len(df_ibovespa_indexData)), :]
    teste = df_ibovespa_indexData.iloc[int(.85*len(df_ibovespa_indexData)):, :]
    ```
    '''
    graf_5 = load_img('Assets/Graficos/treino_teste.jpg')
    st.image(graf_5)
    '''

    Inicialmente, selecionamos as características e o target para o nosso modelo
    ```python
    # Definindo as variáveis de características e target do modelo
    caracteristicas = ["Abertura","Máxima","Mínima","Vol."]
    target = 'Último'
    ```

    Em seguida, criamos nosso modelo de regressão
    ```python
    # Criando e treinando o modelo
    modelo = xgb.XGBRegressor()
    modelo.fit(treino[caracteristicas], treino[target])
    ```
    ```

    |                                 XGBRegressor                                |
    |XGBRegressor(base_score=None, booster=None, callbacks=None,                  |
    |            colsample_bylevel=None, colsample_bynode=None,                   |
    |            colsample_bytree=None, early_stopping_rounds=None,               |
    |            enable_categorical=False, eval_metric=None, feature_types=None,  |
    |            gamma=None, gpu_id=None, grow_policy=None, importance_type=None, |
    |            interaction_constraints=None, learning_rate=None, max_bin=None,  |
    |            max_cat_threshold=None, max_cat_to_onehot=None,                  |
    |            max_delta_step=None, max_depth=None, max_leaves=None,            |
    |            min_child_weight=None, missing=nan, monotone_constraints=None,   |
    |            n_estimators=100, n_jobs=None, num_parallel_tree=None,           |
    |            predictor=None, random_state=None, ...)                          |
    
    ```
    '''
    st.divider()
    '''

    ## Previsão

    Após o ajuste dos dados, iremos realizar a previsão do fechamento do Ibovespa utilizando o modelo que criamos.
    Utilizaremos a função 'predict' para a previsão dos dados.
    ```python
    # Criando e mostrando a previsão nos dados de teste
    previsoes = modelo.predict(teste[caracteristicas])
    print('Previsões do modelo:')
    print(previsoes)
    ```
    ```
    Previsões do modelo:
    [101723.625  99988.08  101242.07  102306.07  100162.05   99860.3
    100003.68  101212.35   99553.875  99227.74   99597.98  100655.06
    100655.06   99391.75   99090.125  96282.25   97211.03   96396.99
    97164.875  96320.805  95931.52   94704.586  95143.375  94796.336
    94435.83   95248.26   96616.125  95628.12   96905.66   98047.18
    98579.79   99160.62   99147.95   99173.93   99241.91   99760.98
    100809.01  101833.55  101419.4   101454.92  101185.57   95993.21
    94211.61   93897.37   95401.7    97421.086 100054.984 100759.914
    102089.37  104115.61  105074.734 102198.45  104267.44  105048.1
    106887.33  106963.914 106414.35  106297.53  107070.07  108213.17
    111577.58  111736.6   111792.47  110379.016 111262.78  111814.625
    111518.39  113224.95  114118.43  114273.195 112681.02  113579.89
    115199.22  116040.56  116099.04  116099.82  119113.28  119247.695
    115439.47  117023.43  117231.625 119247.695 118407.3   118407.3
    118322.836 118110.74  118426.15  118426.15  118469.28  118407.3
    118407.3   118376.945 118407.3   118407.3   118407.3   118407.3
    118407.3   118407.3   116744.625 118521.41  115075.02  118918.5
    116835.55  116217.1   119266.55  119247.695 118407.3   118407.3
    118407.3   118407.3   118407.3   119086.88  118407.3   118407.3
    118335.125 118376.945 113315.69  112869.195 115358.1   112862.305
    110065.58  111281.44  107258.14  107449.805 111984.055 112843.414
    111547.08  111193.82  110223.86  113579.89  114218.875 114458.12
    114218.875 115184.54  114402.98  115373.07  114706.17  115278.805
    114118.43  112091.92  114576.41  115174.29  116143.78  116358.836
    ...
    118407.3   117368.91  118374.266 117214.14  117264.33  119484.984
    118376.945 118513.766 118407.3   118335.125 118407.3   118376.945
    118376.945 118376.945 118433.76  118459.54  118407.3   118376.945
    118511.78  119266.55  119132.13  117451.43  116468.82 ]
    ```

    Em seguida, vamos verificar a acurácia do nosso modelo.
    ```python
    # Mostrando a acurácia do modelo
    acuracia = modelo.score(teste[caracteristicas], teste[target])
    print('Acurácia:')
    print(acuracia)
    ```
    ```
    Acurácia:
    0.8809057960319034
    ```
    Com um score de 0,8809057960319034 o modelo tem aproximadamente 88% de acurácia para prever as próximas observações de fechamento do mercado.
    '''
    st.divider()
    '''
    Vamos então, plotar o gráfico das previsões, dos valores de treino e de teste para um melhor entendimento do nosso resultado.
    ```python
    # Plotando as previsões e o preço no fechamento
    plt.figure(figsize=(10,6))
    plt.plot(treino['Último'], color='green', label='Treino')
    plt.plot(teste['Último'], color = 'blue', label='Teste')
    plt.plot(teste[target].index, previsoes, color = 'orange',label='Previsão')
    plt.title('Previsão do modelo')
    plt.xlabel('Data')
    plt.ylabel('Preço no fechamento')
    plt.legend()
    plt.grid()
    plt.show()
    '''
    graf_6 = load_img('Assets/Graficos/previsao_target.jpg')
    st.image(graf_6)
    st.divider()
    '''

    ## Conclusão:

    Após organizar, modificar e ajustar os dados através do modelo de previsão Extreme Gradient Boosting Regressor, fomos capazes de prever os dados de teste com uma acurácia adequada (acima de 70%).
    
    Provavelmente, o modelo em questão seria capaz de prever alguma das próximas observações mantendo uma boa precisão em relação aos dados futuros.
    '''
with tab5:
    '''

    ## Referências

    1. DHADUK, Hardikkumar. Stock market forecasting using Time Series analysis With ARIMA model. Analytics Vidhya, 2021. Disponível em: https://www.analyticsvidhya.com/blog/2021/07/stock-market-forecasting-using-time-series-analysis-with-arima-model/. Acesso em: 15, agosto de 2023.

    2. ORDORICA, David. Forecasting Time Series with Auto-Arima. All Data Science, 2021. Disponível em: https://www.alldatascience.com/time-series/forecasting-time-series-with-auto-arima/. Acessado em: 15, agosto de 2023.

    3. SMITH, Taylor G. Forecasting the stock market with pmdarima. alkaline-ml, 2019. Disponível em: https://alkaline-ml.com/2019-12-18-pmdarima-1-5-2/. Acessado em: 15, agosto de 2023.

    4. Índice Bovespa (Ibovespa B3). B3, 2023. Disponível em: https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/ibovespa.htm. Acessado em: 15, agosto de 2023.

    5. Dados Históricos - Ibovespa. Investing.com, 2023. Disponível em: https://br.investing.com/indices/bovespa-historical-data. Acessado em: 15, agosto de 2023.

    6. Random forest. In: WIKIPÉDIA: a enciclopédia livre. [São Francisco, CA: Fundação Wikimedia], 2023. Disponível em: https://en.wikipedia.org/wiki/Random_forest. Acessado em: 15, agosto de 2023.

    7. Sobreajuste. In: WIKIPÉDIA: a enciclopédia livre. [São Francisco, CA: Fundação Wikimedia], 2023. Disponível em: https://pt.wikipedia.org/wiki/Sobreajuste. Acessado em: 15, agosto de 2023.

    8. PARUCHURI, Vik. Predict The Stock Market With Machine Learning And Python. YouTube, 2022. Disponível em: https://www.youtube.com/watch?v=1O_BenficgE. Acessado em: 15, agosto de 2023.

    9. BROWNLEE, Jason. XGBoost for Regression, 2021. Disponível em: https://machinelearningmastery.com/xgboost-for-regression/. Acessado em: 15, agosto de 2023.
    '''

