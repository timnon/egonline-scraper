from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

import datetime
from datetime import timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
import plotly.express as px


############################################################################# %%
## Scrape data
############################################################################# %%

driver = webdriver.Chrome('/usr/local/bin/chromedriver')
driver.get("https://web.egonline.ch/customer/login")

def get_data(tab='heat_tab',data_ix=0,year=2021):
    element = driver.find_element_by_id(tab)
    element.click()
    time.sleep(1)

    s = None
    for month in range(1,13):
        # data
        script = f"$('#monthpicker').val('{month}.{year}');"
        driver.execute_script(script)
        script = "set_statistic('1837','month','');"
        driver.execute_script(script)
        time.sleep(1)
        script = f"return window.myChart.data.datasets[{data_ix}].data"
        v = driver.execute_script(script)

        # dates
        start_date = datetime.date(year,month,1)
        end_date = start_date+relativedelta(months=1)
        date_range = pd.date_range(start_date, end_date)[:-1]

        s_ = pd.Series(dict(zip(date_range,v))).astype(float)
        if s is None:
            s = s_
        else:
            s = s.append(s_)
    return s

############ %%

queries = {
'heat':{'tab':'heat_tab','data_ix':0},
'energy':{'tab':'energy_tab','data_ix':1},
'energy_solar':{'tab':'energy_tab','data_ix':0},
'water_hot':{'tab':'water_tab','data_ix':0},
'water_cold':{'tab':'water_tab','data_ix':1},
}

df = pd.DataFrame()
for name,params in queries.items():
    print(name)
    s = get_data(**params)
    df[name] = s


df.to_csv('2021.csv',index_label='date')

############################################################################# %%
## Build df_X
############################################################################# %%
df = pd.read_csv('2021.csv').set_index('date')

df_weather = pd.read_csv('ugz_ogd_meteo_d1_2021.csv')

df_weather = df_weather.groupby(['Datum','Parameter'])['Wert'].median().reset_index().pivot('Datum','Parameter','Wert')
df_weather.index = df_weather.index.astype(str).str[:10]

df.index = df.index.astype(str).str[:10]

df_ = pd.merge(df,df_weather,left_index=True,right_index=True,how='left')
#df_['T_d_1'] = ( df_['T'].shift(1)+df_['T'].shift(2)+df_['T'].shift(3) )/3

df_['T_1'] = df_['T'].shift(1)
df_['T_2'] = df_['T'].shift(2)

df_['energy_solar_1'] = df_['energy_solar'].shift(1)
df_['energy_solar_2'] = df_['energy_solar'].shift(2)

#df_ = df_.rolling(3).mean()

############ %%

#feat_cols = ['energy','energy_solar','water_hot','water_cold']+['T','T_1','T_2','heat_1']
feat_cols = ['energy','energy_solar','energy_solar_1','energy_solar_2']+['T','T_1','T_2']

df_X = df_[feat_cols+['heat']]
df_X = df_X.dropna()

#px.scatter_matrix(df_X,width=2000,height=2000)
#px.line( df_X[['heat','energy','energy_solar','T_1']],width=1000,height=500)


df_X = df_X[ ( df_X.index >= '2021-11-17' ) & ( df_X.index <= '2021-12-25') ]


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

mod = LinearRegression()
#mod = RandomForestRegressor(min_samples_leaf=5)

Y = df_X['heat']
X = df_X[feat_cols]

mod.fit(X,Y)

pd.Series(dict(zip(feat_cols,mod.coef_))).plot.bar()


df_X['heat_pred'] = mod.predict(X)

print( r2_score(df_X['heat'],df_X['heat_pred']) )

#px.line( df_X[['water_hot','water_cold']],width=1000,height=500)
#px.line( df_X[feat_cols+['heat','heat_pred']],width=1000,height=500)




############################################################################# %%
## Explainer Dashboard
############################################################################# %%


px.scatter(df_X,y='heat',x='T')
from explainerdashboard import RegressionExplainer, ExplainerDashboard
explainer = RegressionExplainer(mod, X, Y)


#Start the Dashboard
db = ExplainerDashboard(explainer,title="heat",whatif=False)
#Running the app on a local port 3050
db.run(port=3050)


############################################################################# %%
## jax model
############################################################################# %%

import os

from IPython.display import set_matplotlib_formats
import jax.numpy as np
from jax import random, vmap
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

plt.style.use("bmh")
if "NUMPYRO_SPHINXBUILD" in os.environ:
    set_matplotlib_formats("svg")

assert numpyro.__version__.startswith("0.8.0")

def logistic(a,b,x):
    return 1/(1+np.exp(-b*(x-a)))

def model(temp, temp_1, temp_2, energy, energy_solar, energy_solar_1, energy_solar_2, water_cold=None, water_hot=None, heat=None):
    base = numpyro.sample("base", dist.Normal(0.0, 10))
    temp_ = numpyro.sample("temp_", dist.Exponential(1))
    temp_1_ = numpyro.sample("temp_1_", dist.Exponential(1))
    temp_2_ = numpyro.sample("temp_2_", dist.Exponential(1))
    energy_ = numpyro.sample("energy_", dist.Exponential(1))
    energy_solar_ = numpyro.sample("energy_solar_", dist.Exponential(1))
    energy_solar_1_ = numpyro.sample("energy_solar_1_", dist.Exponential(1))
    energy_solar_2_ = numpyro.sample("energy_solar_2_", dist.Exponential(1))
    # water_cold_ = numpyro.sample("water_cold_", dist.Normal(0.0, 20))
    # water_hot_ = numpyro.sample("water_hot_", dist.Normal(0.0, 20))

    at_home_slope = 1 #numpyro.sample("at_home_slope", dist.Uniform(0,10))
    at_home_energy = numpyro.sample("at_home_energy", dist.Uniform(0,5))
    at_home = logistic(at_home_energy,at_home_slope,energy)

    #at_home = water_hot >= 3 #at_home_energy
    at_home_ = numpyro.sample("at_home_", dist.Normal(0.0, 20))

    temp_exp = 1 #numpyro.sample("temp_exp", dist.Uniform(1,2))

    heat_pred = \
    -temp_*(temp-base)**temp_exp + \
    -temp_1_*(temp_1-base)**temp_exp + \
    -temp_2_*(temp_2-base)**temp_exp + \
    -energy_*energy + \
    -energy_solar_*energy_solar + \
    -energy_solar_1_*energy_solar_1 + \
    -energy_solar_2_*energy_solar_2 + \
    at_home_*at_home
    # water_cold_*water_cold + \
    # water_hot_*water_hot + \

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(heat_pred,sigma), obs=heat)


# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(model)
num_samples = 2000
mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
mcmc.run(
    rng_key_,
    heat=df_X['heat'].values,
    temp=df_X['T'].values,
    temp_1=df_X['T_1'].values,
    temp_2=df_X['T_2'].values,
    energy=df_X['energy'].values,
    energy_solar=df_X['energy_solar'].values,
    energy_solar_1=df_X['energy_solar_1'].values,
    energy_solar_2=df_X['energy_solar_2'].values
    # water_cold=df_X['water_cold'].values,
    # water_hot=df_X['water_hot'].values
)

mcmc.print_summary()
samples = mcmc.get_samples()
C = { coeff : float(np.mean(samples[coeff])) for coeff in samples }
px.bar( pd.Series(C) )


#px.histogram( samples['at_home_slope'] )


############################################################################# %%
## Inference
############################################################################# %%


from numpyro.infer import Predictive

rng_key, rng_key_ = random.split(rng_key)
predictive = Predictive(model, samples)

predictions = predictive(rng_key_,
temp=df_X['T'].values,
temp_1=df_X['T_1'].values,
temp_2=df_X['T_2'].values,
energy=df_X['energy'].values,
energy_solar=df_X['energy_solar'].values,
energy_solar_1=df_X['energy_solar_1'].values,
energy_solar_2=df_X['energy_solar_2'].values
#water_cold=df_X['water_cold'].values,
#water_hot=df_X['water_hot'].values
)["obs"]

df_X["heat_pred"] = jnp.mean(predictions, axis=0).tolist()
print('r2:', r2_score(df_X['heat'],df_X['heat_pred']) )

px.line( df_X[['heat','heat_pred','energy','energy_solar','T']],width=1000,height=500)



############################################################################# %%
## apply model
############################################################################# %%


df_X['heat_pred'] = \
C['base'] + \
C['temp_']*df_X['T_1'] + \
C['energy_']*df_X['energy'] + \
C['energy_solar_']*df_X['energy_solar'] + \
C['water_cold_']*df_X['water_cold'] + \
C['water_hot_']*df_X['water_hot']


logistic(C['at_home_energy'],C['at_home_slope'],df_X['water_hot'])


from sklearn.metrics import r2_score

r2_score(df_X['heat'],df_X['heat_pred'])



px.line( df_X[['heat','heat_pred','energy','water_hot']],width=1000,height=500)






############ %%
