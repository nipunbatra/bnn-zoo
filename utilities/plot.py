import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import jax.numpy as jnp

def plot_actualdata(X,Y,x_test_1,y_test_1,x_test_2,y_test_2):
    plt.scatter(X,Y,color='black',alpha=0.5)
    plt.scatter(x_test_1,y_test_1,color='red',alpha=0.5)
    plt.scatter(x_test_2,y_test_2,color='red',alpha=0.5)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim(-0.5, 1.0)
    plt.ylim(-2,2)
    sns.despine()
    # plt.show()

def calibration_regression(mean,sigma,Y,ax=None):
    if ax is None:
      fig,ax = plt.subplots()
    df = pd.DataFrame()
    df['mean']=mean
    df['sigma']=sigma
    df['Y']=Y
    df['z']=(df['Y']-df['mean'])/df['sigma']
    df['perc'] = st.norm.cdf(df['z'])
    df['r'] = (df['perc']*10).astype('int')
    k=jnp.arange(0,1.1,.1)
    counts=[]
    
    for i in range(0,11):
    
      l = df[df['perc']<0.5+i*0.05]
      l = l[l['perc']>=0.5-i*0.05]
      counts.append(len(l)/len(df))
    #   print(0.5+i*0.05,0.5-i*0.05,len(l)/len(df))

    plt.plot(k,counts,color='red',label='Prediction')
    plt.plot(k,k,color='black',label='Ideal')
    plt.plot(k,counts,'o',color='red')
    plt.plot(k,k,'o',color='black')
    plt.xticks(k)
    plt.yticks(k)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.xlabel('Decile')
    plt.ylabel('Ratio of points')
    sns.despine()
    return df





