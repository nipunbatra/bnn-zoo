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
    sns.despine()

def calibration_regression(mean,sigma,Y,ax=None):
    if ax is None:
      fig,ax = plt.subplots()
    df = pd.DataFrame()
    df['mean']=mean
    df['sigma']=sigma
    df['Y']=Y
    df['z']=(df['Y']-df['mean'])/df['sigma']
    df['perc'] = st.norm.cdf(df['z'])
    k=jnp.arange(0,1.1,.1)
    counts=[]
    df2 = pd.DataFrame()
    df2['Interval'] = k
    df2['Ideal'] = k
    for i in range(0,11):
      l = df[df['perc']<0.5+i*0.05]
      l = l[l['perc']>=0.5-i*0.05]
      counts.append(len(l)/len(df))
    df2['Counts']=counts

    ax.plot(k,counts,color='red',label='Prediction')
    ax.plot(k,k,color='black',label='Ideal')
    ax.plot(k,counts,'o',color='red')
    ax.plot(k,k,'o',color='black')
    # plt.yticks(k)
    sns.despine()
    return df,df2
def plot_prediction(X,Y,x_stack,y_stack,mean,sigma,title,ax=None,n_points=300):

    ax.plot(x_stack,mean, color='red',linewidth=3)
    for i_std in range(1,4):
      ax.fill_between(x_stack.reshape(n_points), jnp.array((mean-i_std*sigma)), jnp.array((mean+i_std*sigma)), color='lightsalmon',alpha=2/(3*i_std), label=f'$\mu\pm{i_std}\sigma$')
    ax.scatter(x_stack[:int(n_points/3)], y_stack[:int(n_points/3)],color='crimson',alpha=0.5)
    ax.scatter(X, Y,color='black',alpha=0.7)
    ax.scatter(x_stack[int(n_points*2/3):], y_stack[int(n_points*2/3):],color='crimson',alpha=0.5)
    ax.vlines(min(X),min(mean-3*sigma),max(mean+3*sigma),colors='black',linestyles='--')
    ax.vlines(max(X),min(mean-3*sigma),max(mean+3*sigma),colors='black',linestyles='--')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_ylim([min(mean-3*sigma),max(mean+3*sigma)])
    ax.set_xlim([min(x_stack),max(x_stack)])
    ax.set_title(title)
    plt.legend()
    sns.despine()
