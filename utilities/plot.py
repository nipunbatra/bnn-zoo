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
    k=jnp.arange(0,1.1,.1)
    counts=[]
    df2 = pd.DataFrame()
    df2['Interval'] = k
    df2['Ideal'] = k
    for i in range(0,11):
    
      l = df[df['perc']<0.5+i*0.05]
      l = l[l['perc']>=0.5-i*0.05]
      counts.append(len(l)/len(df))
    #   print(0.5+i*0.05,0.5-i*0.05,len(l)/len(df))
    df2['Counts']=counts

    ax.plot(k,counts,color='red',label='Prediction')
    ax.plot(k,k,color='black',label='Ideal')
    ax.plot(k,counts,'o',color='red')
    ax.plot(k,k,'o',color='black')
    # ax.xticks(k)
    # ax.yticks(k)
    # ax.xlim([0,1])
    # ax.ylim([0,1])
    # ax.legend()
    # ax.xlabel('Decile')
    # ax.ylabel('Ratio of points')
    sns.despine()
    return df,df2




def plot_prediction(X,Y,x_test_1,y_test_1,x_test_2,y_test_2,mean,sigma,mean_lx,sigma_lx,mean_ux,sigma_ux):
    plt.figure(figsize=(10,5))
    plt.plot(X,mean, "r--", linewidth=2)
    plt.plot(x_test_1,mean_lx, "r--", linewidth=2)
    plt.plot(x_test_2,mean_ux, "r--", linewidth=2)
    for i_std in range(1,4):
      plt.fill_between(X.reshape(100), jnp.array((mean-i_std*sigma)), jnp.array((mean+i_std*sigma)), color='red',alpha=1/(3*i_std), label='std'+str(i_std))
    for i_std in range(1,4):
      plt.fill_between(x_test_1.reshape(100), jnp.array((mean_lx-i_std*sigma_lx)), jnp.array((mean_lx+i_std*sigma_lx)), color='red',alpha=1/(3*i_std), label='std'+str(i_std))
    for i_std in range(1,4):
      plt.fill_between(x_test_2.reshape(100), jnp.array((mean_ux-i_std*sigma_ux)), jnp.array((mean_ux+i_std*sigma_ux)), color='red',alpha=1/(3*i_std), label='std'+str(i_std))
    plt.scatter(X, Y,color='black',alpha=0.5)
    plot_actualdata(X,Y,x_test_1,y_test_1,x_test_2,y_test_2)

    # plt.plot(X, final_mean+final_sigma)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim(-2,2)
    sns.despine()
    plt.show()