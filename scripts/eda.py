# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def col_assign(datas, num_col, cat_col):
    for i in datas.columns:
        if datas[i].dtypes == 'float':
            num_col.append(i)
        else:
            cat_col.append(i)

    return num_col, cat_col
def fix_cat_cols(data,cat_col):
    for i in cat_col:
        data[i] = data[i].fillna(data[i].mode()[0])
def fix_num_cols(data,num_col):
    for i in num_col:
        data[i] = data[i].fillna(data[i].mean())
    return data




def top_produce(data,handset, manus):
    top_handset = data['handset'].value_counts().frame().nlargest(10)
    top_manus = data['manus'].value_counts().frame().nlargest(3)
    return top_handset, top_manus
def aggregation(datas):  
    group = [] 
    new_df = datas.loc[datas["Handset Manufacturer"]=="Apple",:]
    new_df1 = datas.loc[datas["Handset Manufacturer"]=="Samsung",:]
    new_df2 = datas.loc[datas["Handset Manufacturer"]=="Huawei",:]
    num_large = new_df['Handset Type'].value_counts().nlargest(5)
    num_large1 = new_df1['Handset Type'].value_counts().nlargest(5)
    num_large2 = new_df2['Handset Type'].value_counts().nlargest(5)
    num_large = pd.DataFrame(num_large.reset_index())
    num_large1 = pd.DataFrame(num_large1.reset_index())
    num_large2 = pd.DataFrame(num_large2.reset_index())
    num_large.rename(columns = {'index':'Apple', 'Handset Type':'No_apple_product'}, inplace = True)
    num_large1.rename(columns = {'index':'Samsung', 'Handset Type':'No_samsung_product'}, inplace = True)
    num_large2.rename(columns = {'index':'Huawei','Handset Type':'No_huawei_product'}, inplace = True)
    group.append(num_large)
    group.append(num_large1)
    group.append(num_large2)
    group = pd.concat(group, axis =1)
    return group

"""# **TASK ONE**

### 1.1
"""
def group_count(data,x,y):
    x_per_y = pd.DataFrame(data.groupby([x]).agg({y:'count'}).reset_index())
    return x_per_y

def group_sum(data,x,y):
    x_persum_y = pd.DataFrame(data.groupby([x]).agg({y:'sum'}).reset_index())
    return x_persum_y
def group_double_sum(data,x,y,z):
    xy_per_z = pd.DataFrame(data.groupby([x,y]).agg({z:'count'}).reset_index())
    return xy_per_z
aggregates_user = []
Dur_Per_User=[]
Total_Data_Vol_user =[]
Dl_UL_per_user=[]
'''
def aggregate(datas):
  agg2 = pd.DataFrame(datas.groupby(['Total_Data_Volume(MB)']).agg(TOTAL_DUR = ('MSISDN/Number', sum)).reset_index())
  aggregates_user.append(agg)
  Dur_Per_User.append(agg1)
  Total_Data_Vol_user.append(agg2)
  Dl_UL_per_user.append(agg3)
aggregate(data)
aggregates_user = pd.concat(aggregates_user, axis=1)
Dur_Per_User = pd.concat(Dur_Per_User, axis=1)
Total_Data_Vol_user = pd.concat(Total_Data_Vol_user, axis=1)
Dl_UL_per_user = pd.concat(Dl_UL_per_user, axis=1)
'''
"""### 1.2.3"""

from sklearn.preprocessing import MinMaxScaler
num_cols =[]
cat_cols= []
def convertbyte_scale(datas, substrings, replaces, div_value ):
    my_bytes = [j for j in datas.columns if substrings in j]
    for i in  my_bytes:
        datas[i.replace(substrings,replaces)] = datas[i]/div_value
        datas.drop(i, axis = 1, inplace = True)
    return datas

def min_scale(datas, col_names, x,y):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(x, y))
    datas[col_names] = scaler.fit_transform(datas[col_names])

def min_scale(datas, col_names):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    datas[col_names] = scaler.fit_transform(datas[col_names])
    return datas
def non_grahical_EDA(data,relevant_num,relevant_cat):
    for cols in relevant_num:
        print(data[cols].describe())
        print(f"Column name is {cols}")
        print(f'skewness for this column is {data[cols].skew()}')
        print(f'kurtosis for this column is {data[cols].kurtosis()}')
        Q3,Q1 = np.percentile(data[cols], [75,25])
        IQR = Q3 - Q1
        print(f'The IQR is {IQR}')
        print(f'The number of Unique value of column {cols} is : {data[cols].nunique()}')
        print('____________________________________________________________________')
    for cols in relevant_cat:
        print(data[cols].describe(include=['O']))
def univariate_plot(data,relevant_num):
    for cols in relevant_num:
        sns.histplot(data=data, x= cols )
        sns.boxplot(data=data, x= cols )
        sns.kdeplot(data=data, x= cols )
        plt.show()


def bivariate_plot(data,relevant_app,x):
    for i in relevant_app:
        sns.scatterplot(data=data,x=x,y=i,alpha=0.5)
        plt.title(f'graph of {i} against {x}')
        plt.xlabel(x)
        plt.ylabel(i)
        plt.show()
'''
def variable_transformation(data,x,y,newl):
    data[y] = pd.qcut(data[x], 10,labels=False,duplicates= 'drop')
    New_df = pd.DataFrame()
    New_df['total_data_UL+DL'] = data['Total_Data_volume (MB)']
    New_df['MSISDN/Number'] = data['MSISDN/Number']
    New_df['top_5_decile_Dur. (MS)'] = data['top_5_decile_Dur. (MS)']

    new_df = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==3,:]
    new_df1 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==2,:]
    new_df2 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==0,:]
    new_df3 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==8,:]
    new_df4 = New_df.loc[New_df["top_5_decile_Dur. (MS)"]==7,:]

    new_df = pd.DataFrame(new_df.reset_index())
    new_df1 = pd.DataFrame(new_df1.reset_index())
    new_df2 = pd.DataFrame(new_df2.reset_index())
    new_df3 = pd.DataFrame(new_df3.reset_index())
    new_df4 = pd.DataFrame(new_df4.reset_index())



    newl.append(new_df)
    newl.append(new_df1)
    newl.append(new_df2)
    newl.append(new_df3)
    newl.append(new_df4)

    top_5s = pd.concat(newl,axis=0)

    top_5s.drop("index",axis=1,inplace=True)
    return top_5s

def corr(data):
    df_data = pd.DataFrame()

    df_data['Social Media data'] = data['Social Media DL (MB)'] + data['Social Media UL (MB)']
    df_data['Google data'] = data['Google DL (MB)'] + data['Google UL (MB)']
    df_data['Email data'] = data['Email DL (MB)'] + data['Email UL (MB)']
    df_data['Youtube data'] = data['Youtube DL (MB)'] + data['Youtube UL (MB)']
    df_data['Netflix data'] = data['Netflix DL (MB)'] + data['Netflix UL (MB)']
    df_data['Gaming data'] = data['Gaming DL (MB)'] + data['Gaming UL (MB)']
    df_data['Other data'] = data['Other DL (MB)'] + data['Other UL (MB)']

    return df_data.corr()
    '''
def PCA(data,principalDf, principal1_name, principal2_name):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data= principalComponents,columns = [principal1_name, principal2_name])
    return principalDf

def fix_outlier(df, column):
    df[column] = np.where(df[column]>df[column].quantile(0.95),
                         df[column].mean(),
                         df[column])
    return df[column]
