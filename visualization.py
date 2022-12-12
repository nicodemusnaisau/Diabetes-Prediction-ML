import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df_pd = pd.read_csv("diabetes.csv")


def library():
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt

def histogram():
    """
    This function can be used to create a histogram, which is a plot that shows the frequency of 
    different values within a data set. For example, you could use it to show the frequency of different outcomes
    """
    df_pd = pd.read_csv("diabetes.csv")
    df_pd['Outcome'].map(lambda r: {0: 'NonDiabetes', 1: 'Diabetes'}[r])

    num_cols = ['Glucose', 'BMI', 'DiabetesPedigreeFunction']
    palette = {0: 'blue', 1: 'yellow'}
    for i in num_cols:
        sns.displot(data = df_pd, kde=True, x = df_pd[str(i)], hue='Outcome', palette = palette)
        plt.show()
        
def barplot(df, col, lim=10, yname=None):
    
    '''
    This function makes a bar plot for the Spark dataframe df 
    for categorical column col only selecting top categories 
    as specified by lim. 
    '''
    df_pd = pd.read_csv("diabetes.csv")
    # Grouping by the categories, counting on each categories 
    # and ordering them by the count
    classes = df.groupBy(col).count().orderBy('count', ascending=False)
    
    # Take first 'lim' number of rows and convert to pandas  
    pd_df = classes.limit(lim).toPandas()
    
    # Making plot 
    pd_df.plot(kind='bar', x=col, legend=False)
    plt.ylabel(yname)
    plt.show()
    print(pd_df)
    
def correlation_heatmap ():
    """ 
    code will compute the correlation between all pairs of variables in the data set and print the correlation matrix
    """
    df_pd = pd.read_csv("diabetes.csv")
    corr=df_pd.corr()

    sns.set(font_scale=1.15)
    plt.figure(figsize=(14, 10))

    sns.heatmap(corr, vmax=.8, linewidths=0.01,
                square=True,annot=True,cmap='Blues',linecolor="black")
    plt.title('Correlation between features')    

    
def show():
    histogram()
    barplot(df_pd, 'Outcome', lim=30, yname='Number of Count')
    correlation_heatmap()
    