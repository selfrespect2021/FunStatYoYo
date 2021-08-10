import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import math
import arabic_reshaper
from bidi.algorithm import get_display
import random as rd

knumber_bins_per_line=2
first_last_name=['المعدل','اسم التلميذ','ر.ت',"رقم التلميذ",]
all_subjects=['النشاط العلمي ',
       'التربية الإسلامية'
          , 'الرياضيات',
          'الاجتماعيات',
          'اللغة الفرنسية',
    'اللغة العربية']
    
selected_feature="المعدل العام"


useful_col=first_last_name+all_subjects
st.title("احصائيات عبد اللطيف بلكاني")

def getComparisonFeature(df):
    if st.checkbox("اختر معيار لتشكيل المجموعات"):
    # get the list of columns
        criteria = list(df.select_dtypes(include=['float64']).columns)
        criterion = st.selectbox("", criteria)
        return criterion
    return "المعدل العام"

def getTaskType():
    # Select columns to display
    if st.checkbox("#1- Choisisez ce que vous voulez faire."):
    # get the list of columns
        tasks = ["Création des groupes","Statistiques"]
        tasks = st.selectbox("", tasks)
        return tasks
    return ""
#Get requested task
option = getTaskType()
if option == "Statistiques":
    st.write("Télèchargez le fichier excel qui contient les moyennes par matières.")
elif option =="Création des groupes":
    st.write("Télèchargez le fichier excel qui contient les moyennes générales des élèves.")

def plotDensity(df, selected_feature, title="",generate_report=True,name_fig="PNG"):
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots()
    axs=df[selected_feature].plot.density(color='red',label='Fille')
    axs.set_xlabel('Les moyennes')
    axs.set_ylabel('La densité')
    axs.set_title(title)
    st.pyplot(fig)
    if generate_report:
        plt.savefig(name_fig)


def plotDensity2(df1,df2,label1="",label2="", title="",generate_report=True,name_fig="PNG"):
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots()
    axs=df1.plot.density(color='red',label=label1)
    axs=df2.plot.density(color='blue',label=label2)
    axs.legend()
    axs.set_xlabel('La moyenne')
    axs.set_ylabel('La densité')
    axs.set_title(title)
    st.pyplot(fig)
    if generate_report:
        plt.savefig(name_fig)



def group_split(df, grp_pct=.5, seed=5145):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(grp_pct * m)
    validate_end = int(grp_pct * m) + train_end
    grp1 = df.iloc[perm[:train_end]]
    grp2 = df.iloc[perm[train_end:validate_end]]
    return grp1,grp2

def isParityBroken(grp):
    return grp['Genre'].value_counts(ascending=True)[0] - grp['Genre'].value_counts(ascending=True)[0] <2
    
def getBestSplit(df,selected_feature,KeepParity=True):
    best_std=10
    best_mean=100
    best_seed = 0
    for seed in range(12,100,1):
        grp1, grp2 = group_split(df,grp_pct=0.5,seed=seed)
        if KeepParity and (isParityBroken(grp1) or isParityBroken(grp2)):
            pass 
        std = abs(grp1[selected_feature].std()-grp2[selected_feature].std())
        mean = abs(grp1[selected_feature].mean()-grp2[selected_feature].mean())
    
        if std<best_std and mean<best_mean :
            best_std = std
            best_mean = mean
            best_seed = seed
    st.write("L'erreur sur l'écart type des deux groupes est: ","**{:.4f}**".format(abs(best_std-best_std)))
    st.write("L'erreur sur la moyenne des deux groupes est: ","**{:.4f}**".format(mean))

    return group_split(df,grp_pct=0.5,seed=best_seed)

# Upload CSV or XLS files
with st.sidebar.header('#2- Télécharger le fichier csv/excel.'):
    uploaded_file = st.sidebar.file_uploader("", type=["csv","xls"])

# Some basic stats analysis on classes
if uploaded_file is not None:
    @st.cache
    def load_file():
        if "csv" in uploaded_file.name:
            csv = pd.read_csv(uploaded_file)
            return csv
        elif "xls" in uploaded_file.name:
            df = pd.read_excel (uploaded_file,skiprows=9)
            df.columns=[i.replace("\n","") for i in df.columns]
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return df

    def getDescription(df,cols):
        return df[cols].describe(include='all')

    def getUsefulColumns():
        # Select columns to display
        if st.checkbox("Cochez ici si vous voulez choisir des matières spécifiques"):
        # get the list of columns
            columns = df.columns.tolist()
            delta_cols= set(columns) - set(first_last_name)
            st.write("#### Choisissez les matières qui vous intéressent :")
            selected_cols = st.multiselect("", delta_cols)
            return selected_cols
        else:
            return all_subjects

    if option == "Statistiques":
        df = load_file()
        # create excel writer object
        writer = pd.ExcelWriter('StatReport.xlsx', engine='xlsxwriter')
        # write dataframe to excel
        df.to_excel(writer, sheet_name="Fichier des données",index=False,startrow=2, startcol=2)

        st.header('** Les données **')
        st.write(df)
        cols = getUsefulColumns()
        st.write('---')
        st.header('** Statistiques globales **')
        st.write(getDescription(df,cols))
        getDescription(df,cols).to_excel(writer, sheet_name="Fichier des données",startrow=2, startcol=4+len(df.columns))
        st.header('** Liste des élèves ayant besoin de soutien par matière **')
        st.write('* Dans le cadre de ce script, on considère qu\'un élève a besoin de soutien dans une matière si sa note dans celle-ci est < 5 / 10.')
        #Get mean of each subject and print list of student below the absolute avergae (for now we enjoy 5/10)
        mean_mat=df[cols].mean()
        for i in cols:
            st.write( "{:.2f}".format(mean_mat[i])+"<>"+i)
            st.write(df[df[i]< 5][[i]+first_last_name])
            selection= df[df[i]< 5][[i]+first_last_name]
            if len(selection.index) >0:
                selection.to_excel(writer, sheet_name=str(i),index=False,startrow=2, startcol=2)


        st.header('** Histogrammes des notes par matière**')
        plt.rcParams.update({'font.size': 5})
        arabic_char = {get_display(arabic_reshaper.reshape(k)): k for k in cols}
        _ = math.ceil(len(cols)/knumber_bins_per_line)
        fig, axs = plt.subplots(_, knumber_bins_per_line, sharey=True, sharex=True)

        #Dirty way to hide useless bins
        count = 0
        for i, _c in enumerate(arabic_char.values()):
            ax = axs.flat[i]
            ax.hist(df[[_c]], bins=10,rwidth=0.5,weights=np.ones(len(df[[_c]])) / len(df[[_c]]),color='k', alpha=0.5)
            ax.set_title(get_display(arabic_reshaper.reshape(_c)))
            count +=1
        # Cleanup empty bins 
        for item in range(count,knumber_bins_per_line*math.ceil(len(cols)/knumber_bins_per_line)):
            fig.delaxes(axs.flat[item])
        fig.supxlabel('Les moyennes des matières')
        fig.supylabel('Le pourcentage des élèves')
        st.pyplot(fig)

        download=st.button('Télécharger')
        if download:
            workbook  = writer.book
            worksheet = writer.sheets['Fichier des données']
            # Insert an image.
            plt.savefig("Histogrammes.png")

            worksheet.insert_image('H3', 'Histogrammes.png')
            writer.save()

    else:
        df = load_file()
        # create excel writer object
        writer = pd.ExcelWriter('GroupSplit.xlsx', engine='xlsxwriter')
        # write dataframe to excel
        df.to_excel(writer, sheet_name="données",index=False, startrow=2, startcol=2)
        st.header('** Les données **')
        st.write(df)
        st.header('** Récap **')
        selected_feature = getComparisonFeature(df)
        st.write(df[selected_feature].describe())
        df[selected_feature].describe().to_excel(writer, sheet_name="données", startrow=4+len(df.index), startcol=2)

        plotDensity(df,selected_feature,"Densité des notes ("+selected_feature+") du groupe entier",name_fig="GlobalDensity.png")
       

        if 'Genre' in df.columns:
            colGender = "Genre"
            g= "Garçon"
            f= "Fille"
        elif ' arabic stuf' in df.columns:
            colGender = " arabic "
            g=" garcon in arabic"
            f=" fille in arabic"
        else:
            colGender=""
        if colGender !="":
            plotDensity2(df[df[colGender]==f][selected_feature],
                df[df[colGender]!=f][selected_feature],g,f,
                title="Densité des notes ("+selected_feature+") du groupe par genre",name_fig="GenreDensity.png")

        st.header('** Création de 2 groupes homogènes **')
        KeepParity =True
        if st.checkbox("Cochez ici si vous voulez ignorer la parité fille/garcon"):
            KeepParity = False
        grp1,grp2 = getBestSplit(df,selected_feature,KeepParity)
        st.write("*Groupe 1:*")
        st.write(grp1)
        st.write(getDescription(grp1,selected_feature))
        st.write("*Groupe 2:*")
        st.write(grp2)
        st.write(getDescription(grp2,selected_feature))
        # Plot both kde
        plotDensity2(grp2[selected_feature],grp1[selected_feature],"Groupe 1","Groupe 2",
            title="Densité des notes ("+selected_feature+") des groupes constitués",name_fig="GroupsDensity.png")
        download=st.button('Télécharger')
        if download:
            grp1.to_excel(writer, sheet_name="Groupes",index=False,startrow=2, startcol=2)
            grp2.to_excel(writer, sheet_name="Groupes",index=False,startrow=6+len(grp1.index), startcol=2)
            # Get the xlsxwriter workbook and worksheet objects.
            workbook  = writer.book
            worksheet = writer.sheets['Groupes']
            # Insert an image.
            worksheet.insert_image('K3', 'GroupsDensity.png')
            worksheet2 = writer.sheets['données']
            worksheet2.insert_image('C3', 'GenreDensity.png')
            worksheet2.insert_image('C25', 'GlobalDensity.png')
            writer.save()

else:
    st.write("...")
