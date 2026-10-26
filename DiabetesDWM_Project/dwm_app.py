# Diabetes Data Warehouse & Mining Lab — educational demo, not a diagnostic tool.
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
CSV = BASE / 'diabetes_dataset.csv'
DB = BASE / 'diabetes_warehouse.db'
FEATURES = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

st.set_page_config(page_title='Diabetes DWM Studio', page_icon='📊', layout='wide')
st.markdown('''<style>
.main {background:#f7f9fc;} .block-container {padding-top:1.4rem;}
.hero {padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#102a43,#176b87);color:white;margin-bottom:1rem}
.hero h1 {color:white;margin:0}.hero p {color:#e4f4fa;margin:.4rem 0 0}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_raw():
    df = pd.read_csv(CSV)
    df = df.drop_duplicates().copy()
    df['Gender'] = df['Gender'].map({0:'Female',1:'Male'}).fillna(df['Gender'].astype(str))
    df['AgeBand'] = pd.cut(df['Age'], bins=[0,29,44,59,120], labels=['18–29','30–44','45–59','60+'], include_lowest=True).astype(str)
    df['GlucoseBand'] = pd.cut(df['Glucose'], bins=[-np.inf,99,125,np.inf], labels=['Normal range','Elevated','High'], right=True).astype(str)
    df['BMIBand'] = pd.cut(df['BMI'], bins=[-np.inf,18.5,25,30,np.inf], labels=['Underweight','Healthy range','Overweight','Obesity range']).astype(str)
    df['OutcomeLabel'] = df['Outcome'].map({0:'No diabetes label',1:'Diabetes label'})
    return df

def create_warehouse(df):
    # Rebuild a small star schema from the source CSV (repeatable ETL).
    con=sqlite3.connect(DB)
    cur=con.cursor()
    cur.executescript('''DROP TABLE IF EXISTS fact_patient; DROP TABLE IF EXISTS dim_patient_profile; DROP TABLE IF EXISTS dim_age; DROP TABLE IF EXISTS dim_glucose; DROP TABLE IF EXISTS dim_bmi; DROP TABLE IF EXISTS dim_outcome;
    CREATE TABLE dim_patient_profile(profile_key INTEGER PRIMARY KEY, gender TEXT UNIQUE);
    CREATE TABLE dim_age(age_key INTEGER PRIMARY KEY, age_band TEXT UNIQUE);
    CREATE TABLE dim_glucose(glucose_key INTEGER PRIMARY KEY, glucose_band TEXT UNIQUE);
    CREATE TABLE dim_bmi(bmi_key INTEGER PRIMARY KEY, bmi_band TEXT UNIQUE);
    CREATE TABLE dim_outcome(outcome_key INTEGER PRIMARY KEY, outcome_label TEXT UNIQUE);
    CREATE TABLE fact_patient(fact_key INTEGER PRIMARY KEY, profile_key INTEGER, age_key INTEGER, glucose_key INTEGER, bmi_key INTEGER, outcome_key INTEGER,
      glucose REAL, blood_pressure REAL, skin_thickness REAL, insulin REAL, bmi REAL, pedigree REAL, age INTEGER, record_count INTEGER DEFAULT 1,
      FOREIGN KEY(profile_key) REFERENCES dim_patient_profile(profile_key), FOREIGN KEY(age_key) REFERENCES dim_age(age_key),
      FOREIGN KEY(glucose_key) REFERENCES dim_glucose(glucose_key), FOREIGN KEY(bmi_key) REFERENCES dim_bmi(bmi_key), FOREIGN KEY(outcome_key) REFERENCES dim_outcome(outcome_key));''')
    dimensions=[('dim_patient_profile','gender','Gender'),('dim_age','age_band','AgeBand'),('dim_glucose','glucose_band','GlucoseBand'),('dim_bmi','bmi_band','BMIBand'),('dim_outcome','outcome_label','OutcomeLabel')]
    maps={}
    for table, col, src_col in dimensions:
        vals=sorted(df[src_col].dropna().astype(str).unique())
        cur.executemany(f'INSERT INTO {table}({col}) VALUES (?)', [(v,) for v in vals])
        keycol={'dim_patient_profile':'profile_key','dim_age':'age_key','dim_glucose':'glucose_key','dim_bmi':'bmi_key','dim_outcome':'outcome_key'}[table]
        maps[src_col]={v:k for k,v in cur.execute(f'SELECT {keycol},{col} FROM {table}').fetchall()}
    rows=[]
    for _,r in df.iterrows():
        rows.append((maps['Gender'][str(r.Gender)],maps['AgeBand'][str(r.AgeBand)],maps['GlucoseBand'][str(r.GlucoseBand)],maps['BMIBand'][str(r.BMIBand)],maps['OutcomeLabel'][str(r.OutcomeLabel)],float(r.Glucose),float(r.BloodPressure),float(r.SkinThickness),float(r.Insulin),float(r.BMI),float(r.DiabetesPedigreeFunction),int(r.Age)))
    cur.executemany('INSERT INTO fact_patient(profile_key,age_key,glucose_key,bmi_key,outcome_key,glucose,blood_pressure,skin_thickness,insulin,bmi,pedigree,age) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',rows)
    con.commit(); con.close()

def query(sql, params=()):
    con=sqlite3.connect(DB)
    try: return pd.read_sql_query(sql,con,params=params)
    finally: con.close()

def ensure_warehouse(df):
    if not DB.exists(): create_warehouse(df)

raw=load_raw(); ensure_warehouse(raw)
st.markdown('<div class="hero"><h1>📊 Diabetes Data Warehouse & Data Mining Studio</h1><p>Star schema · ETL · OLAP · SQL exploration · clustering · classification · model evaluation</p></div>',unsafe_allow_html=True)
st.caption('Academic DWM project built on the supplied diabetes dataset. Dataset labels and model outputs are for coursework only, not medical diagnosis.')
with st.sidebar:
    st.header('Project controls')
    st.write(f'**Source rows:** {len(raw):,}')
    st.write(f'**Warehouse:** `{DB.name}`')
    if st.button('Run ETL / rebuild warehouse',use_container_width=True):
        create_warehouse(raw); st.success('ETL completed: dimensions and fact table rebuilt.'); st.rerun()
    st.divider()
    st.markdown('**Star schema**')
    st.code('''dim_patient_profile ─┐
dim_age ─────────────┤
dim_glucose ─────────┼── fact_patient
dim_bmi ─────────────┤
dim_outcome ─────────┘''')

tabs=st.tabs(['Overview & ETL','Warehouse Schema','OLAP Queries','Data Mining','SQL Explorer','Project Guide'])
with tabs[0]:
    st.subheader('Warehouse overview')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Fact records',f'{len(raw):,}'); c2.metric('Dimensions',5); c3.metric('Numeric measures',7); c4.metric('Duplicate rows removed',int(pd.read_csv(CSV).duplicated().sum()))
    st.markdown('**ETL pipeline**')
    st.write('1. **Extract:** read `diabetes_dataset.csv`.  2. **Transform:** remove duplicate rows, map gender and derive age/glucose/BMI bands.  3. **Load:** populate dimension tables and the central fact table in SQLite.')
    st.write('**Measure summary**')
    st.dataframe(raw[FEATURES].describe().T.round(2),use_container_width=True)
    col1,col2=st.columns(2)
    with col1: st.plotly_chart(px.histogram(raw,x='AgeBand',color='OutcomeLabel',barmode='group',title='Records by age band'),use_container_width=True)
    with col2: st.plotly_chart(px.histogram(raw,x='Glucose',color='OutcomeLabel',nbins=25,title='Glucose distribution by dataset label'),use_container_width=True)
with tabs[1]:
    st.subheader('Star schema and data dictionary')
    st.write('**Grain:** one row in `fact_patient` represents one source dataset record. **Fact measures:** glucose, blood pressure, skin thickness, insulin, BMI, pedigree function, age, and record_count.')
    schema=pd.DataFrame([
      ['fact_patient','Fact','Numeric measurements and dimension foreign keys'],['dim_patient_profile','Dimension','Gender profile'],['dim_age','Dimension','Age band'],['dim_glucose','Dimension','Glucose band'],['dim_bmi','Dimension','BMI band'],['dim_outcome','Dimension','Source outcome label'] ],columns=['Table','Type','Purpose'])
    st.dataframe(schema,use_container_width=True,hide_index=True)
    st.markdown('**Relationships**')
    st.code('fact_patient.profile_key → dim_patient_profile.profile_key\nfact_patient.age_key → dim_age.age_key\nfact_patient.glucose_key → dim_glucose.glucose_key\nfact_patient.bmi_key → dim_bmi.bmi_key\nfact_patient.outcome_key → dim_outcome.outcome_key')
    st.markdown('**Dimension row counts**')
    st.dataframe(query('SELECT "dim_patient_profile" AS table_name, COUNT(*) AS rows FROM dim_patient_profile UNION ALL SELECT "dim_age",COUNT(*) FROM dim_age UNION ALL SELECT "dim_glucose",COUNT(*) FROM dim_glucose UNION ALL SELECT "dim_bmi",COUNT(*) FROM dim_bmi UNION ALL SELECT "dim_outcome",COUNT(*) FROM dim_outcome UNION ALL SELECT "fact_patient",COUNT(*) FROM fact_patient'),use_container_width=True,hide_index=True)
with tabs[2]:
    st.subheader('OLAP operations')
    op=st.selectbox('Choose OLAP operation',['Roll-up','Drill-down','Slice','Dice','Pivot / cross-tab','Measure aggregation'])
    if op=='Roll-up':
        st.caption('Aggregate detailed records into a higher-level age-band summary.')
        out=query('''SELECT a.age_band, COUNT(*) AS records, ROUND(AVG(f.glucose),2) AS avg_glucose, ROUND(AVG(f.bmi),2) AS avg_bmi, ROUND(100.0*AVG(CASE WHEN o.outcome_label="Diabetes label" THEN 1.0 ELSE 0.0 END),2) AS diabetes_label_pct FROM fact_patient f JOIN dim_age a USING(age_key) JOIN dim_outcome o USING(outcome_key) GROUP BY a.age_band ORDER BY a.age_band''')
        st.dataframe(out,use_container_width=True,hide_index=True); st.plotly_chart(px.bar(out,x='age_band',y='records',color='diabetes_label_pct',title='Roll-up: records by age band',labels={'age_band':'Age band','records':'Records','diabetes_label_pct':'Diabetes label %'}),use_container_width=True)
    elif op=='Drill-down':
        band=st.selectbox('Select age band',sorted(raw.AgeBand.unique()))
        out=query('''SELECT a.age_band,g.glucose_band,b.bmi_band,COUNT(*) AS records,ROUND(AVG(f.glucose),2) AS avg_glucose,ROUND(AVG(f.bmi),2) AS avg_bmi FROM fact_patient f JOIN dim_age a USING(age_key) JOIN dim_glucose g USING(glucose_key) JOIN dim_bmi b USING(bmi_key) WHERE a.age_band=? GROUP BY a.age_band,g.glucose_band,b.bmi_band ORDER BY records DESC''',(band,))
        st.dataframe(out,use_container_width=True,hide_index=True)
    elif op=='Slice':
        gender=st.selectbox('Slice by gender',['All']+sorted(raw.Gender.unique().tolist()))
        sql='''SELECT p.gender,COUNT(*) AS records,ROUND(AVG(f.glucose),2) AS avg_glucose,ROUND(AVG(f.bmi),2) AS avg_bmi,ROUND(AVG(f.blood_pressure),2) AS avg_blood_pressure FROM fact_patient f JOIN dim_patient_profile p USING(profile_key)'''
        out=query(sql+(' WHERE p.gender=?' if gender!='All' else '')+' GROUP BY p.gender',(gender,) if gender!='All' else ())
        st.dataframe(out,use_container_width=True,hide_index=True)
    elif op=='Dice':
        ages=st.multiselect('Age bands',sorted(raw.AgeBand.unique()),default=sorted(raw.AgeBand.unique()))
        outcomes=st.multiselect('Outcome labels',sorted(raw.OutcomeLabel.unique()),default=sorted(raw.OutcomeLabel.unique()))
        if ages and outcomes:
            placeholders=','.join('?'*len(ages)); placeholders2=','.join('?'*len(outcomes))
            out=query(f'''SELECT a.age_band,o.outcome_label,COUNT(*) AS records,ROUND(AVG(f.glucose),2) AS avg_glucose,ROUND(AVG(f.bmi),2) AS avg_bmi FROM fact_patient f JOIN dim_age a USING(age_key) JOIN dim_outcome o USING(outcome_key) WHERE a.age_band IN ({placeholders}) AND o.outcome_label IN ({placeholders2}) GROUP BY a.age_band,o.outcome_label ORDER BY a.age_band''',tuple(ages+outcomes))
            st.dataframe(out,use_container_width=True,hide_index=True)
        else: st.info('Select at least one age band and outcome label.')
    elif op=='Pivot / cross-tab':
        pivot=pd.pivot_table(raw,index='AgeBand',columns='OutcomeLabel',values='Glucose',aggfunc='mean').round(2).reset_index()
        st.write('Average glucose by age band and outcome label'); st.dataframe(pivot,use_container_width=True,hide_index=True)
        st.plotly_chart(px.imshow(pivot.set_index('AgeBand'),text_auto=True,aspect='auto',title='Pivot: average glucose'),use_container_width=True)
    else:
        metric=st.selectbox('Measure',['Glucose','BloodPressure','Insulin','BMI','Age','DiabetesPedigreeFunction'])
        out=query(f'SELECT COUNT(*) AS count, ROUND(MIN({metric if metric!="DiabetesPedigreeFunction" else "pedigree"}),2) AS minimum, ROUND(MAX({metric if metric!="DiabetesPedigreeFunction" else "pedigree"}),2) AS maximum, ROUND(AVG({metric if metric!="DiabetesPedigreeFunction" else "pedigree"}),2) AS average FROM fact_patient')
        st.dataframe(out,use_container_width=True,hide_index=True)
    st.markdown('**Example OLAP SQL**')
    st.code('''-- Roll-up: age band aggregation
SELECT a.age_band, COUNT(*) AS records, AVG(f.glucose) AS avg_glucose
FROM fact_patient f JOIN dim_age a USING(age_key)
GROUP BY a.age_band;

-- Slice: only Male records
SELECT p.gender, AVG(f.bmi) AS avg_bmi
FROM fact_patient f JOIN dim_patient_profile p USING(profile_key)
WHERE p.gender = 'Male' GROUP BY p.gender;''',language='sql')
with tabs[3]:
    st.subheader('Data mining workbench')
    mine=st.radio('Mining task',['Decision tree classification','Random forest comparison','K-means clustering','Correlation & feature importance'],horizontal=True)
    X=raw[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(raw[FEATURES].median()); y=raw['Outcome'].astype(int)
    if mine in ['Decision tree classification','Random forest comparison']:
        test_size=st.slider('Test set (%)',10,40,20,5)/100
        if mine=='Decision tree classification': model=DecisionTreeClassifier(max_depth=4,class_weight='balanced',random_state=42); modelname='Decision Tree'
        else: model=RandomForestClassifier(n_estimators=150,class_weight='balanced',random_state=42); modelname='Random Forest'
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
        model.fit(Xtr,ytr); pred=model.predict(Xte)
        a,b,c,d=st.columns(4); a.metric('Accuracy',f'{accuracy_score(yte,pred):.3f}'); b.metric('Precision',f'{precision_score(yte,pred,zero_division=0):.3f}'); c.metric('Recall',f'{recall_score(yte,pred,zero_division=0):.3f}'); d.metric('F1 score',f'{f1_score(yte,pred,zero_division=0):.3f}')
        cm=confusion_matrix(yte,pred,labels=[0,1]); st.plotly_chart(px.imshow(cm,text_auto=True,x=['Predicted 0','Predicted 1'],y=['Actual 0','Actual 1'],title=f'{modelname} confusion matrix',color_continuous_scale='Blues'),use_container_width=True)
        imp=pd.DataFrame({'Feature':FEATURES,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False)
        st.plotly_chart(px.bar(imp,x='Importance',y='Feature',orientation='h',title=f'{modelname} feature importance'),use_container_width=True)
        st.caption('Metrics describe this dataset split only; they are not evidence of clinical effectiveness.')
    elif mine=='K-means clustering':
        k=st.slider('Number of clusters (k)',2,6,3)
        scaled=StandardScaler().fit_transform(X)
        km=KMeans(n_clusters=k,random_state=42,n_init=10); labels=km.fit_predict(scaled)
        clustered=raw.copy(); clustered['Cluster']=labels.astype(str)
        st.metric('Within-cluster sum of squares (inertia)',f'{km.inertia_:.2f}')
        st.plotly_chart(px.scatter(clustered,x='Glucose',y='BMI',color='Cluster',symbol='OutcomeLabel',hover_data=['Age','Gender'],title='K-means patient-record clusters'),use_container_width=True)
        st.dataframe(clustered.groupby('Cluster')[FEATURES].mean().round(2),use_container_width=True)
        st.caption('Clusters are unsupervised groupings of records, not medical risk categories.')
    else:
        corr=raw[FEATURES+['Outcome']].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr,text_auto='.2f',aspect='auto',title='Correlation matrix',color_continuous_scale='RdBu_r',zmin=-1,zmax=1),use_container_width=True)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); m=RandomForestClassifier(n_estimators=150,random_state=42).fit(Xtr,ytr)
        imp=pd.DataFrame({'Feature':FEATURES,'Importance':m.feature_importances_}).sort_values('Importance',ascending=False)
        st.plotly_chart(px.bar(imp,x='Importance',y='Feature',orientation='h',title='Random Forest feature importance'),use_container_width=True)
with tabs[4]:
    st.subheader('Read-only SQL explorer')
    st.write('Query the SQLite warehouse. Only SELECT statements are accepted.')
    default='''SELECT a.age_band, o.outcome_label, COUNT(*) AS records, ROUND(AVG(f.glucose),2) AS avg_glucose
FROM fact_patient f
JOIN dim_age a USING(age_key)
JOIN dim_outcome o USING(outcome_key)
GROUP BY a.age_band, o.outcome_label
ORDER BY a.age_band;'''
    sql=st.text_area('SQL query',value=default,height=160)
    if st.button('Run SQL query'):
        if not sql.strip().lower().startswith('select') or ';' in sql.strip().rstrip(';'):
            st.error('For safety, enter a single SELECT query only.')
        else:
            try: st.dataframe(query(sql.strip().rstrip(';')),use_container_width=True,hide_index=True)
            except Exception as e: st.error(f'SQL error: {e}')
with tabs[5]:
    st.subheader('How to run and demonstrate')
    st.markdown('''**Windows**
1. Extract the ZIP file.
2. Open the extracted project folder in Command Prompt.
3. Run `run_windows.bat` (or use the commands below).

**Manual setup**
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run dwm_app.py
```

**Suggested viva flow:** explain ETL → star schema and grain → dimensions/measures → roll-up/drill-down/slice/dice/pivot → classification metrics → K-means clusters → limitations.

**DWM concepts covered:** data staging/transformation, star schema, fact/dimension tables, surrogate keys, SQLite warehouse, OLAP aggregation and filtering, pivot/crosstab, SQL exploration, EDA, classification, clustering, correlation, feature importance, confusion matrix, accuracy, precision, recall and F1.

**Limitations:** this is a compact academic warehouse using a single source file; no real-time feeds, enterprise orchestration, or clinical validation. Outcome labels reflect the supplied dataset, and should not be treated as diagnostic truth.''')
