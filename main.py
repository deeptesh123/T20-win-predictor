import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
bg_image="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQIzQDFIT8RImXm7Fj6WZ7KMV4WCeyjbwmmGw&s");
  background-size: cover;
}
</style>
"""
st.markdown(bg_image, unsafe_allow_html=True)
innings_1=pd.read_csv("first_innings.csv")
innings_2=pd.read_csv("second_innings.csv")
df2=pd.read_csv("pure.csv")
if "pipe" not in st.session_state:
    train=pd.read_csv("model training dataset.csv")
    x=train.iloc[:,:-1]
    y=train.iloc[:,-1]
    x.drop(columns=["Unnamed: 0"],inplace=True)
    ct=ColumnTransformer([("trf",OneHotEncoder(sparse_output=False,drop="first"),['Venue','Bat First', 'Bat Second'])],remainder="passthrough")
    st.session_state.pipe=Pipeline(steps=[("step 1",ct),("step 2",LogisticRegression(solver="liblinear"))])
    st.session_state.pipe.fit(x,y)
def innings_progression(match,first,second):
    match_df_2=innings_2[innings_2["Match ID"]==match]
    match_df_1=innings_1[innings_1["Match ID"]==match]
    target=match_df_2.iloc[0]["current score"]
    match_df_2["Current score"]=match_df_2["current score"]-target
    match_df_2.drop(columns=["current score"],inplace=True)
    match_df_2.rename(columns={"Current score":"current score"},inplace=True)
    match_df_2["balls"]=(match_df_2["Over"]-1)*6+match_df_2["Ball"]
    match_df_2.drop(columns=["Over","Ball","Innings"],inplace=True)
    match_df_1["balls"]=(match_df_1["Over"]-1)*6+match_df_1["Ball"]
    match_df_1.drop(columns=["Over","Ball","Innings"],inplace=True)
    match_df_1["innings"]=first
    match_df_2["innings"]=second
    wickets_1=match_df_1[match_df_1["Wicket"]==1]
    wickets_2=match_df_2[match_df_2["Wicket"]==1]
    match_df=pd.concat([match_df_1,match_df_2],ignore_index=True)
    wickets=pd.concat([wickets_1,wickets_2],ignore_index=True)
    fig_prog=px.line(match_df,x="balls",y="current score",color="innings")
    fig_wickets=px.scatter(wickets,x="balls",y="current score",color="innings")
    fig_wickets.update_traces(marker=dict(size=10))
    fig_wickets.update_layout(showlegend=False)
    fig_prog=fig_prog.add_trace(fig_wickets.data[0])
    fig_prog=fig_prog.add_trace(fig_wickets.data[1])
    return fig_prog
def match_progression(match):
    match_df_2=innings_2[innings_2["Match ID"]==match]
    target=match_df_2.iloc[0]["current score"]
    match_df_2["Current score"]=match_df_2["current score"]-target
    match_df_2.drop(columns=["current score"],inplace=True)
    match_df_2.rename(columns={"Current score":"current score"},inplace=True)
    match_df_2["balls"]=(match_df_2["Over"]-1)*6+match_df_2["Ball"]
    match_df_2.drop(columns=["Ball","Innings"],inplace=True)
    match_df=df2.drop(columns=['Ball','Target Score','Date'])
    match_df=match_df[match_df["Match ID"]==match]
    columns=match_df.columns.to_list()
    overs_df=pd.DataFrame(columns=columns)
    initial=match_df.iloc[0].values.tolist()
    initial[5]=0
    initial[7]=120
    initial[8]=10
    overs_df.loc[0]=initial
    cnt=1
    for i in range(5,len(match_df),6):
        overs_df.loc[cnt]=match_df.iloc[i].values.tolist()
        cnt+=1
    over_prog=match_df_2.groupby("Over").sum()[["Runs From Ball","Wicket"]].reset_index()
    wickets=over_prog["Wicket"].to_list()
    wickets=wickets[:-1]
    wickets.insert(0,0)
    over_prog["wickets"]=wickets
    overs_df.drop(columns=["Match ID","Chased Successfully","Over"],inplace=True)
    win,lose=[],[]
    columns=overs_df.columns
    for i in range(len(overs_df)):
        record=pd.DataFrame(overs_df.iloc[i])
        values=[[rec[0] for rec in record.values]]
        x=pd.DataFrame(values,columns=columns)
        x.drop(columns=["Unnamed: 0"],inplace=True)
        probabilities=st.session_state.pipe.predict_proba(x)[0]
        win.append(round(probabilities[1]*100))
        lose.append(round(probabilities[0]*100))
    over_prog["chasing team win"]=win
    over_prog["chasing team lose"]=lose    
    fig_bar=px.bar(over_prog,x="Over",y="Runs From Ball")
    fig_wickets=px.line(over_prog,x="Over",y="wickets",markers=True)
    fig_wickets.update_traces(marker=dict(size=7,color="yellow"),line=dict(color="yellow",width=3))
    fig_win=px.line(over_prog,x="Over",y="chasing team win")
    fig_win.update_traces(line=dict(color="green",width=3))
    fig_lose=px.line(over_prog,x="Over",y="chasing team lose")
    fig_lose.update_traces(line=dict(color="red",width=3))
    fig_wickets=fig_wickets.add_trace(fig_bar.data[0])
    fig_wickets=fig_wickets.add_traces(fig_win.data[0])
    fig_wickets=fig_wickets.add_traces(fig_lose.data[0])
    return fig_wickets
nav=option_menu(menu_title=None,options=["win percentage calculator","match analysis"],orientation="horizontal")
col1,col2=st.columns([1,1])
if nav=="match analysis":
    with col1: first_innings=st.selectbox("Team batting first",options=df2["Bat First"].unique(),index=None)
    with col2: second_innings=st.selectbox("Team batting second",options=df2["Bat Second"].unique(),index=None)
    valid_date=df2[(df2["Bat First"]==first_innings) & (df2["Bat Second"]==second_innings)]
    with col1: date=st.selectbox("Date",options=valid_date["Date"].unique(),index=None)
    valid_venue=valid_date[valid_date["Date"]==date]
    with col2: venue=st.selectbox("Venue",options=valid_venue["Venue"].unique(),index=None)
    option=st.radio("",options=["innings progression(worm)","match progression (win probability analysis)"],horizontal=True)    
    analyze=st.button("Analyze")
    if analyze and date and first_innings and second_innings and venue:
        match_id=df2[(df2["Date"]==date) & (df2["Bat First"]==first_innings) & (df2["Bat Second"]==second_innings) & (df2["Venue"]==venue)]["Match ID"].unique()[0]
        if option=="innings progression(worm)": fig=innings_progression(match_id,first_innings,second_innings)
        else: fig=match_progression(match_id)
        st.plotly_chart(fig)
else:
    disable=False
    with col1: first_innings=st.selectbox("Team batting first",options=df2["Bat First"].unique(),index=None)
    with col2: second_innings=st.selectbox("Team batting second",options=df2["Bat Second"].unique(),index=None)
    with col1: venue=st.selectbox("Venue",options=df2["Venue"].unique(),index=None)
    with col2: target=st.number_input("Target",min_value=0,value=None)
    with col1: runs=st.number_input("Runs Scored",min_value=0,value=None)
    if target and runs and target<=runs:
        disable=True
        st.error("Runs scored cannot be greater than target.")
    else: disable=False
    with col2: overs=st.text_input("Overs Completed",value=None,disabled=disable)
    if overs and float(overs)>20.0:
        disable=True
        st.error("Overs cannot be greater that 20.")
    else: disable=False
    with col1: wicket=st.number_input("Wickets Fallen",min_value=0,max_value=10,disabled=disable)
    if wicket>=10:
        disable=True
        st.error("More than 10 wickets cannot fall.")
    else: disable=False
    balls=0
    if overs:
        if "." in overs:
            balls=int(overs.split(".")[1])
            overs=overs.split(".")[0]
        balls+=int(overs)*6
        wickets=10-wicket
        balls_left=120-balls
        if overs!="0": crr=round(runs*6/balls,2)
        else: crr=0
        rrr=round((target-runs)*6/balls_left,2)
        record=[venue,first_innings,second_innings,target-runs,balls_left,wickets,crr,rrr]
        user_data=pd.DataFrame([record],columns=['Venue', 'Bat First', 'Bat Second', 'runs left', 'balls left','wickets left', 'crr', 'rrr'])
    chk=(not disable) and first_innings and second_innings and venue and target and (runs!=None) and overs and wickets
    col1,col2,col3,col4,col5=st.columns([1,1,1,1,1])
    with col3: predict=st.button("Predict",disabled=(not chk))
    if predict:
        lose,win=st.session_state.pipe.predict_proba(user_data)[0]
        col1,col2,col3=st.columns([1,2,1])
        with col2: st.header(f"Win Probability")
        st.header(first_innings+":  "+str(round(lose*100))+"%")
        st.progress(round(lose*100),"")
        st.header(second_innings+":  "+str(round(win*100))+"%")
        st.progress(round(win*100),"")
