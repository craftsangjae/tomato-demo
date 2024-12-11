import streamlit as st
import pandas as pd
from src.predictor import GrowthPredictor, PlantStat


# TODO: 현재는 Local 파일로 예측하도록 작성. 추후에는 서버로 전송하여 예측하도록 수정 필요
predictor = GrowthPredictor("data/")


st.set_page_config(page_title="완숙토마토 성장 예측 모델", layout="wide")

st.title("완숙토마토 성장 예측 모델")


with st.form("input_form"):
    left_column, right_column = st.columns([1, 1])  # Adjust column width ratios if needed

    with left_column:
        st.subheader("토마토 생육 및 수확 데이터")
        length = st.number_input("초장 길이(cm)", placeholder="해당 식물의 초장 길이를 입력하세요.")
        num_fruit = st.number_input("누적 착과수", placeholder="해당 식물이 착과한 갯수의 총합을 입력하세요.")
        num_harvest = st.number_input("누적 수확수", placeholder="해당 식물로부터 수확한 갯수의 총합을 입력하세요.")
        weight = st.number_input("누적 수확중량", placeholder="해당 식물로부터 수확한 중량의 총합을 입력하세요.")
        avg_weight = weight / num_harvest if num_harvest > 0 else 0.0

    with right_column:
        st.subheader("토마토 생육 및 수확 데이터 입력")
        uploaded_file = st.file_uploader("csv 파일을 업로드하세요", type=["csv"])

    submitted = st.form_submit_button("예측하기")

with st.container():
    st.subheader("예측 결과")
    if submitted and uploaded_file is not None:
        stat = PlantStat(length=length, num_fruit=num_fruit, num_harvest=num_harvest, weight=weight)

        output_df = predictor.predict(stat, uploaded_file)

        length_df = output_df['초장']
        fruit_df = output_df['착과수']
        harvest_df = output_df['누적수확수']
        weight_df = output_df['누적수확중량']
    else:
        length_df = pd.DataFrame()
        fruit_df = pd.DataFrame()
        harvest_df = pd.DataFrame()
        weight_df = pd.DataFrame()

    column1, column2, column3, column4 = st.columns(4)
    with column1:
        st.line_chart(x_label="날짜", y_label="초장 길이(cm)", data=length_df)
    with column2:
        st.line_chart(x_label="날짜", y_label="착과수", data=fruit_df)
    with column3:
        st.line_chart(x_label="날짜", y_label="누적 수확수", data=harvest_df)
    with column4:
        st.line_chart(x_label="날짜", y_label="누적 수확중량", data=weight_df)




