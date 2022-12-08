import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, weights_file):
    model = load_model(weights_file)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction.tolist()[0]

def main():

    st.title("Image Classification with Google's Teachable Machine")

    uploaded_file = st.file_uploader("Choose a Image...", type="jpg")

    # 画像がアップロードされた場合...
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # teachable_machine_classification関数に画像を引き渡してクラスを推論する
        prediction = teachable_machine_classification(image, 'keras_model.h5')        
        st.caption(f'推論結果：{prediction}番') # 戻り値の確認（デバッグ用）

        classNo = np.argmax(prediction)          # 一番確率の高いクラス番号を算出
        st.caption(f'判定結果：{classNo}番')      # 戻り値の確認（デバッグ用）

        pred0 = round(prediction[0],3) * 100  # の確率(%)
        pred1 = round(prediction[1],3) * 100  # の確率(%)
        pred2 = round(prediction[2],3) * 100  # の確率(%)

        # 出力結果（分岐）
        if classNo == 0:
            st.subheader(f"これは{pred0}％の確率で「広島カープ」です！")
        elif classNo == 1:
            st.subheader(f"これは{pred1}％の確率で「中央大学」です！")
        else:
            st.subheader(f"これは{pred2}％の確率で「シンシナティ・レッズ」です！")


if __name__ == "__main__":
    main()