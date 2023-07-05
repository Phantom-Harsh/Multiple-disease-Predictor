#importing libraries
from tensorflow import keras
from keras.models import load_model
from streamlit_option_menu import option_menu
import pickle
import streamlit as st
import io
from PIL import ImageOps, Image
import numpy as np
import joblib
import tensorflow as tf
from streamlit_chat import message
import os
import openai
openai.api_key = "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"

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Heart Disease Prediction',
                            'Hepatitis Mortality Predictor',
                            'Parkinsons Prediction',
                            'Diabetes Prediction',
                            'Plant Diesease Detection',
                            'Pneumonia Detector',
                            'Help - ChatBot'],
                           
                           icons=['heart-pulse', 'snow', 'person',
                                  'stack', 'suit-club', 'lungs', 'robot'],                
                           default_index=0)

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction using ML')
    st.button("About", help="Cardiovascular disease or heart disease describes a range of conditions that affect your heart.\n Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease. \nFrom WHO statistics every year 17.9 million dying from heart disease. \nThe medical study says that human life style is the main reason behind this heart problem. Apart from this there are many key factors which warns that the person may/maynot getting chance of heart disease.\nFrom the dataset if we create suitable machine learning technique which classify the heart disease more accurately, it is very helpful to the health organisation as well as patients.\nDataset used Link : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?datasetId=216167&sortBy=voteCount")

    age = st.slider('age', 29, 77, 40, 1)
    cp = st.slider('cp', 0, 3, 1, 1)
    sex = st.slider('sex', 0, 1, 0, 1, help="0=female,1=male")
    trestbps = st.slider('trestbps', 94, 200, 80, 1)
    chol = st.slider('chol', 126, 564, 246, 2)
    fbs = st.slider('fbs', 0, 1, 0, 1)
    restecg = st.slider('restecg', 0, 2, 1, 1)
    exang = st.slider('exang', 0, 1, 0, 1)
    oldpeak = st.slider('oldpeak', 0.0, 6.2, 3.2, 0.2)
    slope = st.slider('slope', 0, 2, 1, 1)
    ca = st.slider('ca', 0, 4, 2, 1)
    thal = st.slider('thal', 0, 3, 1, 1)
    thalach = st.slider('thalach', 71, 202, 150, 1)

    X_test_sc = [[age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak, slope, ca, thal]]

    load_clf = pickle.load(
        open('heart_disease_model.pkl', 'rb'))
    
    pretty_result = {"age": age, "cp":cp, "sex": sex,"trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg, "exang": exang, "oldpeak": oldpeak, "slope": slope,"ca":ca,"thal":thal,"thalach":thalach}
    st.json(pretty_result)
    
    prediction = load_clf.predict(X_test_sc)
    answer = prediction[0]
    if st.button('Predict'):
        if answer == 0:
            st.success("Heart Disease was Not Detected")
        else:
            st.error("Heart Disease was Detected")
    st.markdown(
        "Note", help="This prediction is based on the Machine Learning Algorithm, Support Vector Machine.")

    #
    # DONE
    #

# Diabetes Disease Prediction Page

if (selected == 'Diabetes Prediction'):
    st.title(
        "Diabetes Risk Prediction for Females")
    st.markdown(
        "About", help="This a Web app that tells you if you are a female whether you are at risk for Diabetes or not.")
    st.header("Just fill in the information below")

    Pregnancies = st.slider("Input Your Number of Pregnancies", 0, 16)
    Glucose = st.slider("Input your Gluclose", 74, 200)
    BloodPressure = st.slider("Input your Blood Pressure", 30, 130)
    SkinThickness = st.slider("Input your Skin thickness", 0, 100)
    Insulin = st.slider("Input your Insulin", 0, 200)
    BMI = st.slider("Input your BMI", 14.0, 60.0)
    DiabetesPedigreeFunction = st.slider(
        "Input your Diabetes Pedigree Function", 0.0, 6.0)
    Age = st.slider("Input your Age", 0, 100)

    inputs = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
               Insulin, BMI, DiabetesPedigreeFunction, Age]]

    model = pickle.load(
        open('model_DIAB.pkl', 'rb'))
    if st.button('Predict'):
        result = model.predict(inputs)
        updated_res = result.flatten().astype(int)
        if updated_res == 0:
            st.success(
                "Unlikely for diabetes, but prioritize self-care nonetheless.")
            st.text("The AdaBoost Classifier was used.")
        else:
            st.warning(
                "Possible diabetes risk, prioritize self-care.")
            st.text("The AdaBoost Classifier was used.")


#
# Done
#

# Parkinsons Disease Prediction Page
if (selected == "Parkinsons Prediction"):
    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4     = st.columns(4)
    with col1:
        fo = st.text_input('MDVP: Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP: Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP: Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP: Jitter(%)')
        
    with col1:
        Jitter_Abs = st.text_input('MDVP: Jitter(Abs)')
    with col2:
        RAP = st.text_input('MDVP: RAP')
    with col3:
        PPQ = st.text_input('MDVP: PPQ')
    with col4:
        DDP = st.text_input('Jitter: DDP')
        
    with col1:
        Shimmer = st.text_input('MDVP: Shimmer')
    with col2:
        Shimmer_dB = st.text_input('MDVP: Shimmer(dB)')
    with col3:
        APQ3 = st.text_input('Shimmer: APQ3')
    with col4:
        APQ5 = st.text_input('Shimmer: APQ5')
        
    with col1:
        APQ = st.text_input('MDVP: APQ')
    with col2:
        DDA = st.text_input('Shimmer: DDA')
    with col3:
        NHR = st.text_input('NHR')
    with col4:
        HNR = st.text_input('HNR')
        
    with col1:
        RPDE = st.text_input('RPDE')
    with col2:
        DFA = st.text_input('DFA')
    with col3:
        spread1 = st.text_input('spread1')
    with col4:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    # creating a button for Prediction

    parkinsons_model = pickle.load(open(
        '/home/phantom/Desktop/Project 1/parkinsons_model.sav', 'rb'))
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict(
            [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if (parkinsons_prediction[0] == 1):
            st.warning("The person has Parkinson's disease")
        else:
            st.success("The person does Not have Parkinson's disease")

#
#  Done
#

# Hepatitis Mortality Predictor Page
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key

def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value

if (selected == 'Hepatitis Mortality Predictor'):
    st.title("Hepatitis Mortality Predictor")

    age = st.number_input("Age", 7, 80)
    gender_dict = {"male": 1, "female": 2}
    sex = st.radio("Sex", tuple(gender_dict.keys()))
    feature_dict = {"No": 1, "Yes": 2}
    steroid = st.radio("Do You Take Steroids?", tuple(feature_dict.keys()))
    antivirals = st.radio("Do You Take Antivirals?",
                          tuple(feature_dict.keys()))
    fatigue = st.radio("Do You Have Fatigue", tuple(feature_dict.keys()))
    spiders = st.radio("Presence of Spider Naeve", tuple(feature_dict.keys()))
    ascites = st.selectbox("Ascities", tuple(feature_dict.keys()))
    varices = st.selectbox("Presence of Varices", tuple(feature_dict.keys()))
    bilirubin = st.number_input("bilirubin Content", 0.0, 8.0)
    alk_phosphate = st.number_input("Alkaline Phosphate Content", 0.0, 296.0)
    sgot = st.number_input("Sgot", 0.0, 648.0)
    albumin = st.number_input("Albumin", 0.0, 6.4)
    protime = st.number_input("Prothrombin Time", 0.0, 100.0)
    histology = st.selectbox("Histology", tuple(feature_dict.keys()))

    feature_list = [age, get_value(sex, gender_dict), get_fvalue(steroid), get_fvalue(antivirals), get_fvalue(fatigue), get_fvalue(
        spiders), get_fvalue(ascites), get_fvalue(varices), bilirubin, alk_phosphate, sgot, albumin, int(protime), get_fvalue(histology)]
    
    pretty_result = {"age": age, "sex": sex, "steroid": steroid, "antivirals": antivirals, "fatigue": fatigue, "spiders": spiders, "ascites": ascites,
                     "varices": varices, "bilirubin": bilirubin, "alk_phosphate": alk_phosphate, "sgot": sgot, "albumin": albumin, "protime": protime, "histolog": histology}
    st.json(pretty_result)
    single_sample = np.array(feature_list).reshape(1, -1)
    
    model_choice = "DecisionTree"    
    loaded_model = joblib.load(
        open('hepB_model.pkl', 'rb'))
    prediction = loaded_model.predict(single_sample)
    pred_prob = loaded_model.predict_proba(single_sample)    

    if st.button("Predict"):
        prediction = loaded_model.predict(single_sample)
        pred_prob = loaded_model.predict_proba(single_sample)
    
        if prediction == 1:
            st.warning("Patient Dies")
            pred_probability_score = {
                "Die": pred_prob[0][0]*100, "Live": pred_prob[0][1]*100}
            st.subheader(
                "Prediction Probability Score using {}".format(model_choice))
            st.json(pred_probability_score)
            st.subheader("Prescriptive Analytics")
        
            prescriptive_message_temp = """
	    <div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		    <h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		    <ul>
		    <li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		    <li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		    <li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		    <li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		    <li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		    <ul>
		    <h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		    <ul>
		    <li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		    <li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		    <li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		    <ul>
	    </div>
	    """
            st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

        else:
            st.success("Patient Lives")
            pred_probability_score = {
                "Die": pred_prob[0][0]*100, "Live": pred_prob[0][1]*100}
            st.subheader(
                "Prediction Probability Score using {}".format(model_choice))
            st.json(pred_probability_score)

#
#  pretty well
#

# Plant Diesease Detection Predictor Page
def load_modell(path):

    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(
            include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(
            include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    return model
#
def clean_image(image):
    image = np.array(image)    
    image = np.array(Image.fromarray(
        image).resize((512, 512), Image.ANTIALIAS))    
    image = image[np.newaxis, :, :, :3]    
    return image

def get_prediction(model, image):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)    
    test = datagen.flow(image)    
    predictions = model.predict(test)
    predictions_arr = np.array(np.argmax(predictions))
    return predictions, predictions_arr

def make_results(predictions, predictions_arr):
    result = {}
    if int(predictions_arr) == 0:
        result = {"status": " is Healthy ",
                  "prediction": f"{int(predictions[0][0].round(2)*100)}%"}
    if int(predictions_arr) == 1:
        result = {"status": ' has Multiple Diseases ',
                  "prediction": f"{int(predictions[0][1].round(2)*100)}%"}
    if int(predictions_arr) == 2:
        result = {"status": ' has Rust ',
                  "prediction": f"{int(predictions[0][2].round(2)*100)}%"}
    if int(predictions_arr) == 3:
        result = {"status": ' has Scab ',
                  "prediction": f"{int(predictions[0][3].round(2)*100)}%"}
    return result


if (selected == "Plant Diesease Detection"):
    st.title('Plant Diesease Detection')
    st.write(
        "Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not")
    filee = st.file_uploader("Choose a Image file", type=["png", "jpg"])
    b = False
    st.write("Take a photo")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("start Camera"):
            b = 0
        else:
            b = 1
    with col2:
        if st.button("stop Camera"):
            b = 1
        else:
            b = 0
                
    camera_photo = st.camera_input("", disabled=b)    
    if filee is None:
        uploaded_file = camera_photo        
    else:
        uploaded_file = filee
    st.image(uploaded_file)

    if uploaded_file != None:        
        progress = st.text("Crunching Image")
        my_bar = st.progress(0)
        i = 0
            
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(np.array(Image.fromarray(
            np.array(image)).resize((700, 400), Image.ANTIALIAS)), width=None)
        my_bar.progress(i + 40)
    
        image = clean_image(image)
        predictions, predictions_arr = get_prediction(model, image)
        my_bar.progress(i + 30)    
        result = make_results(predictions, predictions_arr)
    
        my_bar.progress(i + 30)
        progress.empty()
        i = 0
        my_bar.empty()
        
        st.write(
            f"The plant {result['status']} with {result['prediction']} prediction.")

#
#  half
#

# Help - ChatBot Page
def get_initial_message():
    messages = [
        {"role": "system", "content": "You are a helpful AI Doctor in Healthcare. Who anwers brief questions about AI."},
        {"role": "user", "content": "I want to learn about this disease"},
        {"role": "assistant",
            "content": "Sure, what do you want to know aboout this disease"}
    ]
    return messages

def get_chatgpt_response(messages, model="gpt-3.5-turbo"):
    print("model: ", model)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response['choices'][0]['message']['content']

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

if (selected == 'Help - ChatBot'):
    st.title("Chatbot: ChatGPT & Streamlit Chat")
    st.subheader("AI Doctor:")
    
    model = st.selectbox(
        "Select a model",
        ("gpt-3.5-turbo", "gpt-4")
    )
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    query = st.text_input("Query: ", key="input")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = get_initial_message()

    if query:
        with st.spinner("generating..."):
            messages = st.session_state['messages']
            messages = update_chat(messages, "user", query)
            response = get_chatgpt_response(messages, model)
            messages = update_chat(messages, "assistant", response)
            st.session_state.past.append(query)
            st.session_state.generated.append(response)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state['past'][i],
                        is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
        with st.expander("Show Messages"):
            st.write(messages)

#
# Done but require payment lol
#

# Pneumonia Detector Prediction Page
if (selected == 'Pneumonia Detector'):  
    st.title("Pneumonia Detector")  
    st.button("About", help=" What is Pneumonia? \nPneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.")
    img = st.file_uploader(label="Load X-Ray Chest image", type=['jpeg', 'jpg', 'png'], key="xray")

    if img is not None:
        # Preprocessing Image
        i11 = Image.open(img).convert("RGB")
        p_img = i11.resize((224, 224))
        p_img=np.array(p_img) / 255.0    

        if st.checkbox('Zoom image'):
            image = np.array(Image.open(img))
            st.image(image, use_column_width=True)
        else:
            st.image(p_img)

    # Loading model
        MODEL = "/home/phantom/Desktop/Project 1/pneumonia_classifiers.h5"
        loading_msg = st.empty()
        loading_msg.text("Predicting...")             
        model = keras.models.load_model(f"{MODEL}", compile=True)        

    # Predicting result        
        prediction=1        
        prob = model.predict(np.reshape(p_img, [1, 224, 224, 3]))
        prob = prob.reshape(1,-1)[0]        
        if prob[0] > 0.5:
            prediction = True
        else:
            prediction = False    
        loading_msg.text('')

        if prediction:            
            st.warning("Pneumonia Detected! :slightly_frowning_face")
        else:            
            st.success("No Pneumonia Detected, Healthy! :smile")

        st.text(f"Probability of Pneumonia is {round(prob[0] * 100, 2)}%")
    
