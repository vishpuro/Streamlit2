import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

####--- Sklearn ---####
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.metrics import silhouette_score

# Basic webpage setup
st.set_page_config(page_title="Course Recommender System",layout="wide",initial_sidebar_state="expanded")


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
	return backend.load_ratings()


@st.cache_data
def load_course_sims():
	return backend.load_course_sims()


@st.cache_data
def load_courses():
	return backend.load_courses()


@st.cache_data
def load_bow():
	return backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():

	with st.spinner('Loading datasets...'):
		ratings_df = load_ratings()
		sim_df = load_course_sims()
		course_df = load_courses()
		course_bow_df = load_bow()

	# Select courses
	st.success('Datasets loaded successfully...')
	st.markdown("""---""")
	st.subheader("Select courses that you have audited or completed: ")

	# Build an interactive table for `course_df`
	gb = GridOptionsBuilder.from_dataframe(course_df)
	gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
	gb.configure_selection(selection_mode="multiple", use_checkbox=True)
	gb.configure_side_bar()
	grid_options = gb.build()
	
	# Create a grid response
	response = AgGrid(
		course_df,
		gridOptions=grid_options,
		enable_enterprise_modules=True,
		update_mode=GridUpdateMode.MODEL_CHANGED,
		data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
		fit_columns_on_grid_load=False,)
	
	results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
	results = results[['COURSE_ID', 'TITLE']]
	st.subheader("Your courses: ")
	st.table(results)
	return results


def train(model_name, params):

	if model_name == backend.models[0]:
	# Start training course similarity model
		with st.spinner('Training...'):
			time.sleep(0.5)
			backend.train(model_name)
		st.success('Done!')
	# TODO: Add other model training code here
	#elif model_name == backend.models[1]:
	#	pass
	else:
		pass


def predict(model_name, user_ids, params):
	res = None
	# Start making predictions based on model name, test user ids, and parameters
	with st.spinner('Generating course recommendations: '):
		time.sleep(0.5)
		res = backend.predict(model_name, user_ids, params)
	st.success('Recommendations generated!')
	return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
	"Select model:",
	backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
	# Add a slide bar for selecting top courses
	top_courses = st.sidebar.slider('Top courses',min_value=0, max_value=100,value=10, step=1)
	# Add a slide bar for choosing similarity threshold
	course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',min_value=0, max_value=100,value=50, step=10)
	params['top_courses'] = top_courses
	params['sim_threshold'] = course_sim_threshold
	# TODO: Add hyper-parameters for other models
	# User profile model
elif model_selection == backend.models[1]:
	profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',min_value=0, max_value=100,value=1, step=1)
	params['profile_sim_threshold'] = profile_sim_threshold
	# Clustering model
elif (model_selection == backend.models[2])or(model_selection == backend.models[3]):
	n_enrollments = st.sidebar.slider('Number of enrollments to count as a popular course',min_value=10, max_value=200,value=100, step=10)
	params['n_enrollments'] = n_enrollments

elif model_selection == backend.models[4]:
	k_max = st.sidebar.slider('Define the K in K-Nearest Neighbours',min_value=10, max_value=100,value=40, step=10)
	params['k_max'] = k_max
	
	options1 = ["cosine", "pearson", "msd"]
	selection1 = st.sidebar.segmented_control("Similarity Measure", options1, selection_mode="single",default="cosine")
	st.sidebar.markdown(f"Your selected option: {selection1}.")
	params['similarity_measure'] = selection1

	options2 = ["True", "False"]
	selection2 = st.sidebar.segmented_control("User Based?", options2, selection_mode="single",default="False")
	st.sidebar.markdown(f"Your selected option: {selection2}.")
	params['user_based'] = selection2

elif model_selection == backend.models[5]:
	n_factors = st.sidebar.slider('Number of latent factors',min_value=10, max_value=200,value=10, step=10)
	params['n_factors'] = n_factors

elif model_selection == backend.models[6]:
	embedding_size = st.sidebar.slider('Embedding size',min_value=16, max_value=128,value=16, step=8)
	params['embedding_size'] = embedding_size
	
elif (model_selection == backend.models[7]) and ('user_latent_features' in st.session_state) and ('item_latent_features' in st.session_state):
	options_reg = ["Ridge", "Lasso", "ElasticNet"]
	selection_reg = st.sidebar.segmented_control("Regression Algorithm", options_reg, selection_mode="single",default="Ridge")
	st.sidebar.markdown(f"Your selected option: {selection_reg}.")
	params['reg_type'] = selection_reg
	params['user_latent_features'] = st.session_state.user_latent_features
	params['item_latent_features'] = st.session_state.user_latent_features
	
elif (model_selection == backend.models[8]) and ('user_latent_features' in st.session_state) and ('item_latent_features' in st.session_state):
	options_clas = ["Logistic", "Xgboost"]
	selection_clas = st.sidebar.segmented_control("Classification Algorithm", options_clas, selection_mode="single",default="Logistic")
	st.sidebar.markdown(f"Your selected option: {selection_clas}.")
	params['clas_type'] = selection_clas
	params['user_latent_features'] = st.session_state.user_latent_features
	params['item_latent_features'] = st.session_state.user_latent_features

# Prediction
st.sidebar.subheader('3. Train model and predict results')

# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")

if pred_button and selected_courses_df.shape[0] > 0:
	# Create a new id for current user session
    
	# Start training process
	new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    
	#train(model_selection, params) 
    
	user_ids = [new_id]
	
	if model_selection == backend.models[6]:
		res_df,user_latent_features,item_latent_features = predict(model_selection, user_ids, params)
		st.session_state['user_latent_features'] = user_latent_features
		st.session_state['item_latent_features'] = item_latent_features
		
	elif (model_selection == backend.models[7]) or (model_selection == backend.models[8]):
		if ('user_latent_features' not in st.session_state) or ('item_latent_features' not in st.session_state):
			st.write("Embeddings not found \nPlease run Neural Network model to generate Embeddings")
		else:
			res_df = predict(model_selection, user_ids, params)
	else:
		res_df = predict(model_selection, user_ids, params)
	
	if (model_selection == backend.models[2])or(model_selection == backend.models[3]):
		res_df = res_df[['COURSE_ID']]
	else:
		res_df = res_df[['COURSE_ID', 'SCORE']]
		
	course_df = load_courses()
	res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
	st.dataframe(res_df)

