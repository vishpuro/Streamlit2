import time
import streamlit as st

####--- Surprise ---####
from surprise.dataset import DatasetAutoFolds
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

####--- Sklearn ---####
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

####--- Tensorflow ---####
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
import numpy as np


models = ("Course Similarity",
	"User Profile",
	"Clustering",
	"Clustering with PCA",
	"KNN",
	"NMF",
	"Neural Network",
	"Regression with Embedding Features",
	"Classification with Embedding Features")


def load_ratings():
	return pd.read_csv("ratings.csv")

def load_course_sims():
	return pd.read_csv("sim.csv")


def load_courses():
	df = pd.read_csv("course_processed.csv")
	df['TITLE'] = df['TITLE'].str.title()
	return df


def load_bow():
	return pd.read_csv("courses_bows.csv")

def load_course_genres():
	return pd.read_csv("course_genres.csv")

def load_user_profiles():
	return pd.read_csv("user_profiles.csv")


def add_new_ratings(new_courses):
	res_dict = {}
	if len(new_courses) > 0:
		# Create a new user id, max id + 1
		ratings_df = load_ratings()
		new_id = ratings_df['user'].max() + 1
		users = [new_id] * len(new_courses)
		ratings = [4.0] * len(new_courses)
		res_dict['user'] = users
		res_dict['item'] = new_courses
		res_dict['rating'] = ratings
		new_df = pd.DataFrame(res_dict)
		updated_ratings = pd.concat([ratings_df, new_df])
		updated_ratings.to_csv("ratings.csv", index=False)
	return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
	bow_df = load_bow()
	grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
	idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
	id_idx_dict = {v: k for k, v in idx_id_dict.items()}
	del grouped_df
	return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
	all_courses = set(idx_id_dict.values())
	unselected_course_ids = all_courses.difference(enrolled_course_ids)
	# Create a dictionary to store your recommendation results
	res = {}
	# First find all enrolled courses for user
	for enrolled_course in enrolled_course_ids:
		for unselect_course in unselected_course_ids:
			if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
				idx1 = id_idx_dict[enrolled_course]
				idx2 = id_idx_dict[unselect_course]
				sim = sim_matrix[idx1][idx2]
				if unselect_course not in res:
					res[unselect_course] = sim
				else:
					if sim >= res[unselect_course]:
						res[unselect_course] = sim
	res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
	return res



def profile_generate_recommendation_scores(user_id,unknown_courses,user_profile_df,course_genres_df):
	"""
	Generate recommendation scores for users and courses.

	Returns:
	users (list): List of user IDs.
	courses (list): List of recommended course IDs.
	scores (list): List of recommendation scores.
	"""

	users = []      # List to store user IDs
	courses = []    # List to store recommended course IDs
	scores = []     # List to store recommendation scores

	# Iterate over each user ID in the test_user_ids list

	# Get the user profile data for the current user
	test_user_profile = user_profile_df[user_profile_df['user'] == user_id]

	# Get the user vector for the current user id (replace with your method to obtain the user vector)
	test_user_vector = test_user_profile.iloc[0, 1:].values


	# Filter the course_genres_df to include only unknown courses
	unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
	unknown_course_ids = unknown_course_df['COURSE_ID'].values

	# Calculate the recommendation scores using dot product
	recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)

	# Append the results into the users, courses, and scores list
	for i in range(0, len(unknown_course_ids)):
		score = recommendation_scores[i]
		users.append(user_id)
		courses.append(unknown_course_ids[i])
		scores.append(recommendation_scores[i])

	return users, courses, scores

def combine_cluster_labels(user_ids, labels):
	# Convert labels to a DataFrame
	labels_df = pd.DataFrame(labels)    
	# Merge user_ids DataFrame with labels DataFrame based on index
	cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
	# Rename columns to 'user' and 'cluster'
	cluster_df.columns = ['user', 'cluster']
	return cluster_df



class RecommenderNet(keras.Model):
	"""
	Neural network model for recommendation.
	
	This model learns embeddings for users and items, and computes the dot product
	of the user and item embeddings to predict ratings or preferences.
	
	Attributes:
	- num_users (int): Number of users.
	- num_items (int): Number of items.
	- embedding_size (int): Size of embedding vectors for users and items.
	"""
	def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
		"""
		Constructor.
		
		Args:
		- num_users (int): Number of users.
		- num_items (int): Number of items.
		- embedding_size (int): Size of embedding vectors for users and items.
		"""
		super(RecommenderNet, self).__init__(**kwargs)
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_size = embedding_size
		
		# Define a user_embedding vector
		# Input dimension is the num_users
		# Output dimension is the embedding size
		# A name for the layer, which helps in identifying the layer within the model.
		
		self.user_embedding_layer = layers.Embedding(
			input_dim=num_users,
			output_dim=embedding_size,
			name='user_embedding_layer',
			embeddings_initializer="he_normal",
			embeddings_regularizer=keras.regularizers.l2(1e-6),
			)
		self.user_dense_layer = layers.Dense(
			units=32,
			activation='linear',
			name='item_dense_layer',
			kernel_initializer="he_normal",
			activity_regularizer=keras.regularizers.l2(1e-6),
			)
		# Define a user bias layer
		# Bias is applied per user, hence output_dim is set to 1.
		self.user_bias = layers.Embedding(
			input_dim=num_users,
			output_dim=1,
			name="user_bias")
		
		# Define an item_embedding vector
		# Input dimension is the num_items
		# Output dimension is the embedding size
		self.item_embedding_layer = layers.Embedding(
			input_dim=num_items,
			output_dim=embedding_size,
			name='item_embedding_layer',
			embeddings_initializer="he_normal",
			embeddings_regularizer=keras.regularizers.l2(1e-6),
			)
		self.item_dense_layer = layers.Dense(
			units=32,
			activation='linear',
			name='item_dense_layer',
			kernel_initializer="he_normal",
			activity_regularizer=keras.regularizers.l2(1e-6),
			)
		# Define an item bias layer
		# Bias is applied per item, hence output_dim is set to 1.
		self.item_bias = layers.Embedding(
			input_dim=num_items,
			output_dim=1,
			name="item_bias")
	
	def call(self, inputs):
		"""
		Method called during model fitting.
		
		Args:
		- inputs (tf.Tensor): Input tensor containing user and item one-hot vectors.
		
		Returns:
		- tf.Tensor: Output tensor containing predictions.
		"""
		# Compute the user embedding vector
		user_vector = self.user_embedding_layer(inputs[:, 0])
		user_vector = self.user_dense_layer(user_vector)
		user_vector = self.user_dense_layer(user_vector)
		# Compute the user bias
		user_bias = self.user_bias(inputs[:, 0])
		# Compute the item embedding vector
		item_vector = self.item_embedding_layer(inputs[:, 1])
		item_vector = self.item_dense_layer(item_vector)
		item_vector = self.item_dense_layer(item_vector)
		# Compute the item bias
		item_bias = self.item_bias(inputs[:, 1])
		# Compute dot product of user and item embeddings
		dot_user_item = tf.tensordot(user_vector, item_vector, 2)
		# Add all the components (including bias)
		x = dot_user_item + user_bias + item_bias
		# Apply ReLU activation function
		return tf.nn.sigmoid(x)

def process_dataset(raw_data):
	"""
	Preprocesses the raw dataset by encoding user and item IDs to indices.
	
	Args:
	- raw_data (DataFrame): Raw dataset containing user, item, and rating information.
	
	Returns:
	- encoded_data (DataFrame): Processed dataset with user and item IDs encoded as indices.
	- user_idx2id_dict (dict): Dictionary mapping user indices to original user IDs.
	- course_idx2id_dict (dict): Dictionary mapping item indices to original item IDs.
	"""
	
	encoded_data = raw_data.copy() # Make a copy of the raw dataset to avoid modifying the original data.
	
	# Mapping user ids to indices
	user_list = encoded_data["user"].unique().tolist() # Get unique user IDs from the dataset.
	user_id2idx_dict = {x: i for i, x in enumerate(user_list)} # Create a dictionary mapping user IDs to indices.
	user_idx2id_dict = {i: x for i, x in enumerate(user_list)} # Create a dictionary mapping user indices back to original user IDs.
	
	# Mapping course ids to indices
	course_list = encoded_data["item"].unique().tolist() # Get unique item (course) IDs from the dataset.
	course_id2idx_dict = {x: i for i, x in enumerate(course_list)} # Create a dictionary mapping item IDs to indices.
	course_idx2id_dict = {i: x for i, x in enumerate(course_list)} # Create a dictionary mapping item indices back to original item IDs.
	
	# Convert original user ids to idx
	encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
	# Convert original course ids to idx
	encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
	# Convert rating to int
	encoded_data["rating"] = encoded_data["rating"].values.astype("int")
	
	return encoded_data, user_idx2id_dict, course_idx2id_dict # Return the processed dataset and dictionaries mapping indices to original IDs.

def generate_train_test_datasets(dataset, scale=True):
	"""
	Splits the dataset into training, validation, and testing sets.
	
	Args:
	- dataset (DataFrame): Dataset containing user, item, and rating information.
	- scale (bool): Indicates whether to scale the ratings between 0 and 1. Default is True.
	
	Returns:
	- x_train (array): Features for training set.
	- x_val (array): Features for validation set.
	- x_test (array): Features for testing set.
	- y_train (array): Labels for training set.
	- y_val (array): Labels for validation set.
	- y_test (array): Labels for testing set.
	"""
	
	min_rating = min(dataset["rating"]) # Get the minimum rating from the dataset
	max_rating = max(dataset["rating"]) # Get the maximum rating from the dataset
	
	dataset = dataset.sample(frac=1, random_state=42) # Shuffle the dataset to ensure randomness
	x = dataset[["user", "item"]].values # Extract features (user and item indices) from the dataset
	if scale:
		# Scale the ratings between 0 and 1 if scale=True
		y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
	else:
		# Otherwise, use raw ratings
		y = dataset["rating"].values
	
	# Assuming training on 80% of the data and testing on 10% of the data
	train_indices = int(0.8 * dataset.shape[0])
	test_indices = int(0.9 * dataset.shape[0])
	# Assigning subsets of features and labels for each set
	x_train, x_val, x_test, y_train, y_val, y_test = (
		x[:train_indices], # Training features
		x[train_indices:test_indices], # Validation features
		x[test_indices:], # Testing features
		y[:train_indices], # Training labels
		y[train_indices:test_indices], # Validation labels
		y[test_indices:], # Testing labels
		)
	return x_train, x_val, x_test, y_train, y_val, y_test # Return the training, validation, and testing sets


# Model training
def train(model_name, params):
	# TODO: Add model training code here
	if model_name==models[4]:
		reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(3, 5))
		course_dataset = Dataset.load_from_file("ratings.csv", reader=reader)
		trainset=DatasetAutoFolds.build_full_trainset(course_dataset)
		model=KNNBasic()
		model.fit(trainset)

	else:
		pass


# Prediction
def predict(model_name, user_ids, params):
	sim_threshold = 0.6
	k_max=40
	profile_sim_threshold=1
	n_erollments=100
	embedding_size=16
	if "sim_threshold" in params:
		sim_threshold = params["sim_threshold"] / 100.0
	if "k_max" in params:
		k=params["k_max"]
	if "similarity_measure" in params:
		similarity_measure = params["similarity_measure"]
	if "user_based" in params:
		if params["user_based"]=='True':
			user_based=True
		else:
			user_based=False
	if "profile_sim_threshold" in params:
		profile_sim_threshold = params["profile_sim_threshold"]
	if "n_erollments" in params:
		n_erollments = params["n_erollments"]
	if "embedding_size" in params:
		embedding_size = params["embedding_size"]
	idx_id_dict, id_idx_dict = get_doc_dicts()
	sim_matrix = load_course_sims().to_numpy()
	users = []
	courses = []
	scores = []
	res_dict = {}
	
	for user_id in user_ids:
	
		######################################################### model 0 Course Similarity ####################################################
		if model_name == models[0]:
			ratings_df = load_ratings()
			user_ratings = ratings_df[ratings_df['user'] == user_id]
			enrolled_course_ids = user_ratings['item'].to_list()
			res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
			for key, score in res.items():
				if score >= sim_threshold:
					users.append(user_id)
					courses.append(key)
					scores.append(score)
			
			res_dict['USER'] = users
			res_dict['COURSE_ID'] = courses
			res_dict['SCORE'] = scores
			res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
		######################################################### model 1 User profile #########################################################
		if model_name == models[1]:
			ratings_df = load_ratings()
			course_genres_df = load_course_genres()
			user_ratings = ratings_df[ratings_df['user'] == user_id]
			enrolled_course_ids = user_ratings['item'].to_list()
			all_courses = set(course_genres_df['COURSE_ID'].values)
			unknown_courses = all_courses.difference(enrolled_course_ids)
			
			add_items={}
			add_items['user']=np.zeros(len(course_genres_df.COURSE_ID))+ratings_df.user.max()+1
			add_items['item']=course_genres_df.COURSE_ID
			add_items['rating']=np.zeros(len(course_genres_df.COURSE_ID))+4
			
			ratings_df=pd.concat([ratings_df,pd.DataFrame(add_items)])
			ratings_ordered_df=ratings_df.pivot(index='user',columns='item',values='rating').fillna(0).loc[:,course_genres_df.COURSE_ID]
			user_profile_df=pd.DataFrame(np.dot(ratings_ordered_df,course_genres_df.iloc[:,2:]),index=ratings_ordered_df.index,columns=course_genres_df.iloc[:,2:].columns)
			user_profile_df.drop(ratings_df.user.max(),inplace=True)
			user_profile_df.reset_index(inplace=True)
			user_profile_df.to_csv("user_profiles.csv")
			
			users,courses,scores = profile_generate_recommendation_scores(user_id,unknown_courses,user_profile_df,course_genres_df)
			res_dict['USER'] = users
			res_dict['COURSE_ID'] = courses
			res_dict['SCORE'] = scores
			res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
			res_df = res_df[res_df['SCORE']>=profile_sim_threshold].sort_values(by='SCORE',ascending=False)
		######################################################### model 2 Clustering ###########################################################
		if model_name == models[2]:
			with st.status("Starting Clustering model...", expanded=True):
				ratings_df = load_ratings()
				course_genres_df = load_course_genres()
				user_ratings = ratings_df[ratings_df['user'] == user_id]
				enrolled_course_ids = user_ratings['item'].to_list()
				all_courses = set(course_genres_df['COURSE_ID'].values)
				unknown_courses = all_courses.difference(enrolled_course_ids)
				
				add_items={}
				add_items['user']=np.zeros(len(course_genres_df.COURSE_ID))+ratings_df.user.max()+1
				add_items['item']=course_genres_df.COURSE_ID
				add_items['rating']=np.zeros(len(course_genres_df.COURSE_ID))+4
				
				ratings_df=pd.concat([ratings_df,pd.DataFrame(add_items)])
				ratings_ordered_df=ratings_df.pivot(index='user',columns='item',values='rating').fillna(0).loc[:,course_genres_df.COURSE_ID]
				user_profile_df=pd.DataFrame(np.dot(ratings_ordered_df,course_genres_df.iloc[:,2:]),index=ratings_ordered_df.index,columns=course_genres_df.iloc[:,2:].columns)
				user_profile_df.drop(ratings_df.user.max(),inplace=True)
				user_profile_df.reset_index(inplace=True)
				user_profile_df.to_csv("user_profiles.csv")
				
				st.write("Scaling Data...")
				
				scaler = StandardScaler()
				# Standardizing the selected features (feature_names) in the user_profile_df DataFrame
				features = scaler.fit_transform(user_profile_df.iloc[:,1:])
				
				st.write("Fitting KMeans...")
				
				inertia=[]
				silhouette=[]
				progress_text = "Fitting..."
				my_bar = st.progress(0, text=progress_text)
				for i in range(1,30,1):
					model=KMeans(n_clusters=i).fit(features)
					inertia.append(model.inertia_)
					labels=model.predict(features)
					if len(np.unique(labels))>=2:
						silhouette.append(silhouette_score(features, labels=labels))
					else:
						silhouette.append(np.nan)
					my_bar.progress(i/30, text="Fitting k: "+str(i)+" of 30")
				my_bar.empty()
				
				n_clust=np.argmax(silhouette[3:])+1
				
				st.write("Best number of clusters: ",n_clust)
				st.write("Refitting...")
				
				model=KMeans(n_clusters=n_clust).fit(features)
				cluster_labels = model.labels_
				cluster_orig=combine_cluster_labels(user_profile_df.user, cluster_labels)
				test_users_labelled_orig = pd.merge(ratings_df, cluster_orig, left_on='user', right_on='user')
				courses_cluster = test_users_labelled_orig[['item', 'cluster']]
				courses_cluster['count'] = [1] * len(courses_cluster)
				courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()
				user_labels=test_users_labelled_orig[['user','cluster']].groupby(by='user').mean()
				
				## - First get all courses belonging to the same cluster and figure out what are the popular ones (such as course enrollments beyond a threshold like 100)
				cluster_courses=[courses_cluster_grouped[(courses_cluster_grouped['cluster']==i)&(courses_cluster_grouped['enrollments']>=n_erollments)].sort_values(by='enrollments',ascending=False)['item'].tolist() for i in range(courses_cluster_grouped['cluster'].max()+1)]
				
				st.write("Outputing results...")
				
				users = []
				courses = []
				for user in user_labels.index:
					cluster = int(user_labels.loc[user].tolist()[0])
					courses_in_cluster = cluster_courses[cluster]
					user_courses = test_users_labelled_orig[(test_users_labelled_orig['user']==user)]['item'].tolist()
					recommended_courses = list(set(courses_in_cluster)-set(user_courses))
					
					for rc in recommended_courses:
						users.append(user)
						courses.append(rc)
				
				res_dict['USER'] = users
				res_dict['COURSE_ID'] = courses
				res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID'])
				res_df=res_df[res_df['USER']==user_id]
		######################################################### model 3 Clustering ###########################################################
		if model_name == models[3]:
			with st.status("Starting Clustering (PCA) model...", expanded=True):
				ratings_df = load_ratings()
				course_genres_df = load_course_genres()
				user_ratings = ratings_df[ratings_df['user'] == user_id]
				enrolled_course_ids = user_ratings['item'].to_list()
				all_courses = set(course_genres_df['COURSE_ID'].values)
				unknown_courses = all_courses.difference(enrolled_course_ids)
				
				add_items={}
				add_items['user']=np.zeros(len(course_genres_df.COURSE_ID))+ratings_df.user.max()+1
				add_items['item']=course_genres_df.COURSE_ID
				add_items['rating']=np.zeros(len(course_genres_df.COURSE_ID))+4
				
				ratings_df=pd.concat([ratings_df,pd.DataFrame(add_items)])
				ratings_ordered_df=ratings_df.pivot(index='user',columns='item',values='rating').fillna(0).loc[:,course_genres_df.COURSE_ID]
				user_profile_df=pd.DataFrame(np.dot(ratings_ordered_df,course_genres_df.iloc[:,2:]),index=ratings_ordered_df.index,columns=course_genres_df.iloc[:,2:].columns)
				user_profile_df.drop(ratings_df.user.max(),inplace=True)
				user_profile_df.reset_index(inplace=True)
				user_profile_df.to_csv("user_profiles.csv")
				
				st.write("Scaling Data...")
				
				scaler = StandardScaler()
				# Standardizing the selected features (feature_names) in the user_profile_df DataFrame
				features = scaler.fit_transform(user_profile_df.iloc[:,1:])
				
				st.write("Calculating PCA Components...")
				
				pca_model=PCA(n_components=14)
				pca_model.fit(features)
				evr=pca_model.explained_variance_ratio_
				evr_cumsum=np.cumsum(evr)
				n_components=sum(evr_cumsum<0.9)+1
				
				st.write("Transforming Data with n_components = ",n_components)
				
				pca_model=PCA(n_components=n_components)
				pca_df=pd.DataFrame(pca_model.fit_transform(features),columns=['PC'+str(i) for i in range(n_components)])
				
				st.write("Fitting Kmeans...")
				
				inertia=[]
				silhouette=[]
				progress_text = "Fitting..."
				my_bar = st.progress(0, text=progress_text)
				for i in range(1,30,1):
					model=KMeans(n_clusters=i).fit(pca_df)
					inertia.append(model.inertia_)
					labels=model.predict(pca_df)
					if len(np.unique(labels))>=2:
						silhouette.append(silhouette_score(pca_df, labels=labels))
					else:
						silhouette.append(np.nan)
					my_bar.progress(i/30, text="Fitting k: "+str(i)+" of 30")
				my_bar.empty()
				
				n_clust=np.argmax(silhouette[3:])+1
				
				st.write("Best number of clusters: ",n_clust)
				st.write("Refitting...")
				
				model=KMeans(n_clusters=n_clust).fit(pca_df)
				cluster_labels = model.labels_
				cluster_pca=combine_cluster_labels(user_profile_df.user, cluster_labels)
				test_users_labelled_pca = pd.merge(ratings_df, cluster_pca, left_on='user', right_on='user')
				courses_cluster = test_users_labelled_pca[['item', 'cluster']]
				courses_cluster['count'] = [1] * len(courses_cluster)
				courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()
				user_labels=test_users_labelled_pca[['user','cluster']].groupby(by='user').mean()
				
				## - First get all courses belonging to the same cluster and figure out what are the popular ones (such as course enrollments beyond a threshold like 100)
				cluster_courses=[courses_cluster_grouped[(courses_cluster_grouped['cluster']==i)&(courses_cluster_grouped['enrollments']>=n_erollments)].sort_values(by='enrollments',ascending=False)['item'].tolist() for i in range(courses_cluster_grouped['cluster'].max()+1)]
				
				st.write("Outputing results...")
				
				users = []
				courses = []
				for user in user_labels.index:
					cluster = int(user_labels.loc[user].tolist()[0])
					courses_in_cluster = cluster_courses[cluster]
					user_courses = test_users_labelled_pca[(test_users_labelled_pca['user']==user)]['item'].tolist()
					recommended_courses = list(set(courses_in_cluster)-set(user_courses))
					
					for rc in recommended_courses:
						users.append(user)
						courses.append(rc)
				
				res_dict['USER'] = users
				res_dict['COURSE_ID'] = courses
				res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID'])
				res_df=res_df[res_df['USER']==user_id]
				
		######################################################### model 4 knn-surprise #############################################            
		if model_name==models[4]:
			with st.status("Starting KNN model...", expanded=True):
				reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(3, 5))
				course_dataset = Dataset.load_from_file("ratings.csv", reader=reader)
				#trainset=DatasetAutoFolds.build_full_trainset(course_dataset)
				trainset, testset = train_test_split(course_dataset, test_size=.95)
				model=KNNBasic(k=k_max,sim_option={"name":similarity_measure,"user_based":user_based})

				st.write("Fitting KNN...")
				
				model.fit(trainset)
				
				ratings_df = load_ratings()
				course_genres_df = load_course_genres()
				user_ratings = ratings_df[ratings_df['user'] == user_id]
				enrolled_course_ids = user_ratings['item'].to_list()
				all_courses = set(course_genres_df['COURSE_ID'].values)
				unknown_courses = list(all_courses.difference(enrolled_course_ids))
				
				test_data= pd.DataFrame({'user':[user_id]*len(unknown_courses),'item':unknown_courses,'rating':[4]*len(unknown_courses)})

				st.write("Predicting Courses...")
				
				for i in range(test_data.shape[0]):
					result=model.predict(uid=test_data.loc[i,'user'],iid=test_data.loc[i,'item'],r_ui=test_data.loc[i,'rating'])
					users.append(int(result.uid))
					courses.append(result.iid)
					scores.append(float(result.est))

				st.write("Outputting...")
				
				res_dict['USER'] = users
				res_dict['COURSE_ID'] = courses
				res_dict['SCORE'] = scores
				res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
				res_df.sort_values(by='SCORE',ascending=False)
		######################################################### model 6 Neural Network#############################################            
		if model_name==models[6]:
			with st.status("Starting Neural Network model...", expanded=True):
				ratings_df = load_ratings()
				course_genres_df = load_course_genres()
				user_ratings = ratings_df[ratings_df['user'] == user_id]
				enrolled_course_ids = user_ratings['item'].to_list()
				all_courses = set(course_genres_df['COURSE_ID'].values)
				unknown_courses = list(all_courses.difference(enrolled_course_ids))
				test_dataset= pd.DataFrame({'user':[user_id]*len(unknown_courses),'item':unknown_courses,'rating':[4]*len(unknown_courses)})
				
				st.write("Encoding data...")
				
				encoded_data, user_idx2id_dict, course_idx2id_dict = process_dataset(ratings_df)
				encoded_data_test, user_idx2id_dict_test, course_idx2id_dict_test = process_dataset(test_dataset)
				x_train, x_val, x_test, y_train, y_val, y_test = generate_train_test_datasets(encoded_data)
				
				num_users = len(ratings_df['user'].unique())
				num_items = len(ratings_df['item'].unique())
				
				model = RecommenderNet(num_users, num_items, embedding_size)
				early_stopping =EarlyStopping(monitor='val_loss', patience=2)
				## - call model.compile() method to set up the loss and optimizer and metrics for the model training, you may use
				
				st.write("Fitting Neural Network...")
				
				model.compile(optimizer=keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.RootMeanSquaredError()])
				history=model.fit(x=x_train, y=y_train,batch_size=64,epochs=10,validation_data=(x_val,y_val),verbose=1,callbacks = [early_stopping,keras.callbacks.ModelCheckpoint("RNN.keras",save_best_only=True)]) 
				#  - -Save the entire model in the SavedModel format and then save only the weights of the model using 
				model.save("recommender_net_model.keras")
				## - - model.save_weights("recommender_net_weights.weights.h5")
				model.save_weights("recommender_net_weights.weights.h5")
				
				st.write("Predicting results...")
				
				test_data=encoded_data_test[encoded_data_test['user']==user_id][['user','item']].to_numpy()
				
				pred=model.predict(test_data)
				pred=(pred*2)+3
				test_dataset.loc[:,'rating']=pred
				res_df=test_dataset
				res_df.sort_values(by='rating',ascending=False)
				res_df.rename(columns={'user':'USER','item':'COURSE_ID','rating':'SCORE'},inplace=True)
		
	return res_df
