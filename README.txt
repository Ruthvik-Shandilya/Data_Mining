Product Recommendation System using Sentiment Analysis

***********************Execution Steps************************************:
1. Preprocessing.py
	Input : Dataset.csv
	Output: Dataset_preprocessed.csv
	
2. Feature_Split.py
	Input : Dataset_preprocessed.csv
	Output: train_dataset.csv and test_dataset.csv

3. svm.py
	Input : train_dataset.csv and test_dataset.csv
	Output: Output_SVM.csv and also Confusion matrix, Accuracy.
	
4. NB.py
	Input : train_dataset.csv and test_dataset.csv
	Output: Confusion matrix and Accuracy.
	
5. KNN.py
	Input : train_dataset.csv and test_dataset.csv
	Output: Output_KNN.csv and also Confusion matrix, Accuracy,(Execution take 20 mins).
	
6. main.py
	Input : Output_KNN.csv
	Output: Product recommendation
	
* We have provided 1429_1.csv,Dataset.csv,Dataset_preprocessed.csv,train_dataset.csv and test_dataset.csv files. After executing SVM.py , NB.py and KNN.py files, we can execute #main.py which uses KNN algorithm to recommend the product.
	
*************************Steps Explanation********************************:


* The dataset for this project : Amazon customer reviews from Kaggle.com

- Initially the dataset "1429_1.csv" consisted of 34,660 rows and after removing the empty columns, it was reduced to 27,854 rows. 
- On careful evaluation of the dataset, we found the dataset consisted of only 1,384 negative reviews and the rest of the rows were all  positive reviews. So , it was #completely positively skewed set.
- In order to train the model accurately, we randomly chose around 1,300 positive reviews and all the negative reviews and created a separate csv file "Dataset.csv".

* PreProcessing.py
* This python file is used to clean the data as well as pre-process. On executing this file:
1. Empty columns are removed.
2. Select only the required columns to train our model.
3. Remove blank Spaces and new line characters.
4. Convert all the words in the reviews text to lower case.
5. Normalize the text.
6. Removes all the special characters such as @,^,* etc.
7. Removes the repeated words.
8. Removal of stop words.

* Feature_Selection:
1. Product Categories
2. Product Name
3. The review title of each product
4. The review text of each product.
5. The rating provided by the customers.
6. The class labels for each product.

* Feature_Extraction:
1. Calculate Polarity and Subjectivity of each review text using TextBlob.
2. Calculate number of positive words and number of negative words in each review using a positive and negative words dictionary.
3. The ratings provided by the customers for each product.
Note: The class labels are provided based on the ratings of the product(rating : 4 or 5 stars is considered as "positive review(1)" and rating : 1,2,3 are considered as "negative review(-1)".

* After processing this file, we are saving all the processed data into "Dataset_preprocessed.csv" file.

* Feature_Split.py
This file splits the dataset into training and test data. We have used "train_test_split" function from sklearn.Since our dataset consists of 2700 rows, we split the data into 80:20 ratio train:test respectively.

* svm.py
This file provides the implementation of SVM algorithm.The output of this file provides the "Confusion matrix" and the "Accuracy" of the algorithm along the "Precision,Recall and F1-score", the performance measures. We save the output of each predicted product into a separate csv file"Output_<algo-name>.csv".

* NB.py
This file provides the implementation of NB algorithm.The output of this file provides the "Confusion matrix" and the "Accuracy" of the algorithm along the "Precision,Recall and F1-score", the performance measures.

* KNN.py
This file provides the implementation of KNN algorithm.The output of this file provides the "Confusion matrix" and the "Accuracy" of the algorithm along the "Precision,Recall and F1-score", the performance measures. We save the output of each predicted product into a separate csv file"Output_<algo-name>.csv".

* main.py
This is the main file where we use it to recommend the product to the customers based on the results from the algorithm which provides maximum accuracy. We have provided three ways of recommending the product:
1. Using PrettyTable, creates a table of all the products along with the count of positive and negative reviews.
2. Using loops, prints all the products, positive and negative recommender's.
3. The customer can input the product name, the output provides the number of customers who recommend this product and who do not recommend this product.
