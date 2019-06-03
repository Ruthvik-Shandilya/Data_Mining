import pandas as pd
from sklearn.model_selection import train_test_split

data_read = pd.read_csv(r"Dataset_preprocessed.csv")

df = data_read[['reviews.doRecommend','polarity', 'reviews.rating']]
#df = data_read[["reviews.doRecommend","positive_words","negative_words","reviews.rating"]]


train, test = train_test_split(df,test_size=0.2,random_state=1)




df1 = pd.DataFrame(train)
# df["reviews","labels","ratings","Product","Positive_words","Negative_words","Combined_words"]=X
# df["labels"]=Y_train
# df["ratings"]=Z_train
# df["Product"]=P_train
# df["Positive_words"]=R_train
# df["Negative_words"]=S_train
# df["Combined_words"]=Q_train


df1.to_csv("train_dataset.csv",index=False)
#
df2 = pd.DataFrame(test)
# df["reviews","labels","ratings","Product","Positive_words","Negative_words","Combined_words"]=X_test
# df["labels"]=Y_test
# df["ratings"]=Z_test
# df["Product"]=P_test
# df["Positive_words"]=R_test
# df["Negative_words"]=S_test
# df["Combined_words"]=Q_test

df2.to_csv("test_dataset.csv",index=False)