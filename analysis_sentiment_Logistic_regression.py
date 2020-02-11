from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF,Tokenizer,StopWordsRemover
from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
import pandas as pn

appName= "Sentiment Analysis with logistic regression"
spark=SparkSession.builder.appName(appName).config("spark.some.config.option","file:///C:/temp").getOrCreate()

##conf = SparkConf().setMaster("local[2]").setAppName("")
##sc = SparkContext(conf = conf)
movie_csv=spark.read.csv("movie_comm.csv",inferSchema=True,header=True)

data=movie_csv.select("SentimentText",col("sentiments").cast("Int").alias("class"))

data.show(truncate=False)
data_used=data.randomSplit([0.8,0.2])

(data_train,data_test)=(data_used[0],data_used[1])
tokenizer=Tokenizer(inputCol="SentimentText",outputCol="Sentiment_splited")
token_train=tokenizer.transform(data_train)


stopWords=StopWordsRemover(inputCol=tokenizer.getOutputCol(),outputCol="without_useles_word")
stopWords_train=stopWords.transform(token_train)

hashh=HashingTF(inputCol=stopWords.getOutputCol(),outputCol="feature")
text_to_numeric=hashh.transform(stopWords_train)\
.select("class","without_useles_word","feature")


logic_reg=LogisticRegression(labelCol="class",featuresCol="feature",maxIter=10,regParam=0.03)
training_model=logic_reg.fit(text_to_numeric)

####################################Test############################################################
token_test=tokenizer.transform(data_test)
stopWords_test=stopWords.transform(token_test)
text_to_numeric_test=hashh.transform(stopWords_test).select("class","without_useles_word","feature")
predict=training_model.transform(text_to_numeric_test)
prediction_of_sentiments=predict.select("without_useles_word","prediction","class")
prediction_of_sentiments.show(truncate=False)

##################################rate_correct_answers##############################################
rate_correct_predict=prediction_of_sentiments.filter(prediction_of_sentiments["prediction"]==prediction_of_sentiments["class"]).count()\
/data_test.count()
rate_bad_predict=1-rate_correct_predict
print("rate correct answers {} and rate_bad_answers {}".format(rate_correct_predict,rate_bad_predict))


######################################validation#####################################################
true_positif=prediction_of_sentiments.filter(prediction_of_sentiments["prediction"]==prediction_of_sentiments["class"])\
.filter( prediction_of_sentiments["class"]==1).count()
print("########################{}".format(true_positif))
false_positif=prediction_of_sentiments.filter(prediction_of_sentiments["prediction"]!=prediction_of_sentiments["class"])\
.filter( prediction_of_sentiments["prediction"]==1).count()

true_negatif=prediction_of_sentiments.filter(prediction_of_sentiments["prediction"]==prediction_of_sentiments["class"])\
.filter( prediction_of_sentiments["prediction"]==0).count()

false_negatif=prediction_of_sentiments.filter(prediction_of_sentiments["prediction"]!=prediction_of_sentiments["class"])\
.filter( prediction_of_sentiments["prediction"]==0).count()

rappel=true_positif/(true_positif+false_negatif)

precision=true_positif/(true_positif+false_positif)

print("rappel = {}".format(rappel))
print("precision = {}".format(precision))
