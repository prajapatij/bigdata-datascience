package bigdata.science.bayestext.sprk

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.Row
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.LongType
import scala.collection.mutable.WrappedArray


object NaiveBayesExample {
  
  def hashingTFTest(spark: SparkSession) = {
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I A B heard about Spark \n Hi I A B heard about Spark \r Hi I A B heard about Spark "),
      (0.0, "I A B me I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat"))).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    wordsData.printSchema()
    wordsData.select("words").foreach(ws=>println(ws.getAs[WrappedArray[String]](0).length) )

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.printSchema()
    featurizedData.show()
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
    rescaledData.select("features").collect().foreach(println)
  }
  
  def countVectorizerTest2(spark: SparkSession) = {
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "Hi I I I I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat"),
      (1.0, "Logistic regression models are neat"))).toDF("label", "sentence")

    val schema = StructType(Seq(StructField("id", LongType, true), StructField("words", ArrayType(StringType, true), true)))

    val rnd = new scala.util.Random
    val df = sentenceData.select("sentence")
      .map(col => { Row(rnd.nextLong, col.getString(0).split("\\s+")) })(RowEncoder(schema))

    df.printSchema()

    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
//      .setVocabSize(1000)
      .setMinDF(2)
      .fit(df)

    cvModel.transform(df).show(false);
  }
  
  def countVectorizerTest(spark: SparkSession) = {
    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a")))).toDF("id", "words")

      df.printSchema()
    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

    // alternatively, define CountVectorizerModel with a-priori vocabulary
    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")

    cvModel.transform(df).show(false)
  }
  
  def naiveBayesTest(spark: SparkSession) = {
     // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("/bigdata/data/science/bayestext/sample_libsvm_data.txt")
    data.printSchema()
    
    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .fit(trainingData)

    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + (accuracy * 100))
  }
  
   def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("NaiveBayesExample")
      .getOrCreate()

    naiveBayesTest(spark)
    hashingTFTest(spark)
    countVectorizerTest(spark)
    countVectorizerTest2(spark)

    spark.stop()
  }
}