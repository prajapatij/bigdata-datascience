package bigdata.science.bayestext.sprk

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

@RunWith(classOf[JUnitRunner])
class BayesTextWithSparkTest extends FunSuite {
  val baseDirectory = "src/test/data/20news-bydate/"
  val trainingDir = baseDirectory + "20news-bydate-train/"
  val testDir = baseDirectory + "20news-bydate-test/"
  val samplecat = "src/test/data/20news-bydate/20news-bydate-test/misc.forsale/77014"
  
  test("Test first") {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("NaiveBayesExample")
      .getOrCreate()
    
    val testobj = new BayesTextWithSpark
    testobj.buildCategories(trainingDir)
    testobj.categories.foreach(println)    
    val sentenceData = testobj.buildLabelAndSentencesDF(spark, trainingDir)
    println("Sentence data count = " + sentenceData.count)   
    val trainingData = testobj.buildFeaturesAndLabels(sentenceData)
    testobj.buildNaiveBayesModel(trainingData)
    
    val testdata1 = testobj.buildLabelAndSentencesDF(spark, testDir)
    println("Test sentence data count = " + testdata1.count)
    val testdata = testobj.buildFeaturesAndLabels(testdata1)
    val predictions = testobj.model.transform(testdata)
    
      // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + (accuracy*100))
  }
}