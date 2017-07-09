package bigdata.science.bayestext.sprk

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import scala.io.Source
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Tokenizer
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.classification.NaiveBayes

class BayesTextWithSpark {

  val hdfs = FileSystem.get(new Configuration)
  
  var categories: Map[Int, String] = null;
  var stopwords: ArrayBuffer[String] = null;
  var model: NaiveBayesModel = null
  
  def buildNaiveBayesModel(trainingData: DataFrame): Unit = {
    model = new NaiveBayes().fit(trainingData)
  }
  
  def buildFeaturesAndLabels(sentenceData: DataFrame): DataFrame = {
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10000)

    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
    
    rescaledData
  }
  
  def buildLabelAndSentencesDF(spark: SparkSession, categoryrootdir: String): DataFrame = {
    val datafiles = listDataFilesByCategory(categoryrootdir)
    var sentenceData = spark.createDataFrame(Seq.empty[(Int, String)]).toDF("label", "sentence")      
    for(catkey <- datafiles.keys) {
      val outrow = for(datfile <- datafiles(catkey)) yield {
        val outtext = spark.read.textFile(datfile).collect().mkString(" ")
        (catkey, outtext)
      }
      sentenceData = sentenceData.union(spark.createDataFrame(outrow.toSeq).toDF("label", "sentence"))
    }
    
    sentenceData
  }
  
  def listDataFilesByCategory(categoryrootdir: String): Map[Int, Array[String]] = {
    val retmap = Map.empty[Int, Array[String]]
    val givencats = hdfs.listStatus(new Path(categoryrootdir))
      .filter(f=>f.isDirectory()).map(f=>f.getPath.getName)
    val revcategories = categories.map(_.swap) //use reverse of category
    for(category <- givencats) {      
      val datafiles = hdfs.listStatus( new Path(categoryrootdir + category) )
        .map(f=>(categoryrootdir+category+"/"+f.getPath.getName) )
        retmap += (revcategories(category) -> datafiles)
    }
    retmap
  }
  
  def buildStopWords(stopwordslocalfile: String): Unit = {
    val filesrc = Source.fromFile(stopwordslocalfile)
    for(ln <- filesrc.getLines()) {
      stopwords += ln.trim()
    }
    filesrc.close 
  }
  
  def buildCategories(trainingdir: String): Unit = {    
    val lsWithIndex = hdfs.listStatus(new Path(trainingdir))
    .map(f => { if(f.isDirectory()) f.getPath.getName else "" } )
    .filter(_.length > 0).zipWithIndex
//    categories = Map[Int, String]() ++= lsWithIndex.toMap[String, Int].map(_.swap)    
    categories = Map(lsWithIndex.toMap[String, Int].map(_.swap).toSeq: _*)
  }
}