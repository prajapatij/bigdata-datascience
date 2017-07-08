package bigdata.science.bayestext

import scala.collection.mutable.Map
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.File

class BayesText {
  val vocabulary = Map.empty[String, Int] // [word, total_word_count]
  val prob = Map.empty[String, Map[String, Double]] // [category, word, word_count]  
  val totals = Map.empty[String, Int] // [category, total_word_count]
  val stopwords = ArrayBuffer.empty[String]
  val categories = ArrayBuffer.empty[String]
  
  def this(trainingdir: String, stopwordsfile: String) = {
    this()
    println("Executing...")
    val filesrc = Source.fromFile(stopwordsfile)
    for(ln <- filesrc.getLines()) {
      stopwords += ln.trim()
    }
    filesrc.close
    
    //get list of categories
    categories ++= (new File(trainingdir)).list().filter(fnm=>new File(trainingdir+File.separator+fnm).isDirectory())

    println("Counting ...")
    
    for(category <- categories) {
      println("          " + category)
      val (counts, total) = train(trainingdir, category)
      prob += (category -> counts)
      totals += (category -> total)
    }
    //remove word from vocabulary which has count <= 3
    for(key <- vocabulary.keys) {
      if(vocabulary(key) <= 3) vocabulary -= key
    }
    
    val vocabsize = vocabulary.size  
    println("Computing probability...")
    for(category <- categories) {
      println("          " + category)
      val denominator = totals(category) + vocabsize
      for(wrdkey <- vocabulary.keys) {
        var count = 1.0
        if(prob(category).contains(wrdkey)) {
          count = prob(category)(wrdkey)
        }
        prob(category).update(wrdkey, ((count + 1)*1.0 / denominator))
      }
    }
    println("DONE TRAINING ")
  }
  
  def train(trainingdir: String, category: String): (Map[String, Double], Int) = {
    val currdir = (trainingdir + category)
    val files = (new File(currdir)).list()
    val counts = Map.empty[String, Double]
    var total = 0
    
    for(file <- files) {
//      println(currdir + File.separator + file)
      val srciso = Source.fromFile((currdir+File.separator+file), "iso8859-1")
      for(ln <- srciso.getLines()) {
        val tokens = ln.split("\\s+")
        for(token <- tokens) {
          (token.replaceAll("[\'\".,?:-]", "").trim.toLowerCase) match {
            case cleantoken if(cleantoken.length() > 0 && !this.stopwords.contains(cleantoken)) => {
             if(!vocabulary.contains(cleantoken)) { vocabulary += (cleantoken -> 0) }
             vocabulary.update(cleantoken, (vocabulary(cleantoken)+1))
             if(!counts.contains(cleantoken)) { counts += (cleantoken -> 0) }
             counts.update(cleantoken, (counts(cleantoken)+1))
             total += 1
            }
            case _ => 
          }
        }
      }
      srciso.close
    }
    
    (counts, total)
  }
  
  def classify(filename: String): String = {
    val results = Map.empty[String, Double]
    for(category <- this.categories) {
      results += (category -> 0)
    }
    val frp = Source.fromFile(filename, "iso8859-1")
    for(line <- frp.getLines()) {
      val tokens = line.split("\\s+");
      for(token <- tokens) {
        (token.replaceAll("[\'\".,?:-]", "").trim.toLowerCase) match {
            case cleantoken if(this.vocabulary.contains(cleantoken)) => {
              for(category <- this.categories) {
                if(prob(category)(cleantoken) == 0) { println(s"$category, $cleantoken") }
                results.update(category, 
                    ( results(category) + scala.math.log(prob(category)(cleantoken)) ) )
              }
            }
            case _ => 
        }
      }
    }
    frp.close()
//    results.foreach(println)
    results.toSeq.sortWith(_._2 > _._2)(0)._1
  }
  
}

object BayesText {
  def test(classifier: BayesText, testdir: String): Unit = {
    val categories = (new File(testdir)).list().filter(fp=>new File(testdir + fp).isDirectory())
    var correct = 0
    var total = 0
    for(category <- categories) {
      print(". ")
      val (catCorrect, catTotal) = BayesText.testCategory(classifier, (testdir+category+File.separator), category)
      correct += catCorrect
      total += catTotal
    }
    println()
    val accuracy = ((correct * 1.0) / total) * 100
    println(s"Accuracy is $accuracy ($total test instances)")
  }
  
  def testCategory(classifier: BayesText, directory: String, category: String): (Int, Int) = {
    var total = 0;
    var correct = 0;
    val infiles = (new File(directory)).list()
    for(infile <- infiles) {
      total += 1
      val result = classifier.classify(directory + infile)
      if(result.equals(category)) {
        correct += 1
      }
    }
    println(s"$correct, $category, $total")
    (correct, total)
  }
}