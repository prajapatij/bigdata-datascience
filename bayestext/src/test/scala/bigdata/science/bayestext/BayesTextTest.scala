package bigdata.science.bayestext

import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import java.io.File

@RunWith(classOf[JUnitRunner])
class BayesTextTest  extends FunSuite {
  val baseDirectory = "src/test/data/20news-bydate/"  
  val trainingDir = baseDirectory + "20news-bydate-train/"
  val testDir = baseDirectory + "20news-bydate-test/"
  val samplecat = "src/test/data/20news-bydate/20news-bydate-test/misc.forsale/77014"
  
  test("Verify result accuracy with 0 stop words") {
    val testobj = new BayesText(trainingDir, (baseDirectory + "stopwords0.txt"))
    BayesText.test(testobj, testDir)    
  }
  
  test("Verify result accuracy with 174 stop words") {
    val testobj = new BayesText(trainingDir, (baseDirectory + "stopwords174.txt"))
    BayesText.test(testobj, testDir)    
  }
  
  test("Find 1 Sample category with 0 stop words") {
    val testobj = new BayesText(trainingDir, (baseDirectory + "stopwords0.txt"))
    val samplecatval = testobj.classify(new File(samplecat).getAbsolutePath)
    println("Category found = " + samplecatval)
    assert(samplecatval.equals("misc.forsale"))
  }
  
  test("Find 1 Sample category with 174 stop words") {
    val testobj = new BayesText(trainingDir, (baseDirectory + "stopwords174.txt"))
    val samplecatval = testobj.classify(new File(samplecat).getAbsolutePath)
    println("Category found = " + samplecatval)
    assert(samplecatval.equals("misc.forsale"))
  }
  
  ignore("Test replace regex") {
    println( "\"\",.abcd.?:-".replaceAll("[\'\".,?:-]", "") )
  }
}