package tensor.ml.kge.NewDesign
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import scala.util.Random
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import com.twitter.chill._
import scala.concurrent.forkjoin.ThreadLocalRandom


class Triples (name: String,
		spark : SparkSession, 
		filePathTriples : String) {
    
  
  def readFromCSVFile(delimiter: String = "\t", 
      header:Boolean = false) : DataFrame = {
    
    return null
  }
  
  
  def getAllDistinctEntities() : RDD[String] = {
    return null
  }
  
  def getAllDistinctPredicates() : RDD[String] = {
    return null
  }
  
  
}