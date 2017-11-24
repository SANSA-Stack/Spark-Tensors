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
import tensor.ml.kge.NewDesign.StringRecord

abstract class Convertor {
  
  /**
   * @param stringDataFrame This is the input of string DataFrame (test+training).
   * @return A numerical DataFrame which is converted from the input.
   */
  def convertor( stringDataFrame : DataFrame[StringRecord]) :
   DataFrame[NumericRecord]

      
  def inverseConvertor( numericDataFrame : DataFrame) :
   DataFrame /* a string DataFrame */
   

  def negativeSampler( numericDataFrame : DataFrame) : 
   DataFrame
   
  def compareEntitiesSimilarities() : Any /* should be discussed */
  
  
  
   
}