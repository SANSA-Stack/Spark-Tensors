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

trait CrossValidatior {
  def crossValidator(numericallyTransformedData : DataFrame ) : 
      (Seq[DataFrame],Seq[DataFrame])
}