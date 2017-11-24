package tensor.ml.kge.crossValidation

import org.apache.spark.sql._
import org.apache.spark.sql.types.IntegerType

case class kException(info: String) extends Exception

class kFold(data: DataFrame, k: Int, sk: SparkSession) extends CrossValidation {

  import sk.implicits._

  if (k > 1 && k <= 10)
    throw new kException("The k value should be higher than 1 and lower or equal to 10")

  val id = (1 to data.count().toInt / k).flatMap(List.fill(k)(_))

  def crossValidation() = {

    val fold = sk.sparkContext.parallelize(id)
    val tmp = data.rdd.zip(fold).map(r => Row.fromSeq(r._1.toSeq ++ Seq(r._2)))
    val df = sk.createDataFrame(tmp, data.schema.add("k", IntegerType))

    val train = for (i <- 1 to k) yield {
      df.filter($"k" =!= i).toDF()
    }

    val test = for (i <- 1 to k) yield {
      df.filter($"k" === i).toDF()
    }

    (train, test)
  }

}