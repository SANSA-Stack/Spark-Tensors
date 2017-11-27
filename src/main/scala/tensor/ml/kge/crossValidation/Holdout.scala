package tensor.ml.kge.crossValidation

import org.apache.spark.sql._

case class rateException(info: String) extends Exception

class Holdout(data: DataFrame, rate: Float) extends CrossValidation {

  if (rate < 0 || rate >= 1)
    throw new rateException("Rate value should be higher than 0 and lower than 1")

  def crossValidation() = {
    val train = data.sample(false, rate).toDF()
    val test = data.except(train).toDF()
    (train, test)
  }

}