package tensor.ml.kge.crossValidation

import org.apache.spark.sql._

class Bootstrapping(data: DataFrame) extends CrossValidation {

  def crossValidation() = {
    val train = data.sample(true, 1).toDF()
    val test = data.except(train).toDF()
    (train, test)
  }

}
