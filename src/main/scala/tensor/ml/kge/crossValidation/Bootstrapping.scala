package tensor.ml.kge.crossValidation

import org.apache.spark.sql._

class Bootstrapping(data: DataFrame, sk: SparkSession) extends CrossValidation {

  import sk.implicits._

  def crossValidation() = {

    val train = data.sample(true, 1).toDF()
    val test = data.except(train).toDF()
    (train, test)
  }

}
