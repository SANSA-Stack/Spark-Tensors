package tensor.ml.kge.crossValidation

import org.apache.spark.sql._

trait CrossValidation[T] {

  def crossValidation: (T, T)

}