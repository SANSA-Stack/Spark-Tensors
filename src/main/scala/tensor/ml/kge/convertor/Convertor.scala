package tensor.ml.kge.convertor

import org.apache.spark.sql._

abstract class Convertor(data: DataFrame) {

  def numeric() : DataFrame

}