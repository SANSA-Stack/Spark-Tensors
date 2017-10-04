package tensor.ml.kge.predict

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.NDArray._

import org.apache.spark.sql._

import tensor.ml.kge.dataset._
import tensor.ml.kge.models._

class Evaluate(model: TransE, test: DataFrame, sk: SparkSession) extends Predict(test: DataFrame) {

  def L1(vec: Row) = {
    model.e.slice(vec.getInt(0)) + model.l.slice(vec.getInt(1)) - model.e.slice(vec.getInt(2))
  }
    
  def head(i: Int, r: Row) = {
    Row(i, r.getInt(1), r.getInt(2))
  }

  def leftRank(row: Row) = {

    var x: Seq[Float] = List()
    val y = sum(abs(L1(row))).toScalar

    x = y +: x
    for (i <- 1 until test.count().toInt) {
      x = sum(abs(L1(head(i, row)))).toScalar +: x
    }

    x.sorted.indexOf(y)
  }

  def tail(i: Int, r: Row) = {
    Row(r.getInt(0), r.getInt(1), i)
  }

  def rightRank(row: Row) = {

    var x: Seq[Float] = List()
    val y = sum(abs(L1(row))).toScalar

    x = y +: x
    for (i <- 1 until test.count().toInt) {
      x = sum(abs(L1(tail(i, row)))).toScalar +: x
    }

    x.sorted.indexOf(y)
  }

}