package tensor.ml.kge.predict

import org.apache.spark.sql._

import tensor.ml.kge.dataset.Dataset

abstract class Predict(test: Dataset) {

  var left = Seq[Float]()
  var right = Seq[Float]()

  def leftRank(row: Row): Float

  def rightRank(row: Row): Float

  def ranking() = {

    test.df.collect().map { row =>
      left = leftRank(row) +: left
      right = rightRank(row) +: right
    }

    (left, right)
  }

  def meanRanking() {
    (left.sum / left.length,
      right.sum / right.length)
  }

}