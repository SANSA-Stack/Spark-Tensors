package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.spark.io.LabeledPointIter
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.immutable.ListMap

/**
  * Created by nilesh on 30/6/17.
  */
class UnLabeledPointIter(
   val points: Iterator[LabeledPoint],
   val dimension: Shape,
   val _batchSize: Int,
   val dataName: String = "data")
  extends LabeledPointIter(points, dimension, _batchSize, dataName) {

  override def provideLabel: ListMap[String, Shape] = {
    ListMap()
  }
}
