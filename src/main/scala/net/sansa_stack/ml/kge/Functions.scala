package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.io.NDArrayIter
import ml.dmlc.mxnet.spark.io.{LabeledPointIter, PointIter}
import ml.dmlc.mxnet.{Symbol => s}

import scala.collection.mutable

/**
  * Created by nilesh on 31/05/2017.
  */
object MaxMarginLoss {
  def apply(margin: Float): (Symbol, Symbol) => Symbol = {
    loss(margin) _
  }

  def loss(margin: Float)(positiveScore: Symbol, negativeScore: Symbol): Symbol = {
    var loss = s.max(negativeScore - positiveScore + margin, 0)
//    loss = s.sum(name = "sum")()(Map("data" -> loss, "axis" -> Shape(0)))
    s.make_loss(name = "loss")()(Map("data" -> loss))
  }
}

object Sigmoid {
  def apply(x: Symbol): Symbol = {
    s.Activation(name = "sigmoid")()(Map("data" -> x, "act_type" -> "sigmoid"))
  }
}

object Tanh {
  def apply(x: Symbol): Symbol = {
    s.Activation(name = "tanh")()(Map("data" -> x, "act_type" -> "tanh"))
  }
}

object L2Similarity {
  def apply(x: Symbol, y: Symbol): Symbol = {
    val difference = x - y
    var score = s.square()()(Map("data" -> difference))
    score = s.sum("sum2")()(Map("data" -> score, "axis" -> Shape(2)))
    score*(-1.0)
  }
}

object DotSimilarity {
  def apply(x: Symbol, y: Symbol): Symbol = {
    s.dot("dot")()(Map("lhs" -> x, "rhs" -> y))
  }
}

//object Hits {
//  def hitsAt1(label: NDArray, predicted: NDArray): Float = {
//    val labelA = label.toArray
//    val predA = predicted.toArray
//    labelA.zip(predA).map(x => if(x._1.toInt == x._2.toInt && x._1.toInt == 1) 1 else 0).sum
//  }
//}
//
class Hits(model: FeedForward, testAxis: Int, batchSize: Int, testWith: NDArray, testData: DataIter) {
  private val numItems = testWith.shape(testWith.shape.length - 1)
  private val groundIter = testData.flatMap(_.data).toSeq

  private val predictIter = groundIter.map {
      triples: NDArray =>
        val repeatOriginal = NDArray.repeat(Map("repeats" -> numItems, "axis" -> 0))(triples).get.T
        val repeatTest = NDArray.tile(Map("reps" -> (1, batchSize)))(testWith.reshape(Shape(1, numItems))).get
        repeatOriginal.slice(testAxis).set(repeatTest)
        repeatOriginal.T
  }

  private val groundAxis = groundIter.map(_.T.slice(testAxis).reshape(Shape(batchSize, 1)))

  private val sortedPredict = predictIter.flatMap {
    batch: NDArray =>
      model.predict(new NDArrayIter(IndexedSeq(batch), dataBatchSize=batchSize)).map{
        predictions: NDArray =>
          val scores = predictions.reshape(Shape(batchSize, numItems))
          val sorted = NDArray.argsort(Map("is_ascend" -> false))(scores).get.T
          sorted
      }
  }

  def hits(hitsAt: Int*): Seq[Float] = {
    val hits = (groundAxis zip sortedPredict) map {
      case (ground, predict) =>
        val hits =
          hitsAt.map {
          topK: Int =>
            val topKPredictions = predict.slice(0, topK).T
            val hits = NDArray.broadcast_equal(ground, topKPredictions)
            NDArray.sum(hits).toScalar / batchSize
        }.toArray
        NDArray.array(hits, Shape(1,hitsAt.size))
    }

    val sum = hits.reduce(_ + _) / hits.size.toFloat
    sum.toArray.toSeq
  }

  def mrr: Float = {
    val mrr = (groundAxis zip sortedPredict) map {
      case (ground, predict) =>
        val hits = NDArray.broadcast_equal(ground, predict.T)
        println(hits.shape)
        val ranks = NDArray.argmax(Map("axis" -> 1))(hits) + 1f
        val mrr = NDArray.power(ranks, -1f)
        NDArray.sum(mrr).toScalar / batchSize
    }

    mrr.sum / mrr.size
  }
}