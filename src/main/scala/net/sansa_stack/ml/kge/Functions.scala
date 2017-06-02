package net.sansa_stack.ml.kge.loss

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{Symbol => s}

/**
  * Created by nilesh on 31/05/2017.
  */
object MaxMarginLoss {
  def apply(margin: Float): (Symbol, Symbol) => Symbol = {
    loss(margin) _
  }

  def loss(margin: Float)(positiveScore: Symbol, negativeScore: Symbol): Symbol = {
    val max = s.max(negativeScore - positiveScore + margin, 0)
    s.sum(name = "sum")()(Map("data" -> max))
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
//    val xs = s.square("square")()(Map("data" -> x))
//    val ys = s.square("square")()(Map("data" -> y))
    val diff = x*x - y*y
    s.sqrt("sqrt")()(Map("data" -> diff))
  }
}

object DotSimilarity {
  def apply(x: Symbol, y: Symbol): Symbol = {
    s.dot("dot")()(Map("lhs" -> x, "rhs" -> y))
  }
}