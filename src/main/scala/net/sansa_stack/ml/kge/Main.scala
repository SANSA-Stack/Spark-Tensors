package net.sansa_stack.ml.kge

import ml.dmlc.mxnet.spark.MXNet


/**
  * Created by nilesh on 19/05/2017.
  */
object Main extends App {
//  import ml.dmlc.mxnet.Symbol
  import ml.dmlc.mxnet._
  val x = Symbol.Variable("x")
  val y = Symbol.Variable("y")
  val diff = Symbol.pow(x, 2) + Symbol.pow(y, 3)
  val a = NDArray.ones(1) * 10
  val b = NDArray.ones(1) * 2
  val ga = NDArray.empty(1)
  val ga2 = NDArray.empty(1)
  val executor = diff.bind(Context.cpu(), args=Map("x" -> a, "y" -> b), argsGrad=Map("x" -> ga, "y" -> ga2))
  executor.forward()
  println(executor.outputs(0).toArray.mkString(","))
//  executor.

  // test gradient
  val outGrad = NDArray.ones(1)
  executor.backward(Array(outGrad))
  println(executor.gradDict.toArray.apply(1).x._2.toArray.mkString(","))

}
