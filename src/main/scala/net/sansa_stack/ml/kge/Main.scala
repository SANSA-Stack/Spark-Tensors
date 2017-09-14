package net.sansa_stack.ml.kge

import ml.dmlc.mxnet.spark.MXNet
import ml.dmlc.mxnet.spark.io.TransEOld
import ml.dmlc.mxnet.{Context, NDArray}

import scala.io.Source


/**
  * Created by nilesh on 19/05/2017.
  */
object Main extends App {
  /**
   * Pretty prints a Scala value similar to its source represention.
   * Particularly useful for case classes.
 *
   * @param a - The value to pretty print.
   * @param indentSize - Number of spaces for each indent.
   * @param maxElementWidth - Largest element size before wrapping.
   * @param depth - Initial depth to pretty print indents.
   * @return
   */
   def prettyPrint(a: Any, indentSize: Int = 2, maxElementWidth: Int = 30, depth: Int = 0): String = {
    val indent = " " * depth * indentSize
    val fieldIndent = indent + (" " * indentSize)
    val thisDepth = prettyPrint(_: Any, indentSize, maxElementWidth, depth)
    val nextDepth = prettyPrint(_: Any, indentSize, maxElementWidth, depth + 1)
    a match {
      // Make Strings look similar to their literal form.
      case s: String =>
        val replaceMap = Seq(
          "\n" -> "\\n",
          "\r" -> "\\r",
          "\t" -> "\\t",
          "\"" -> "\\\""
        )
        '"' + replaceMap.foldLeft(s) { case (acc, (c, r)) => acc.replace(c, r) } + '"'
      // For an empty Seq just use its normal String representation.
      case xs: Seq[_] if xs.isEmpty => xs.toString()
      case xs: Seq[_] =>
        // If the Seq is not too long, pretty print on one line.
        val resultOneLine = xs.map(nextDepth).toString()
        if (resultOneLine.length <= maxElementWidth) return resultOneLine
        // Otherwise, build it with newlines and proper field indents.
        val result = xs.map(x => s"\n$fieldIndent${nextDepth(x)}").toString()
        result.substring(0, result.length - 1) + "\n" + indent + ")"
      // Product should cover case classes.
      case p: Product =>
        val prefix = p.productPrefix
        // We'll use reflection to get the constructor arg names and values.
        val cls = p.getClass
        val fields = cls.getDeclaredFields.filterNot(_.isSynthetic).map(_.getName)
        val values = p.productIterator.toSeq
        // If we weren't able to match up fields/values, fall back to toString.
        if (fields.length != values.length) return p.toString
        fields.zip(values).toList match {
          // If there are no fields, just use the normal String representation.
          case Nil => p.toString
          // If there is just one field, let's just print it as a wrapper.
          case (_, value) :: Nil => s"$prefix(${thisDepth(value)})"
          // If there is more than one field, build up the field names and values.
          case kvps =>
            val prettyFields = kvps.map { case (k, v) => s"$fieldIndent$k = ${nextDepth(v)}" }
            // If the result is not too long, pretty print on one line.
            val resultOneLine = s"$prefix(${prettyFields.mkString(", ")})"
            if (resultOneLine.length <= maxElementWidth) return resultOneLine
            // Otherwise, build it with newlines and proper field indents.
            s"$prefix(\n${prettyFields.mkString(",\n")}\n$indent)"
        }
      // If we haven't specialized this type, just use its toString.
      case _ => a.toString
    }
  }
//  import ml.dmlc.mxnet.Symbol
//  import ml.dmlc.mxnet._
//  val x = Symbol.Variable("x")
//  val y = Symbol.Variable("y")
//  val diff = Symbol.pow(x, 2) + Symbol.pow(y, 3)
//  val a = NDArray.ones(1) * 10
//  val b = NDArray.ones(1) * 2
//  val ga = NDArray.empty(1)
//  val ga2 = NDArray.empty(1)
//  val executor = diff.bind(Context.cpu(), args=Map("x" -> a, "y" -> b), argsGrad=Map("x" -> ga, "y" -> ga2))
//  executor.forward()
//  println(executor.outputs(0).toArray.mkString(","))
////  executor.
//
//  // test gradient
//  val outGrad = NDArray.ones(1)
//  executor.backward(Array(outGrad))
//  println(executor.gradDict.toArray.apply(1).x._2.toArray.mkString(","))

  import ml.dmlc.mxnet.{Symbol => s}
//  val x = s.Variable("x")
//  val embedWeight = s.Variable("weight")

//  val entityEmbeddings = s.Embedding("embed")()(Map(
//  "data" -> x,
//  "input_dim" -> 100,
//  "output_dim" -> 50,
//  "name" -> "embed"
//  ))

//  val embed = s.Embedding("embed")()(Map("data" -> x, "input_dim" -> 100,
//    "weight" -> embedWeight, "output_dim" -> 50, "name"->"embed"))
//
//
//  val a = NDArray.ones(1) * 10
//  val exec = embed.simpleBind(Context.cpu(), shapeDict=Map())
//  exec.forward(isTrain = true)

//  val exec = embed.bind(Context.cpu(), args=Map("x" -> a))
//  println(exec.outputs(0).toArray.mkString(","))

  val model = new TransEOld(40943, 18, 100, 14200, 100)
//  val model = new TransE(262928, 19, 100, 18683, 100)
//  val model = new TransE(262928, 47, 100, 100, 100, true)
  model.train()

}
