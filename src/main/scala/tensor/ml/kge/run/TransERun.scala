package tensor.ml.kge.run

import org.apache.spark.sql._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import tensor.ml.kge.dataset.Dataset

object TransERun {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val sk = SparkSession.builder.master("local")
    .appName("Tensor").getOrCreate

  def main(args: Array[String]) = {
    
    val train = new Dataset("train.txt", "\t", "false", sk)    
    val model = new tensor.ml.kge.models.TransE(train, 100, 20, 1, "L1", sk)
    
    model.run()
    
    val test = new Dataset("test.txt", "\t", "false", sk)
    val predict = new tensor.ml.kge.predict.TransE(model, test, sk)
    
    println(predict)
  
  }
  
}