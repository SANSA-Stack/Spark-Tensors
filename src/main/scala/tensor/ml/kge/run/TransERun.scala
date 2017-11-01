package tensor.ml.kge.run

import org.apache.spark.sql._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import tensor.ml.kge.dataset.Dataset
import tensor.ml.kge.models.TransE
import tensor.ml.kge.predict.Evaluate

object TransERun {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val sk = SparkSession.builder.master("local")
    .appName("Tensor").getOrCreate

  def main(args: Array[String]) = {
    
    println("oi")
    val train = new Dataset("train.txt", "\t", "false", sk)
    
    print(train.tb.show)
    
    val model = new TransE(train, 1, 20, "L1", sk)
    println("tchau")
 
    model.run()
    
    val test = new Dataset("test.txt", "\t", "false", sk)
    val predict = new Evaluate(model, test.df, sk)
    
    println(predict)
  
  }
  
}