package tensor.ml.kge.run

import org.apache.spark.sql._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import tensor.ml.kge.dataset.dataframe.Triples
import tensor.ml.kge.crossValidation.Holdout
import tensor.ml.kge.convertor.ByIndex

object TransERun {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val sk = SparkSession.builder.master("local")
    .appName("Tensor").getOrCreate

  def main(args: Array[String]) = {

    val table = new Triples("kge", "/home/lpfgarcia/Desktop/tensor/data/train.txt", sk)
    println(table.triples.show())
    val data = new ByIndex(table.triples, sk).df
    
    println(data.show())
    
    val (train, test) = new Holdout(data, 0.6f).crossValidation()

    println(train.show())
    println(test.show())
            
    var model = new tensor.ml.kge.models.TransE(train, 100, 20, 1, "L1", sk)
    model.run()

    val predict = new tensor.ml.kge.predict.TransE(model, test, sk)
    println(predict)

  }

}