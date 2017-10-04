package tensor.ml.kge.models

import ml.dmlc.mxnet._

import org.apache.spark.sql._ 

object L1 {
  
  def d(vec: Row, e: NDArray, l: NDArray) = {
    e.slice(vec.getInt(0)) + l.slice(vec.getInt(1)) - e.slice(vec.getInt(2))
  }

}