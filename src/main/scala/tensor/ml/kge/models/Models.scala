package tensor.ml.kge.models

import scala.math._
import scala.util._

import org.apache.spark.sql._

import com.intel.analytics.bigdl.nn.Power
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

import tensor.ml.kge.dataset.Dataset

abstract class Models(train: Dataset, batch: Int, k: Int, sk: SparkSession) {

  var e = initialize(train.s)
  var r = normalize(initialize(train.p))

  def initialize(data: Array[String]) = {
    Tensor(data.length, k).rand(-6 / sqrt(k), 6 / sqrt(k))
  }

  def normalize(data: Tensor[Float]) = {
    for (i <- 1 to k)
      data(i) /= data(i).abs().sum()
    data
  }

  val seed = new Random(System.currentTimeMillis())

  def tuple(aux: Row) = {

    if (seed.nextBoolean()) {
      (seed.nextInt(train.s.length) + 1, aux.getInt(1), aux.getInt(2))
    } else {
      (aux.getInt(0), aux.getInt(1), seed.nextInt(train.s.length) + 1)
    }
  }

  import sk.implicits._

  def generate(data: DataFrame) = {
    data.collect().map(i =>
      tuple(i)).toSeq.toDF()
  }

  def dist(data: Row) = {
    e(data.getInt(0)) + r(data.getInt(1)) - e(data.getInt(2))
  }

  def subset(data: DataFrame) = {
    data.sample(false, 2 * (batch.toDouble / data.count().toDouble)).limit(batch).toDF()
  }

  def L1(vec: Tensor[Float]) = {
    vec.abs().sum()
  }

  def L2(vec: Tensor[Float]) = {
    vec.pow(2).sqrt().sum()
  }

}