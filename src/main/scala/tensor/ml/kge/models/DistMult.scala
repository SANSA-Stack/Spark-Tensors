package tensor.ml.kge.models

import scala.math._
import scala.util._

import org.apache.spark.sql._

import com.intel.analytics.bigdl.nn.Power
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

import tensor.ml.kge.dataset.Dataset

class DistMult(train: Dataset, dim: Int, sk: SparkSession) extends Models {

  val batch = 100
  val epochs = 100
  val rate = 0.01f

  var e = initialize(train.s)
  var r = normalize(initialize(train.p))

  var opt = new Adam(learningRate = rate)

  def initialize(data: Array[String]) = {
    Tensor(data.length, dim).randn()
  }

  def normalize(data: Tensor[Float]) = {
    for (i <- 1 to dim)
      data(i) /= data(i).abs().sum()
    data
  }

  import sk.implicits._

  def subset(data: DataFrame) = {
    data.sample(false, 2 * (batch.toDouble / data.count().toDouble)).limit(batch).toDF()
  }

  val seed = new Random(System.currentTimeMillis())

  def tuple(aux: Row) = {

    if (seed.nextBoolean()) {
      (seed.nextInt(train.s.length) + 1, aux.getInt(1), aux.getInt(2))
    } else {
      (aux.getInt(0), aux.getInt(1), seed.nextInt(train.s.length) + 1)
    }
  }

  def generate(data: DataFrame) = {
    data.collect().map(i =>
      tuple(i)).toSeq.toDF()
  }

  def dist(data: DataFrame) = {
    data.collect().map { i =>
      e(i.getInt(0)) * r(i.getInt(1)) * e(i.getInt(2))
    }.reduce((a, b) => a + b).pow(2).sqrt().sum()
  }

  def run() = {

    for (i <- 1 to epochs) {

      e = normalize(e)
      val pos = subset(train.df)
      val neg = generate(pos)

      def delta(x: Tensor[Float]) = {
        (dist(neg) - dist(pos) + 1, x)
      }

      opt.optimize(delta, e)
      opt.optimize(delta, r)
      val err = dist(pos) - dist(neg) + 1
      printf("Epoch: %d: %f\n", i, err)

    }
  }

}