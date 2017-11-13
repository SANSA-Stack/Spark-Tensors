package tensor.ml.kge.models

import scala.math._
import scala.util._

import org.apache.spark.sql._

import com.intel.analytics.bigdl.nn.Power
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

import tensor.ml.kge.dataset.Dataset

class TransE(train: Dataset, m: Float, k: Int, L: String, sk: SparkSession) extends Models {

  val batch = 100
  val epochs = 1000
  val rate = 0.01f

  var e = initialize(train.s)
  var l = initialize(train.p)

  var ept = new Adam(learningRate = rate)
  var lpt = new Adam(learningRate = rate)

  val myL = L match {
    case "L2" => L2 _
    case _    => L1 _
  }

  def initialize(data: Array[String]) = {
    Tensor(data.length, k).rand(-6 / sqrt(k), 6 / sqrt(k))
  }

  def normalize(data: Tensor[Float]) = {
    for (i <- 1 to k)
      data(i) /= data(i).abs().sum()
    data
  }

  import sk.implicits._

  def subset(data: DataFrame) = {
    data.sample(false, batch.toDouble / data.count().toDouble).toDF()
  }

  val seed = new Random(System.currentTimeMillis())

  def tuple(aux: Row, idx: Boolean) = {

    val rd = seed.nextInt(train.s.length) + 1

    if (idx) {
      (rd, aux.getInt(1), aux.getInt(2))
    } else {
      (aux.getInt(0), aux.getInt(1), rd)
    }
  }

  def generate(data: DataFrame) = {

    val rnd = seed.nextBoolean()
    var aux = Seq[(Int, Int, Int)]()

    for (j <- 1 to data.count().toInt) {
      val i = data.rdd.take(j).last
      aux = tuple(i, rnd) +: aux
    }

    aux.toDF()
  }

  def norm(data: DataFrame) = {
    data.collect().map(i =>
      dist(i)).reduce((a, b) => a + b)
  }

  def dist(vec: Row) = {
    e(vec.getInt(0)) + l(vec.getInt(1)) - e(vec.getInt(2))
  }

  def L1(vec: Tensor[Float]) = {
    vec.abs().sum()
  }

  def L2(vec: Tensor[Float]) = {
    vec.pow(2).sqrt().sum()
  }

  def run() = {

    for (i <- 1 to epochs) {

      e = normalize(e)
      l = normalize(l)
      val pos = subset(train.df)
      val neg = generate(pos)

      def delta(x: Tensor[Float]) = {
        (m + myL(norm(pos)) - myL(norm(neg)), x)
      }

      if (m * batch + myL(norm(pos)) > myL(norm(neg))) {

        ept.optimize(delta, e)
        lpt.optimize(delta, l)
        val err = m * batch + myL(norm(pos)) - myL(norm(neg))
        printf("Epoch: %d: %f\n", i, err)
      }


    }
  }

}