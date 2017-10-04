package tensor.ml.kge.models

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.NDArray._
import ml.dmlc.mxnet.optimizer.AdaGrad

import org.apache.spark.sql._

import scala.util._

import tensor.ml.kge.dataset.Dataset

class TransE(train: Dataset, m: Float, k: Int, sk: SparkSession) extends Models {

  val batch = 100
  val epochs = 100
  val rate = 0.01f

  var e = initialize(train.s.length)
  var l = normalize(initialize(train.p.length))

  val seed = new Random(System.currentTimeMillis())

  def initialize(size: Int) = {
    NDArray.uniform(-6 / Math.sqrt(k), 6 / Math.sqrt(k), (size, k))
  }

  def normalize(data: NDArray) = {
    L2Normalization(data, 0, "instance")
  }

  import sk.implicits._

  def subset(data: DataFrame) = {
    data.sample(false, (batch.toFloat / data.count().toFloat)).toDF()
  }

  def tuple(aux: Row, idx: Boolean) = {

    val rd = seed.nextInt(train.s.length)

    if (idx) {
      (rd + 1, aux.getInt(1), aux.getInt(2))
    } else {
      (aux.getInt(0), aux.getInt(1), rd + 1)
    }
  }

  def generate(data: DataFrame) = {

    var aux = Seq[(Int, Int, Int)]()
    val rnd = seed.nextBoolean()

    for (j <- 1 to data.count().toInt) {
      val i = data.rdd.take(j).last
      aux = tuple(i, rnd) +: aux
    }

    aux.toDF()
  }

  def L1(vec: Row) = {
    e.slice(vec.getInt(0)) + l.slice(vec.getInt(1)) - e.slice(vec.getInt(2))
  }

  def run() = {

    var opt = new AdaGrad(learningRate = rate)

    for (i <- 1 to epochs) {

      e = normalize(e)
      val pos = subset(train.df)
      val neg = generate(pos)
      var err: Float = 0

      for (j <- 1 to pos.count().toInt) {

        val ep = pos.rdd.take(j).last
        val en = neg.rdd.take(j).last

        def delta() = {
          L1(ep) - L1(en)
        }

        if (m + sum(abs(L1(ep))).toScalar > sum(abs(L1(en))).toScalar) {

          opt.update(1, e, delta, e)
          opt.update(1, l, delta, l)

          err += m + sum(abs(L1(ep))).toScalar - sum(abs(L1(en))).toScalar
        }

      }

      printf("Epoch: %d: %f\n", i, err)

    }
  }

}