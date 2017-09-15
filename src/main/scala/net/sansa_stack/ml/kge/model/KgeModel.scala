package net.sansa_stack.ml.kge.model

import ml.dmlc.mxnet.module.Module
import ml.dmlc.mxnet.{Symbol => s, _}
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.optimizer.AdaGrad
import ml.dmlc.mxnet.spark.MXNet
import net.sansa_stack.rdf.spark.model.TripleRDD.tripleFunctions
import org.apache.jena.graph.{Triple => JTriple}
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * Created by nilesh on 13/09/2017.
  */
trait KgeModel {
  val numEntities: Int
  val numRelations: Int
  val latentFactors: Int
  val trainBatchSize: Int
  val testBatchSize: Int

  protected val entityWeight = s.Variable("entity_weight")
  protected val relationWeight = s.Variable("relation_weight")

  private val input = s.split()()(Map("data" -> s.Variable("data"), "axis" -> 1, "num_outputs" -> 5))
  protected val head = entityEmbedding(input.get(0))
  protected val relation = relationEmbedding(input.get(1))
  protected val tail = entityEmbedding(input.get(2))
  protected val corruptHead = entityEmbedding(input.get(3))
  protected val corruptTail = entityEmbedding(input.get(4))

  protected def entityEmbedding(id: Symbol): Symbol
  protected def relationEmbedding(id: Symbol): Symbol

  def getLoss: Symbol
  def getScore: Symbol
}

trait NegativeSampler {
  val model: KgeModel

  protected def corruptSample(s: Int, p: Int, o: Int): (Int, Int, Int) = {
    if(Random.nextInt(2) == 0)
      (Random.nextInt(model.numEntities), p, o)
    else
      (s, p, Random.nextInt(model.numEntities))
  }
}

class ModelTrainer(val model: KgeModel, val name: String) extends NegativeSampler {
  private val trainModule = new Module(model.getLoss, contexts=Context.gpu())
  private val testModule = new Module(model.getScore)

  def fit(trainData: DataIter,
          validationData: Option[DataIter] = None,
          numExamples: Int,
          batchSize: Int,
          learningRate: Float,
          weightDecay: Float,
          devs: Array[Context],
          earlyStopping: Boolean = true) = {
    val checkpoint = new EpochEndCallback {
        override def invoke(epoch: Int, symbol: Symbol,
                            argParams: Map[String, NDArray],
                            auxParams: Map[String, NDArray]): Unit = {
          Model.saveCheckpoint(name, epoch + 1, symbol, argParams, auxParams)
        }
      }

    val epochSize = numExamples / batchSize

    trainModule.bind(trainData.provideData)
    trainModule.initParams(new Xavier(factorType = "in", magnitude = 2.34f))
    trainModule.initOptimizer(optimizer = new AdaGrad(learningRate = learningRate, wd = weightDecay))



    trainModule.fit(trainData, validationData)
  }
  def predict(data: DataIter) = {

  }
}